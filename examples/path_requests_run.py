#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
path_requests_run.py
====================

Reads a JSON request file in accordance with the Yang model
for requesting path computation and returns path results in terms
of path and feasibilty.

See: draft-ietf-teas-yang-path-computation-01.txt
"""

from sys import exit
from argparse import ArgumentParser
from pathlib import Path
from collections import namedtuple
from logging import getLogger, basicConfig, CRITICAL, DEBUG, INFO
from json import dumps, loads
from networkx import (draw_networkx_nodes, draw_networkx_edges,
                      draw_networkx_labels)
from numpy import mean
from gnpy.core.service_sheet import convert_service_sheet, Request_element, Element
from gnpy.core.utils import load_json
from gnpy.core.network import load_network, build_network, save_network
from gnpy.core.equipment import load_equipment, trx_mode_params, automatic_nch, automatic_spacing
from gnpy.core.elements import Transceiver, Roadm, Edfa, Fused, Fiber
from gnpy.core.utils import db2lin, lin2db
from gnpy.core.request import (Path_request, Result_element, compute_constrained_path,
                              propagate, jsontocsv, Disjunction, compute_path_dsjctn, requests_aggregation,
                              propagate_and_optimize_mode, blocking_nopath, find_reversed_path)
from gnpy.core.spectrum_assignment import build_OMS_list, reversed_OMS, spectrum_selection , pth_assign_spectrum
from copy import copy, deepcopy
from textwrap import dedent
from math import ceil
import time

#EQPT_LIBRARY_FILENAME = Path(__file__).parent / 'eqpt_config.json'

logger = getLogger(__name__)

parser = ArgumentParser(description = 'A function that computes performances for a list of services provided in a json file or an excel sheet.')
parser.add_argument('network_filename', nargs='?', type = Path, default= Path(__file__).parent / 'meshTopologyExampleV2.xls', help='input topology file in xls or json')
parser.add_argument('service_filename', nargs='?', type = Path, default= Path(__file__).parent / 'meshTopologyExampleV2.xls', help='input service file in xls or json')
parser.add_argument('eqpt_filename', nargs='?', type = Path, default=Path(__file__).parent / 'eqpt_config.json', help='input equipment library in json. Default is eqpt_config.json')
parser.add_argument('-bi', '--bidir', action='store_true', help='considers that all demands are bidir')
parser.add_argument('-v', '--verbose', action='count', default=0, help='increases verbosity for each occurence')
parser.add_argument('-o', '--output', type = Path)


def requests_from_json(json_data,equipment):
    requests_list = []

    for req in json_data['path-request']:
        # init all params from request
        params = {}
        params['request_id'] = req['request-id']
        params['source'] = req['source']
        params['destination'] = req['destination']
        params['trx_type'] = req['path-constraints']['te-bandwidth']['trx_type']
        params['trx_mode'] = req['path-constraints']['te-bandwidth']['trx_mode']
        params['format'] = params['trx_mode']
        params['spacing'] = req['path-constraints']['te-bandwidth']['spacing']
        try :
            nd_list = req['explicit-route-objects']['route-object-include-exclude']
        except KeyError:
            nd_list = []
        params['nodes_list'] = [n['num-unnum-hop']['node-id'] for n in nd_list]
        params['loose_list'] = [n['num-unnum-hop']['hop-type'] for n in nd_list]
        # recover trx physical param (baudrate, ...) from type and mode
        # in trx_mode_params optical power is read from equipment['SI']['default'] and
        # nb_channel is computed based on min max frequency and spacing
        trx_params = trx_mode_params(equipment,params['trx_type'],params['trx_mode'],True)
        params.update(trx_params)
        # print(trx_params['min_spacing'])
        # optical power might be set differently in the request. if it is indicated then the 
        # params['power'] is updated
        try:
            if req['path-constraints']['te-bandwidth']['output-power']:
                params['power'] = req['path-constraints']['te-bandwidth']['output-power']
        except KeyError:
            pass
        # same process for nb-channel
        f_min = params['f_min']
        f_max_from_si = params['f_max']
        try:
            if req['path-constraints']['te-bandwidth']['max-nb-of-channel'] is not None :
                nch = req['path-constraints']['te-bandwidth']['max-nb-of-channel'] 
                params['nb_channel'] = nch         
                spacing = params['spacing']
                params['f_max'] = f_min + nch*spacing
            else :
                params['nb_channel'] = automatic_nch(f_min,f_max_from_si,params['spacing'])
        except KeyError:
            params['nb_channel'] = automatic_nch(f_min,f_max_from_si,params['spacing'])
        consistency_check(params, f_max_from_si)

        try :
            params['path_bandwidth'] = req['path-constraints']['te-bandwidth']['path_bandwidth']
        except KeyError:
            pass
        requests_list.append(Path_request(**params))
    return requests_list

def consistency_check(params, f_max_from_si):
    f_min = params['f_min']
    f_max = params['f_max']
    max_recommanded_nb_channels = automatic_nch(f_min,f_max,
                params['spacing'])
    if params['baud_rate'] is not None:
        #implicitely means that a mode is defined with min_spacing
        if params['min_spacing']>params['spacing'] : 
            msg = f'Request {params["request_id"]} has spacing below transponder {params["trx_type"]}'+\
                f' {params["trx_mode"]} min spacing value {params["min_spacing"]*1e-9}GHz.\n'+\
                'Computation stopped'
            print(msg)
            logger.critical(msg)
            exit()
        if f_max>f_max_from_si:
            msg = dedent(f'''
            Requested channel number {params["nb_channel"]}, baud rate {params["baud_rate"]} GHz and requested spacing {params["spacing"]*1e-9}GHz 
            is not consistent with frequency range {f_min*1e-12} THz, {f_max*1e-12} THz, min recommanded spacing {params["min_spacing"]*1e-9}GHz.
            max recommanded nb of channels is {max_recommanded_nb_channels}
            Computation stopped.''')
            logger.critical(msg)
            exit()    


def disjunctions_from_json(json_data):
    disjunctions_list = []
    try:
        for snc in json_data['synchronization']:
            params = {}
            params['disjunction_id'] = snc['synchronization-id']
            params['relaxable'] = snc['svec']['relaxable']
            params['link_diverse'] = 'link' in snc['svec']['disjointness']
            params['node_diverse'] = 'node' in snc['svec']['disjointness']
            params['disjunctions_req'] = snc['svec']['request-id-number']
            disjunctions_list.append(Disjunction(**params))
        print(disjunctions_list)
    except KeyError:
        pass
    return disjunctions_list


def load_requests(filename,eqpt_filename):
    if filename.suffix.lower() == '.xls':
        logger.info('Automatically converting requests from XLS to JSON')
        json_data = convert_service_sheet(filename,eqpt_filename)
    else:
        with open(filename, encoding='utf-8') as f:
            json_data = loads(f.read())
    return json_data


def compute_path_with_disjunction(network, equipment, pathreqlist, pathlist):
    
    # use a list but a dictionnary might be helpful to find path based on request_id
    # TODO change all these req, dsjct, res lists into dict !
    path_res_list = []

    for i,pathreq in enumerate(pathreqlist):

        # use the power specified in requests but might be different from the one specified for design
        # the power is an optional parameter for requests definition
        # if optional, use the one defines in eqt_config.json
        p_db = lin2db(pathreq.power*1e3)
        p_total_db = p_db + lin2db(pathreq.nb_channel)
        print(f'request {pathreq.request_id}')
        print(f'Computing path from {pathreq.source} to {pathreq.destination}')
        print(f'with path constraint: {[pathreq.source]+pathreq.nodes_list}') #adding first node to be clearer on the output

        total_path = deepcopy(pathlist[i])
        print(f'Computed path (roadms):{[e.uid for e in total_path  if isinstance(e, Roadm)]}')
        # for debug
        # print(f'{pathreq.baud_rate}   {pathreq.power}   {pathreq.spacing}   {pathreq.nb_channel}')
        if total_path :
            if pathreq.baud_rate is not None:
                # means that at this point the mode was entered/forced by user and thus a baud_rate was defined
                total_path = propagate(total_path,pathreq,equipment, show=False)
                temp_snr01nm = round(mean(total_path[-1].snr+lin2db(pathreq.baud_rate/(12.5e9))),2)
                if temp_snr01nm < pathreq.OSNR :
                    msg = f'\tWarning! Request {pathreq.request_id} computed path from {pathreq.source} to {pathreq.destination} does not pass with {pathreq.tsp_mode}\n' +\
                    f'\tcomputedSNR in 0.1nm = {temp_snr01nm} - required osnr {pathreq.OSNR}'
                    print(msg)
                    logger.warning(msg)
                    # total_path = []
                    pathreq.blocking_reason = 'MODE_NOT_FEASIBLE'
            else:
                total_path,mode = propagate_and_optimize_mode(total_path,pathreq,equipment)
                # if no baudrate satisfies spacing, no mode is returned and the last explored mode
                # a warning is shown in the propagate_and_optimize_mode
                # propagate_and_optimize_mode function returns the mode with the highest bitrate
                # that passes. if no mode passes, then a attribute blocking_reason is added on pathreq
                # that contains the reason for blocking: 'NO_PATH', 'NO_FEASIBLE_MODE', ...
                try:
                    if pathreq.blocking_reason in blocking_nopath :
                        total_path = []
                    elif pathreq.blocking_reason in blocking_nomode :
                        pathreq.baud_rate = mode['baud_rate']
                        pathreq.tsp_mode = mode['format']
                        pathreq.format = mode['format']
                        pathreq.OSNR = mode['OSNR']
                        pathreq.tx_osnr = mode['tx_osnr']
                        pathreq.bit_rate = mode['bit_rate']
                    # other blocking reason should not appear at this point                                
                except:
                    pathreq.baud_rate = mode['baud_rate']
                    pathreq.tsp_mode = mode['format']
                    pathreq.format = mode['format']
                    pathreq.OSNR = mode['OSNR']
                    pathreq.tx_osnr = mode['tx_osnr']
                    pathreq.bit_rate = mode['bit_rate']
        # we record the last tranceiver object in order to have th whole 
        # information about spectrum. Important Note: since transceivers 
        # attached to roadms are actually logical elements to simulate
        # performance, several demands having the same destination may use 
        # the same transponder for the performance simaulation. This is why 
        # we use deepcopy: to ensure each propagation is recorded and not 
        # overwritten 
        else:
            msg = 'Total path is empty. No propagation'
            print(msg)
            logger.info(msg)

        path_res_list.append(total_path)
        print('')
    return path_res_list

def correct_route_list(network, pathreqlist):
    # prepares the format of route list of nodes to be consistant
    # remove wrong names, remove endpoints
    # also correct source and destination
    anytype = [n.uid for n in network.nodes()]
    # TODO there is a problem of identification of fibers in case of parallel fibers bitween two adjacent roadms
    # so fiber constraint is not supported
    transponders = [n.uid for n in network.nodes() if isinstance(n, Transceiver)]
    for pathreq in pathreqlist:
        for i,n_id in enumerate(pathreq.nodes_list):
            # replace possibly wrong name with a formated roadm name
            # print(n_id)
            if n_id not in anytype :
                nodes_suggestion = [uid for uid in anytype \
                    if n_id.lower() in uid.lower()]
                if pathreq.loose_list[i] == 'LOOSE':
                    if len(nodes_suggestion)>0 :
                        new_n = nodes_suggestion[0]
                        print(f'invalid route node specified:\
                        \n\'{n_id}\', replaced with \'{new_n}\'')
                        pathreq.nodes_list[i] = new_n
                    else:
                        print(f'\x1b[1;33;40m'+f'invalid route node specified \'{n_id}\', could not use it as constraint, skipped!'+'\x1b[0m')
                        pathreq.nodes_list.remove(n_id)
                        pathreq.loose_list.pop(i)
                else:
                    msg = f'\x1b[1;33;40m'+f'could not find node : {n_id} in network topology. Strict constraint can not be applied.'+'\x1b[0m'
                    logger.critical(msg)
                    raise ValueError(msg)
        if pathreq.source not in transponders:
            msg = f'\x1b[1;31;40m'+f'Request: {pathreq.request_id}: could not find transponder source : {pathreq.source}.'+'\x1b[0m'
            logger.critical(msg)
            print(f'{msg}\nComputation stopped.')
            exit()
            
        if pathreq.destination not in transponders:
            msg = f'\x1b[1;31;40m'+f'Request: {pathreq.request_id}: could not find transponder destination : {pathreq.destination}.'+'\x1b[0m'
            logger.critical(msg)
            print(f'{msg}\nComputation stopped.')
            exit()

        # TODO remove endpoints from this list in case they were added by the user in the xls or json files
    return pathreqlist

def correct_disjn(disjn):
    local_disjn = disjn.copy()
    for el in local_disjn:
        for d in local_disjn:
            if set(el.disjunctions_req) == set(d.disjunctions_req) and\
             el.disjunction_id != d.disjunction_id:
                local_disjn.remove(d)
    return local_disjn


def path_result_json(pathresult):
    data = {
        'response': [n.json for n in pathresult]
    }
    return data


if __name__ == '__main__':
    start = time.time()
    args = parser.parse_args()
    basicConfig(level={2: DEBUG, 1: INFO, 0: CRITICAL}.get(args.verbose, DEBUG))
    logger.info(f'Computing path requests {args.service_filename} into JSON format')
    print('\x1b[1;34;40m'+f'Computing path requests {args.service_filename} into JSON format'+ '\x1b[0m')
    # for debug
    # print( args.eqpt_filename)
    data = load_requests(args.service_filename,args.eqpt_filename)
    equipment = load_equipment(args.eqpt_filename)
    network = load_network(args.network_filename,equipment)

    # Build the network once using the default power defined in SI in eqpt config
    # TODO power density : db2linp(ower_dbm": 0)/power_dbm": 0 * nb channels as defined by
    # spacing, f_min and f_max 
    p_db = equipment['SI']['default'].power_dbm
    
    p_total_db = p_db + lin2db(automatic_nch(equipment['SI']['default'].f_min,\
        equipment['SI']['default'].f_max, equipment['SI']['default'].spacing))
    build_network(network, equipment, p_db, p_total_db)
    save_network(args.network_filename, network)

    oms_list = build_OMS_list(network,equipment)
    # print(len(oms_list))
    # print(oms_list)
    # print(len(network.nodes))
    # susu = 0
    # for o in oms_list:
    #     susu += len(o.el_id_list) -2
    # print(susu)
    # verification :  susu = len(network.nodes) -roadms -transceivers

    rqs = requests_from_json(data, equipment)

    # check that request ids are unique. Non unique ids, may 
    # mess the computation : better to stop the computation
    all_ids = [r.request_id for r in rqs]
    if len(all_ids) != len(set(all_ids)):
        for a in list(set(all_ids)):
            all_ids.remove(a)
        msg = f'Requests id {all_ids} are not unique'
        logger.critical(msg)
        exit()
    rqs = correct_route_list(network, rqs)

    # pths = compute_path(network, equipment, rqs)
    dsjn = disjunctions_from_json(data)

    print('\x1b[1;34;40m'+f'List of disjunctions'+ '\x1b[0m')
    print(dsjn)
    # need to warn or correct in case of wrong disjunction form
    # disjunction must not be repeated with same or different ids
    dsjn = correct_disjn(dsjn)
        
    # Aggregate demands with same exact constraints
    print('\x1b[1;34;40m'+f'Aggregating similar requests'+ '\x1b[0m')

    rqs,dsjn = requests_aggregation(rqs,dsjn)
    # TODO export novel set of aggregated demands in a json file

    print('\x1b[1;34;40m'+'The following services have been requested:'+ '\x1b[0m')
    print(rqs)
    
    print('\x1b[1;34;40m'+f'Computing all paths with constraints'+ '\x1b[0m')
    pths = compute_path_dsjctn(network, equipment, rqs, dsjn)

    print('\x1b[1;34;40m'+f'Propagating on selected path'+ '\x1b[0m')
    propagatedpths = compute_path_with_disjunction(network, equipment, rqs, pths)
    # Note that deepcopy used in compute_path_with_disjunction retrns a list of nodes which are not belonging to 
    # network (they are copies of the node objects). so there can not be propagation on these nodes.

    # propagate on reversed path only if bidir option is activated (takes time)
    reversed_propagatedpths = []
    reversed_pths =[]
    if args.bidir:
        print('\x1b[1;34;40m'+f'Propagating on selected path reversed direction'+ '\x1b[0m')
    for i,p in enumerate(pths) :   
        if propagatedpths[i] and args.bidir : 
            rp = find_reversed_path(p)
            print(f'request {rqs[i].request_id}')
            print(f'Propagating path from {rqs[i].destination} to {rqs[i].source}')
            print(f'Path (roadsm) {[r.uid for r in rp if isinstance(r,Roadm)]}\n')
            reversed_pths.append(find_reversed_path(p))
            reversed_path = propagate(rp , rqs[i], equipment, show=False)
            reversed_propagatedpths.append(deepcopy(reversed_path))
        else:
            reversed_propagatedpths.append([])    
            reversed_pths.append([])


    pth_assign_spectrum(pths, rqs, oms_list,reversed_pths)

    end = time.time()
    print(f'computation time {end-start}')
    print('\x1b[1;34;40m'+f'Result summary'+ '\x1b[0m')
    
    header = ['req id', '  demand','  snr@bandwidth A-Z (Z-A)','  snr@0.1nm A-Z (Z-A)','  Receiver minOSNR', '  mode', '  Gbit/s' , '  nb of tsp pairs', 'N,M or blocking reason']
    data = []
    data.append(header)
    for i, p in enumerate(propagatedpths):
        rp = reversed_propagatedpths[i]
        if rp and p:
            psnrb = f'{round(mean(p[-1].snr),2)} ({round(mean(rp[-1].snr),2)})'
            psnr = f'{round(mean(p[-1].snr+lin2db(rqs[i].baud_rate/(12.5e9))),2)} ({round(mean(rp[-1].snr+lin2db(rqs[i].baud_rate/(12.5e9))),2)})'
        elif p :
            psnrb = f'{round(mean(p[-1].snr),2)}'
            psnr = f'{round(mean(p[-1].snr+lin2db(rqs[i].baud_rate/(12.5e9))),2)}'

        try :
            if rqs[i].blocking_reason == 'NO_PATH':
                line = [f'{rqs[i].request_id}',f' {rqs[i].source} to {rqs[i].destination} :', f'-',\
                f'-',f'-', f'{rqs[i].tsp_mode}' , f'{round(rqs[i].path_bandwidth * 1e-9,2)}' , f'-', f'NO_PATH']
            elif rqs[i].blocking_reason == 'NO_COMPUTED_SNR':
                line = [f'{rqs[i].request_id}',f' {rqs[i].source} to {rqs[i].destination} :', f'-',\
                f'-',f'-', f'{rqs[i].tsp_mode}' , f'{round(rqs[i].path_bandwidth * 1e-9,2)}' , f'-', f'NO_COMPUTED_SNR']
            elif rqs[i].blocking_reason == 'NO_FEASIBLE_BAUDRATE_WITH_SPACING':
                line = [f'{rqs[i].request_id}',f' {rqs[i].source} to {rqs[i].destination} :', f'-',\
                f'-',f'-', f'{rqs[i].tsp_mode}' , f'{round(rqs[i].path_bandwidth * 1e-9,2)}' , f'-', f'NO_FEASIBLE_BAUDRATE_WITH_SPACING']
            elif rqs[i].blocking_reason == 'NO_PATH_WITH_CONSTRAINT':
                line = [f'{rqs[i].request_id}',f' {rqs[i].source} to {rqs[i].destination} :', f'-',\
                f'-',f'-', f'{rqs[i].tsp_mode}' , f'{round(rqs[i].path_bandwidth * 1e-9,2)}' , f'-', f'NO_PATH_WITH_CONSTRAINT']
            elif rqs[i].blocking_reason == 'NO_FEASIBLE_MODE':
                line = [f'{rqs[i].request_id}', f' {rqs[i].source} to {rqs[i].destination} : ', psnrb,\
                psnr,\
                f'-', f'{rqs[i].tsp_mode}' , f'{round(rqs[i].path_bandwidth * 1e-9,2)}' , f'-', f'NO_FEASIBLE_MODE']           
            elif rqs[i].blocking_reason == 'MODE_NOT_FEASIBLE':
                line = [f'{rqs[i].request_id}', f' {rqs[i].source} to {rqs[i].destination} : ', psnrb,\
                psnr,\
                f'-', f'{rqs[i].tsp_mode}' , f'{round(rqs[i].path_bandwidth * 1e-9,2)}' , f'-', f'MODE_NOT_FEASIBLE']    
            elif rqs[i].blocking_reason == 'NO_SPECTRUM':
                line = [f'{rqs[i].request_id}', f' {rqs[i].source} to {rqs[i].destination} : ', psnrb,\
                psnr,\
                f'-', f'{rqs[i].tsp_mode}' , f'{round(rqs[i].path_bandwidth * 1e-9,2)}' , f'-', f'NO_SPECTRUM']
   
        except AttributeError:
            line = [f'{rqs[i].request_id}', f' {rqs[i].source} to {rqs[i].destination} : ', psnrb,\
                psnr,\
                f'{rqs[i].OSNR}', f'{rqs[i].tsp_mode}' , f'{round(rqs[i].path_bandwidth * 1e-9,2)}' ,\
                f'{ceil(rqs[i].path_bandwidth / rqs[i].bit_rate) }', f'({rqs[i].N},{rqs[i].M})']
        data.append(line)
    
    col_width = max(len(word) for row in data for word in row[2:])   # padding
    firstcol_width = max(len(row[0]) for row in data )   # padding
    secondcol_width = max(len(row[1]) for row in data )   # padding
    for row in data:
        firstcol = ''.join(row[0].ljust(firstcol_width)) 
        secondcol = ''.join(row[1].ljust(secondcol_width))
        remainingcols = ''.join(word.center(col_width,' ') for word in row[2:])
        print(f'{firstcol} {secondcol} {remainingcols}')
    print('\x1b[1;33;40m'+f'Result summary shows mean SNR and OSNR (average over all channels)'+ '\x1b[0m')

    if args.output :
        result = []
        # assumes that list of rqs and list of propgatedpths have same order
        for i,p in enumerate(propagatedpths):
            result.append(Result_element(rqs[i],p))
            # temporary return: add bidir paths in the list .
            # TODO : add a container with reversed properties only ; reversed path is implicit, knowing the topology
            # if reversed_propagatedpths[i]:
            #     result.append(Result_element(rqs[i],reversed_propagatedpths[i]))
        temp = path_result_json(result)
        fnamecsv = f'{str(args.output)[0:len(str(args.output))-len(str(args.output.suffix))]}.csv'
        fnamejson = f'{str(args.output)[0:len(str(args.output))-len(str(args.output.suffix))]}.json'
        with open(fnamejson, 'w', encoding='utf-8') as f:
            f.write(dumps(path_result_json(result), indent=2, ensure_ascii=False))
            with open(fnamecsv,"w", encoding='utf-8') as fcsv :
                jsontocsv(temp,equipment,fcsv)
                print('\x1b[1;34;40m'+f'saving in {args.output} and {fnamecsv}'+ '\x1b[0m')

