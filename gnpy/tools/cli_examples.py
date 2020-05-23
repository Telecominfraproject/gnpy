#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
gnpy.tools.cli_examples
=======================

Common code for CLI examples
'''

import gnpy.core.ansi_escapes as ansi_escapes
from gnpy.core.elements import Transceiver, Fiber, RamanFiber
from gnpy.core.equipment import trx_mode_params
import gnpy.core.exceptions as exceptions
from gnpy.core.network import build_network
from gnpy.core.parameters import SimParams
from gnpy.core.science_utils import Simulation
from gnpy.core.utils import db2lin, lin2db, write_csv, automatic_nch
from gnpy.topology.request import (PathRequest, ResultElement, jsontocsv, compute_path_dsjctn, requests_aggregation,
                                   compute_constrained_path, propagate2,
                                   BLOCKING_NOPATH, correct_json_route_list,
                                   deduplicate_disjunctions, compute_path_with_disjunction)
from gnpy.topology.spectrum_assignment import build_oms_list, pth_assign_spectrum
from gnpy.tools.json_io import load_equipment, load_network, load_json, save_network, load_requests, requests_from_json, disjunctions_from_json
from gnpy.tools.plots import plot_baseline, plot_results

from argparse import ArgumentParser
from pathlib import Path
import sys
import os
import logging
from math import ceil
from numpy import linspace, mean
from json import dumps


_logger = logging.getLogger(__name__)
_examples_dir = Path(__file__).parent.parent.parent / 'examples'


def load_common_data(equipment_filename, topology_filename, simulation_filename=None, fuzzy_name_matching=False):
    '''Load common configuration from JSON files'''
    try:
        equipment = load_equipment(equipment_filename)
        network = load_network(topology_filename, equipment, fuzzy_name_matching)
        sim_params = SimParams(**load_json(simulation_filename)) if simulation_filename is not None else None
        if not sim_params:
            if next((node for node in network if isinstance(node, RamanFiber)), None) is not None:
                print(f'{ansi_escapes.red}Invocation error:{ansi_escapes.reset} '
                      f'RamanFiber requires passing simulation params via --sim-params')
                sys.exit(1)
        else:
            Simulation.set_params(sim_params)
    except exceptions.EquipmentConfigError as e:
        print(f'{ansi_escapes.red}Configuration error in the equipment library:{ansi_escapes.reset} {e}')
        sys.exit(1)
    except exceptions.NetworkTopologyError as e:
        print(f'{ansi_escapes.red}Invalid network definition:{ansi_escapes.reset} {e}')
        sys.exit(1)
    except exceptions.ConfigurationError as e:
        print(f'{ansi_escapes.red}Configuration error:{ansi_escapes.reset} {e}')
        sys.exit(1)
    except exceptions.ParametersError as e:
        print(f'{ansi_escapes.red}Simulation parameters error:{ansi_escapes.reset} {e}')
        sys.exit(1)
    except exceptions.ServiceError as e:
        print(f'{ansi_escapes.red}Service error:{ansi_escapes.reset} {e}')
        sys.exit(1)

    return (equipment, network)


def _setup_logging(args):
    logging.basicConfig(level={2: logging.DEBUG, 1: logging.INFO, 0: logging.CRITICAL}.get(args.verbose, logging.DEBUG))


def transmission_main_example(args=None):
    if args is None:
        args = sys.args
    parser = ArgumentParser()
    parser.add_argument('-e', '--equipment', type=Path,
                        default=_examples_dir / 'eqpt_config.json')
    parser.add_argument('--sim-params', type=Path,
                        default=None, help='Path to the JSON containing simulation parameters (required for Raman)')
    parser.add_argument('--show-channels', action='store_true', help='Show final per-channel OSNR summary')
    parser.add_argument('-pl', '--plot', action='store_true')
    parser.add_argument('-v', '--verbose', action='count', default=0, help='increases verbosity for each occurence')
    parser.add_argument('-l', '--list-nodes', action='store_true', help='list all transceiver nodes')
    parser.add_argument('-po', '--power', default=0, help='channel ref power in dBm')
    parser.add_argument('-names', '--names-matching', action='store_true', help='display network names that are closed matches')
    parser.add_argument('filename', nargs='?', type=Path,
                        default=_examples_dir / 'edfa_example_network.json')
    parser.add_argument('source', nargs='?', help='source node')
    parser.add_argument('destination',   nargs='?', help='destination node')
    args = parser.parse_args(args)
    _setup_logging(args)

    (equipment, network) = load_common_data(args.equipment, args.filename, args.sim_params, fuzzy_name_matching=args.names_matching)

    if args.plot:
        plot_baseline(network)

    transceivers = {n.uid: n for n in network.nodes() if isinstance(n, Transceiver)}

    if not transceivers:
        sys.exit('Network has no transceivers!')
    if len(transceivers) < 2:
        sys.exit('Network has only one transceiver!')

    if args.list_nodes:
        for uid in transceivers:
            print(uid)
        sys.exit()

    #First try to find exact match if source/destination provided
    if args.source:
        source = transceivers.pop(args.source, None)
        valid_source = True if source else False
    else:
        source = None
        _logger.info('No source node specified: picking random transceiver')

    if args.destination:
        destination = transceivers.pop(args.destination, None)
        valid_destination = True if destination else False
    else:
        destination = None
        _logger.info('No destination node specified: picking random transceiver')

    #If no exact match try to find partial match
    if args.source and not source:
        #TODO code a more advanced regex to find nodes match
        source = next((transceivers.pop(uid) for uid in transceivers \
                  if args.source.lower() in uid.lower()), None)

    if args.destination and not destination:
        #TODO code a more advanced regex to find nodes match
        destination = next((transceivers.pop(uid) for uid in transceivers \
                  if args.destination.lower() in uid.lower()), None)

    #If no partial match or no source/destination provided pick random
    if not source:
        source = list(transceivers.values())[0]
        del transceivers[source.uid]

    if not destination:
        destination = list(transceivers.values())[0]

    _logger.info(f'source = {args.source!r}')
    _logger.info(f'destination = {args.destination!r}')

    params = {}
    params['request_id'] = 0
    params['trx_type'] = ''
    params['trx_mode'] = ''
    params['source'] = source.uid
    params['destination'] = destination.uid
    params['bidir'] = False
    params['nodes_list'] = [destination.uid]
    params['loose_list'] = ['strict']
    params['format'] = ''
    params['path_bandwidth'] = 0
    trx_params = trx_mode_params(equipment)
    if args.power:
        trx_params['power'] = db2lin(float(args.power))*1e-3
    params.update(trx_params)
    req = PathRequest(**params)

    result_dicts = {}
    network_data = [{
                    'network_name'  : str(args.filename),
                    'source'        : source.uid,
                    'destination'   : destination.uid
                    }]
    result_dicts.update({'network': network_data})
    design_data = [{
                    'power_mode'        : equipment['Span']['default'].power_mode,
                    'span_power_range'  : equipment['Span']['default'].delta_power_range_db,
                    'design_pch'        : equipment['SI']['default'].power_dbm,
                    'baud_rate'         : equipment['SI']['default'].baud_rate
                    }]
    result_dicts.update({'design': design_data})
    simulation_data = []
    result_dicts.update({'simulation results': simulation_data})

    power_mode = equipment['Span']['default'].power_mode
    print('\n'.join([f'Power mode is set to {power_mode}',
                     f'=> it can be modified in eqpt_config.json - Span']))

    pref_ch_db = lin2db(req.power*1e3) #reference channel power / span (SL=20dB)
    pref_total_db = pref_ch_db + lin2db(req.nb_channel) #reference total power / span (SL=20dB)
    build_network(network, equipment, pref_ch_db, pref_total_db)
    path = compute_constrained_path(network, req)

    spans = [s.params.length for s in path if isinstance(s, RamanFiber) or isinstance(s, Fiber)]
    print(f'\nThere are {len(spans)} fiber spans over {sum(spans)/1000:.0f} km between {source.uid} '
          f'and {destination.uid}')
    print(f'\nNow propagating between {source.uid} and {destination.uid}:')

    try:
        p_start, p_stop, p_step = equipment['SI']['default'].power_range_db
        p_num = abs(int(round((p_stop - p_start)/p_step))) + 1 if p_step != 0 else 1
        power_range = list(linspace(p_start, p_stop, p_num))
    except TypeError:
        print('invalid power range definition in eqpt_config, should be power_range_db: [lower, upper, step]')
        power_range = [0]

    if not power_mode:
        #power cannot be changed in gain mode
        power_range = [0]
    for dp_db in power_range:
        req.power = db2lin(pref_ch_db + dp_db)*1e-3
        if power_mode:
            print(f'\nPropagating with input power = {ansi_escapes.cyan}{lin2db(req.power*1e3):.2f} dBm{ansi_escapes.reset}:')
        else:
            print(f'\nPropagating in {ansi_escapes.cyan}gain mode{ansi_escapes.reset}: power cannot be set manually')
        infos = propagate2(path, req, equipment)
        if len(power_range) == 1:
            for elem in path:
                print(elem)
            if power_mode:
                print(f'\nTransmission result for input power = {lin2db(req.power*1e3):.2f} dBm:')
            else:
                print(f'\nTransmission results:')
            print(f'  Final SNR total (0.1 nm): {ansi_escapes.cyan}{mean(destination.snr_01nm):.02f} dB{ansi_escapes.reset}')
        else:
            print(path[-1])

        #print(f'\n !!!!!!!!!!!!!!!!!     TEST POINT         !!!!!!!!!!!!!!!!!!!!!')
        #print(f'carriers ase output of {path[1]} =\n {list(path[1].carriers("out", "nli"))}')
        # => use "in" or "out" parameter
        # => use "nli" or "ase" or "signal" or "total" parameter
        if power_mode:
            simulation_data.append({
                        'Pch_dBm'               : pref_ch_db + dp_db,
                        'OSNR_ASE_0.1nm'        : round(mean(destination.osnr_ase_01nm),2),
                        'OSNR_ASE_signal_bw'    : round(mean(destination.osnr_ase),2),
                        'SNR_nli_signal_bw'     : round(mean(destination.osnr_nli),2),
                        'SNR_total_signal_bw'   : round(mean(destination.snr),2)
                                })
        else:
            simulation_data.append({
                        'gain_mode'             : 'power canot be set',
                        'OSNR_ASE_0.1nm'        : round(mean(destination.osnr_ase_01nm),2),
                        'OSNR_ASE_signal_bw'    : round(mean(destination.osnr_ase),2),
                        'SNR_nli_signal_bw'     : round(mean(destination.osnr_nli),2),
                        'SNR_total_signal_bw'   : round(mean(destination.snr),2)
                                })
    write_csv(result_dicts, 'simulation_result.csv')

    save_network(args.filename, network)

    if args.show_channels:
        print('\nThe total SNR per channel at the end of the line is:')
        print('{:>5}{:>26}{:>26}{:>28}{:>28}{:>28}' \
            .format('Ch. #', 'Channel frequency (THz)', 'Channel power (dBm)', 'OSNR ASE (signal bw, dB)', 'SNR NLI (signal bw, dB)', 'SNR total (signal bw, dB)'))
        for final_carrier, ch_osnr, ch_snr_nl, ch_snr in zip(infos[path[-1]][1].carriers, path[-1].osnr_ase, path[-1].osnr_nli, path[-1].snr):
            ch_freq = final_carrier.frequency * 1e-12
            ch_power = lin2db(final_carrier.power.signal*1e3)
            print('{:5}{:26.2f}{:26.2f}{:28.2f}{:28.2f}{:28.2f}' \
                .format(final_carrier.channel_number, round(ch_freq, 2), round(ch_power, 2), round(ch_osnr, 2), round(ch_snr_nl, 2), round(ch_snr, 2)))

    if not args.source:
        print(f'\n(No source node specified: picked {source.uid})')
    elif not valid_source:
        print(f'\n(Invalid source node {args.source!r} replaced with {source.uid})')

    if not args.destination:
        print(f'\n(No destination node specified: picked {destination.uid})')
    elif not valid_destination:
        print(f'\n(Invalid destination node {args.destination!r} replaced with {destination.uid})')

    if args.plot:
        plot_results(network, path, source, destination, infos)


def path_result_json(pathresult):
    """ create the response dictionnary
    """
    data = {
        'response': [n.json for n in pathresult]
    }
    return data


def path_requests_run(args=None):
    if args is None:
        args = sys.args
    parser = ArgumentParser(description='A function that computes performances for a list of services provided in a JSON file or an Excel sheet.')
    parser.add_argument('network_filename', nargs='?', type=Path,\
                        default=_examples_dir / 'meshTopologyExampleV2.xls',\
                        help='input topology file in xls or json')
    parser.add_argument('service_filename', nargs='?', type=Path,\
                        default=_examples_dir / 'meshTopologyExampleV2.xls',\
                        help='input service file in xls or json')
    parser.add_argument('eqpt_filename', nargs='?', type=Path,\
                        default=_examples_dir / 'eqpt_config.json',\
                        help='input equipment library in json. Default is eqpt_config.json')
    parser.add_argument('-bi', '--bidir', action='store_true',\
                        help='considers that all demands are bidir')
    parser.add_argument('-v', '--verbose', action='count', default=0,\
                        help='increases verbosity for each occurence')
    parser.add_argument('-o', '--output', type=Path)
    args = parser.parse_args(args)
    _setup_logging

    _logger.info(f'Computing path requests {os.path.relpath(args.service_filename)} into JSON format')
    print(f'{ansi_escapes.blue}Computing path requests {os.path.relpath(args.service_filename)} into JSON format{ansi_escapes.reset}')
    # for debug
    # print( args.eqpt_filename)

    (equipment, network) = load_common_data(args.eqpt_filename, args.network_filename)

    # Build the network once using the default power defined in SI in eqpt config
    # TODO power density: db2linp(ower_dbm": 0)/power_dbm": 0 * nb channels as defined by
    # spacing, f_min and f_max
    p_db = equipment['SI']['default'].power_dbm

    p_total_db = p_db + lin2db(automatic_nch(equipment['SI']['default'].f_min,\
        equipment['SI']['default'].f_max, equipment['SI']['default'].spacing))
    build_network(network, equipment, p_db, p_total_db)
    save_network(args.network_filename, network)
    oms_list = build_oms_list(network, equipment)

    try:
        data = load_requests(args.service_filename, equipment, bidir=args.bidir, network=network, network_filename=args.network_filename)
        rqs = requests_from_json(data, equipment)
    except exceptions.ServiceError as e:
        print(f'{ansi_escapes.red}Service error:{ansi_escapes.reset} {e}')
        sys.exit(1)
    # check that request ids are unique. Non unique ids, may
    # mess the computation: better to stop the computation
    all_ids = [r.request_id for r in rqs]
    if len(all_ids) != len(set(all_ids)):
        for item in list(set(all_ids)):
            all_ids.remove(item)
        msg = f'Requests id {all_ids} are not unique'
        _logger.critical(msg)
        sys.exit()
    rqs = correct_json_route_list(network, rqs)

    # pths = compute_path(network, equipment, rqs)
    dsjn = disjunctions_from_json(data)

    print(f'{ansi_escapes.blue}List of disjunctions{ansi_escapes.reset}')
    print(dsjn)
    # need to warn or correct in case of wrong disjunction form
    # disjunction must not be repeated with same or different ids
    dsjn = deduplicate_disjunctions(dsjn)

    # Aggregate demands with same exact constraints
    print(f'{ansi_escapes.blue}Aggregating similar requests{ansi_escapes.reset}')

    rqs, dsjn = requests_aggregation(rqs, dsjn)
    # TODO export novel set of aggregated demands in a json file

    print(f'{ansi_escapes.blue}The following services have been requested:{ansi_escapes.reset}')
    print(rqs)

    print(f'{ansi_escapes.blue}Computing all paths with constraints{ansi_escapes.reset}')
    try:
        pths = compute_path_dsjctn(network, equipment, rqs, dsjn)
    except exceptions.DisjunctionError as this_e:
        print(f'{ansi_escapes.red}Disjunction error:{ansi_escapes.reset} {this_e}')
        sys.exit(1)

    print(f'{ansi_escapes.blue}Propagating on selected path{ansi_escapes.reset}')
    propagatedpths, reversed_pths, reversed_propagatedpths = compute_path_with_disjunction(network, equipment, rqs, pths)
    # Note that deepcopy used in compute_path_with_disjunction returns
    # a list of nodes which are not belonging to network (they are copies of the node objects).
    # so there can not be propagation on these nodes.

    pth_assign_spectrum(pths, rqs, oms_list, reversed_pths)

    print(f'{ansi_escapes.blue}Result summary{ansi_escapes.reset}')
    header = ['req id', '  demand', '  snr@bandwidth A-Z (Z-A)', '  snr@0.1nm A-Z (Z-A)',\
              '  Receiver minOSNR', '  mode', '  Gbit/s', '  nb of tsp pairs',\
              'N,M or blocking reason']
    data = []
    data.append(header)
    for i, this_p in enumerate(propagatedpths):
        rev_pth = reversed_propagatedpths[i]
        if rev_pth and this_p:
            psnrb = f'{round(mean(this_p[-1].snr),2)} ({round(mean(rev_pth[-1].snr),2)})'
            psnr = f'{round(mean(this_p[-1].snr_01nm), 2)}' +\
                   f' ({round(mean(rev_pth[-1].snr_01nm),2)})'
        elif this_p:
            psnrb = f'{round(mean(this_p[-1].snr),2)}'
            psnr = f'{round(mean(this_p[-1].snr_01nm),2)}'

        try :
            if rqs[i].blocking_reason in  BLOCKING_NOPATH:
                line = [f'{rqs[i].request_id}', f' {rqs[i].source} to {rqs[i].destination} :',\
                        f'-', f'-', f'-', f'{rqs[i].tsp_mode}', f'{round(rqs[i].path_bandwidth * 1e-9,2)}',\
                        f'-', f'{rqs[i].blocking_reason}']
            else:
                line = [f'{rqs[i].request_id}', f' {rqs[i].source} to {rqs[i].destination} : ', psnrb,\
                        psnr, f'-', f'{rqs[i].tsp_mode}', f'{round(rqs[i].path_bandwidth * 1e-9, 2)}',\
                        f'-', f'{rqs[i].blocking_reason}']
        except AttributeError:
            line = [f'{rqs[i].request_id}', f' {rqs[i].source} to {rqs[i].destination} : ', psnrb,\
                    psnr, f'{rqs[i].OSNR}', f'{rqs[i].tsp_mode}', f'{round(rqs[i].path_bandwidth * 1e-9,2)}',\
                    f'{ceil(rqs[i].path_bandwidth / rqs[i].bit_rate) }', f'({rqs[i].N},{rqs[i].M})']
        data.append(line)

    col_width = max(len(word) for row in data for word in row[2:])   # padding
    firstcol_width = max(len(row[0]) for row in data)   # padding
    secondcol_width = max(len(row[1]) for row in data)   # padding
    for row in data:
        firstcol = ''.join(row[0].ljust(firstcol_width))
        secondcol = ''.join(row[1].ljust(secondcol_width))
        remainingcols = ''.join(word.center(col_width, ' ') for word in row[2:])
        print(f'{firstcol} {secondcol} {remainingcols}')
    print(f'{ansi_escapes.yellow}Result summary shows mean SNR and OSNR (average over all channels){ansi_escapes.reset}')

    if args.output:
        result = []
        # assumes that list of rqs and list of propgatedpths have same order
        for i, pth in enumerate(propagatedpths):
            result.append(ResultElement(rqs[i], pth, reversed_propagatedpths[i]))
        temp = path_result_json(result)
        fnamecsv = f'{str(args.output)[0:len(str(args.output))-len(str(args.output.suffix))]}.csv'
        fnamejson = f'{str(args.output)[0:len(str(args.output))-len(str(args.output.suffix))]}.json'
        with open(fnamejson, 'w', encoding='utf-8') as fjson:
            fjson.write(dumps(path_result_json(result), indent=2, ensure_ascii=False))
            with open(fnamecsv, "w", encoding='utf-8') as fcsv:
                jsontocsv(temp, equipment, fcsv)
                print('\x1b[1;34;40m'+f'saving in {args.output} and {fnamecsv}'+ '\x1b[0m')
