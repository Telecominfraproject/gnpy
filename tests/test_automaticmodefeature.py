#!/usr/bin/env python3
# Module name : test_automaticmodefeature.py
# Version :
# License : BSD 3-Clause Licence
# Copyright (c) 2018, Telecom Infra Project

"""
@author: esther.lerouzic
checks that empty info on mode, power, nbchannel in service file are supported
    uses combination of [mode, pow, nb channel] filled or empty defined in meshTopologyToy_services.json
    leading to feasible path or not, and check the propagate and propagate_and_optimize_mode
    return the expected based on a reference toy example

"""

from pathlib import Path
import pytest
from gnpy.core.network import build_network
from gnpy.core.utils import automatic_nch, lin2db
from gnpy.core.elements import Roadm
from gnpy.topology.request import compute_path_dsjctn, \
                                  propagate, \
                                  propagate_and_optimize_mode, \
                                  correct_json_route_list
from gnpy.tools.json_io import load_network, \
                               load_equipment, \
                               requests_from_json, \
                               load_requests


PARENT_PATH = Path(__file__).parent.parent
NETWORK_FILE_NAME = PARENT_PATH / 'tests/data/testTopology_expected.json'
SERVICE_FILE_NAME = PARENT_PATH / 'tests/data/testTopology_testservices.json'
EQPT_LIBRARY_NAME = PARENT_PATH / 'tests/data/eqpt_config.json'


@pytest.fixture()
def test_network():
    equipment = load_equipment(EQPT_LIBRARY_NAME)
    network = load_network(NETWORK_FILE_NAME, equipment)
    data = load_requests(SERVICE_FILE_NAME, 
                         EQPT_LIBRARY_NAME,
                         bidir=False,
                         network=network,
                         network_filename=NETWORK_FILE_NAME)

    # Build the network once using the default power defined in SI in eqpt config
    # power density : db2linp(ower_dbm": 0)/power_dbm": 0 * nb channels as defined by
    # spacing, f_min and f_max
    power_db = equipment['SI']['default'].power_dbm

    power_total_db = power_db + lin2db(automatic_nch(equipment['SI']['default'].f_min,
                                             equipment['SI']['default'].f_max,
                                             equipment['SI']['default'].spacing))
    build_network(network, equipment, power_db, power_total_db)

    requests = requests_from_json(data, equipment)
    requests = correct_json_route_list(network, requests)
    disjunctions = []
    paths = compute_path_dsjctn(network, equipment, requests, disjunctions)
    return requests, paths, equipment


@pytest.mark.parametrize("expected_mode", [['16QAM', 'PS_SP64_1', 'PS_SP64_1', \
                                            'PS_SP64_1', 'mode 2 - fake', 'mode 2', \
                                            'PS_SP64_1', 'mode 3', 'PS_SP64_1', \
                                            'PS_SP64_1', '16QAM', 'mode 1', \
                                            'PS_SP64_1', 'PS_SP64_1', 'mode 1', \
                                            'mode 2', 'mode 1', 'mode 2', 'nok']])
def test_automaticmodefeature(expected_mode, test_network):
    requests, paths, equipment = test_network
    path_request_list = []

    for i, pathreq in enumerate(requests):

        # use the power specified in requests but might be different from the one specified for design
        # the power is an optional parameter for requests definition
        # if optional, use the one defines in eqt_config.json
        p_db = lin2db(pathreq.power * 1e3)
        p_total_db = p_db + lin2db(pathreq.nb_channel)
        print(f'request {pathreq.request_id}')
        print(f'Computing path from {pathreq.source} to {pathreq.destination}')
        # adding first node to be clearer on the output
        print(f'with path constraint: {[pathreq.source]+pathreq.nodes_list}')

        total_path = paths[i]
        print(f'Computed path (roadms):{[e.uid for e in total_path  if isinstance(e, Roadm)]}\n')
        # for debug
        # print(f'{pathreq.baud_rate}   {pathreq.power}   {pathreq.spacing}   {pathreq.nb_channel}')
        if pathreq.baud_rate is not None:
            print(pathreq.format)
            path_request_list.append(pathreq.format)
            total_path = propagate(total_path, pathreq, equipment)
        else:
            total_path, mode = propagate_and_optimize_mode(total_path, pathreq, equipment)
            # if no baudrate satisfies spacing, no mode is returned and an empty path is returned
            # a warning is shown in the propagate_and_optimize_mode
            if mode is not None:
                print(mode['format'])
                path_request_list.append(mode['format'])
            else:
                print('nok')
                path_request_list.append('nok')
    print(path_request_list)
    assert path_request_list == expected_mode


def test_path_low_osnr(test_network):
    """Test if path's OSNR is lower than mode's OSNR."""
    requests, paths, equipment = test_network
    #Condition to achieve path's OSNR lower than mode's
    path_request = requests[0]
    total_path = paths[0]
    modes = [{
        'format': 'a',
        'baud_rate': 32e9,
        'OSNR': 100,
        'bit_rate': 200e9,
        'roll_off': 0.15,
        'tx_osnr': 100,
        'min_spacing': 50e9,
        'cost': 1
        },
        {
        'format': 'b',
        'baud_rate': 32e9,
        'OSNR': 50,
        'bit_rate': 100e9,
        'roll_off': 0.15,
        'tx_osnr': 100,
        'min_spacing': 50e9,
        'cost': 1
        }]
    equipment['Transceiver']['Voyager_16QAM'].mode = modes
    mode_expected = equipment['Transceiver']['Voyager_16QAM'].mode[1]
    total_path, mode = propagate_and_optimize_mode(total_path,
                                                   path_request,
                                                   equipment)
    assert mode == mode_expected


def test_no_computed_snr(test_network):
    """Test if path's SNR is zero (empty list in the code),
    resulting in 'NO_COMPUTED_SNR'.
    """
    mode_expected = None
    requests, paths, equipment = test_network
    path_request = requests[0]
    # condition to achieve path.snr == []
    # can only be achivied via large spacing,
    # which will result in no channel propagating
    path_request.spacing = 5e12
    total_path = paths[0]
    with pytest.raises(TypeError):
        total_path, mode = propagate_and_optimize_mode(total_path,
                                                       path_request,
                                                       equipment)