#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Jean-Luc Auge
# @Date:   2018-02-02 14:06:55

from gnpy.core.elements import Edfa
from numpy import zeros, array
from json import load
from gnpy.core.elements import Transceiver, Fiber, Edfa
from gnpy.core.utils import lin2db, db2lin
from gnpy.core.info import create_input_spectral_information, SpectralInformation, Channel, Power, Pref
from gnpy.core.equipment import load_equipment, automatic_fmax
from gnpy.core.network import build_network, load_network, set_roadm_loss
from pathlib import Path
import pytest

TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR / 'data'
test_network = DATA_DIR / 'test_network.json'
eqpt_library = DATA_DIR / 'eqpt_config.json'

# TODO in elements.py code: pytests doesn't pass with 1 channel: interpolate fail
@pytest.fixture(
    params=[(96, 0.05e12), (60, 0.075e12), (45, 0.1e12), (2, 0.1e12)],
    ids=['50GHz spacing', '75GHz spacing', '100GHz spacing', '2 channels'])
def nch_and_spacing(request):
    """parametrize channel count vs channel spacing (Hz)"""
    yield request.param

@pytest.fixture()
def bw():
    """parametrize signal bandwidth (Hz)"""
    return 45e9

@pytest.fixture()
def setup_edfa_variable_gain():
    """init edfa class by reading test_network.json file
    remove all gain and nf ripple"""
    equipment = load_equipment(eqpt_library)
    network = load_network(test_network, equipment)
    build_network(network, equipment,0, 20)
    edfa = [n for n in network.nodes() if isinstance(n, Edfa)][0]
    edfa.gain_ripple = zeros(96)
    edfa.interpol_nf_ripple = zeros(96)
    yield edfa

@pytest.fixture()
def setup_edfa_fixed_gain():
    """init edfa class by reading the 2nd edfa in test_network.json file"""
    equipment = load_equipment(eqpt_library)
    network = load_network(test_network, equipment)
    build_network(network, equipment, 0, 20)
    edfa = [n for n in network.nodes() if isinstance(n, Edfa)][1]
    yield edfa

@pytest.fixture()
def setup_trx():
    """init transceiver class to access snr and osnr calculations"""
    equipment = load_equipment(eqpt_library)
    network = load_network(test_network, equipment)
    build_network(network, equipment, 0, 20)
    trx = [n for n in network.nodes() if isinstance(n, Transceiver)][0]
    return trx

@pytest.fixture()
def si(nch_and_spacing, bw):
    """parametrize a channel comb with nb_channel, spacing and signal bw"""
    nb_channel, spacing = nch_and_spacing
    f_min = 191.3e12
    f_max = automatic_fmax(f_min, spacing, nb_channel)
    return create_input_spectral_information(f_min, f_max, 0.15, bw, 1e-3, spacing)

@pytest.mark.parametrize("gain, nf_expected", [(10, 15), (15, 10), (25, 5.8)])
def test_variable_gain_nf(gain, nf_expected, setup_edfa_variable_gain, si):
    """=> unitary test for variable gain model Edfa._calc_nf() (and Edfa.interpol_params)"""
    edfa = setup_edfa_variable_gain
    frequencies = array([c.frequency for c in si.carriers])
    pin = array([c.power.signal+c.power.nli+c.power.ase for c in si.carriers])
    pin = pin/db2lin(gain)
    baud_rates = array([c.baud_rate for c in si.carriers])
    edfa.operational.gain_target = gain
    pref = Pref(0, -gain)
    edfa.interpol_params(frequencies, pin, baud_rates, pref)
    result = edfa.nf
    assert pytest.approx(nf_expected, abs=0.01) == result[0]

@pytest.mark.parametrize("gain, nf_expected", [(15, 10), (20, 5), (25, 5)])
def test_fixed_gain_nf(gain, nf_expected, setup_edfa_fixed_gain, si):
    """=> unitary test for fixed gain model Edfa._calc_nf() (and Edfa.interpol_params)"""
    edfa = setup_edfa_fixed_gain
    frequencies = array([c.frequency for c in si.carriers])
    pin = array([c.power.signal+c.power.nli+c.power.ase for c in si.carriers])
    pin = pin/db2lin(gain)
    baud_rates = array([c.baud_rate for c in si.carriers])
    edfa.operational.gain_target = gain
    pref = Pref(0, -gain)
    edfa.interpol_params(frequencies, pin, baud_rates, pref)

    assert pytest.approx(nf_expected, abs=0.01) == edfa.nf[0]

def test_si(si, nch_and_spacing):
    """basic total power check of the channel comb generation"""
    nb_channel = nch_and_spacing[0]
    pin = array([c.power.signal+c.power.nli+c.power.ase for c in si.carriers])
    p_tot = pin.sum()
    expected_p_tot = si.carriers[0].power.signal * nb_channel
    assert pytest.approx(expected_p_tot, abs=0.01) == p_tot

@pytest.mark.parametrize("gain", [17, 19, 21, 23])
def test_compare_nf_models(gain, setup_edfa_variable_gain, si):
    """ compare the 2 amplifier models (polynomial and estimated from nf_min and max)
     => nf_model vs nf_poly_fit for intermediate gain values:
     between gain_min and gain_flatmax some discrepancy is expected but target < 0.5dB
     => unitary test for Edfa._calc_nf (and Edfa.interpol_params)"""
    edfa = setup_edfa_variable_gain
    frequencies = array([c.frequency for c in si.carriers])
    pin = array([c.power.signal+c.power.nli+c.power.ase for c in si.carriers])
    pin = pin/db2lin(gain)
    baud_rates = array([c.baud_rate for c in si.carriers])
    edfa.operational.gain_target = gain
    pref = Pref(0, -gain)
    edfa.interpol_params(frequencies, pin, baud_rates, pref)
    nf_model = edfa.nf[0]
    edfa.interpol_params(frequencies, pin, baud_rates, pref)
    nf_poly = edfa.nf[0]
    assert pytest.approx(nf_model, abs=0.5) == nf_poly

@pytest.mark.parametrize("gain", [13, 15, 17, 19, 21, 23, 25, 27])
def test_ase_noise(gain, si, setup_edfa_variable_gain, setup_trx, bw):
    """testing 3 different ways of calculating osnr:
    1-pin-edfa.nf+58 vs
    2-pout/pase afet propagate
    3-Transceiver osnr_ase_01nm
    => unitary test for Edfa.noise_profile (Edfa.interpol_params, Edfa.propagate)"""
    edfa = setup_edfa_variable_gain
    frequencies = array([c.frequency for c in si.carriers])
    pin = array([c.power.signal+c.power.nli+c.power.ase for c in si.carriers])
    baud_rates = array([c.baud_rate for c in si.carriers])
    edfa.operational.gain_target = gain
    pref = Pref(0, 0)
    edfa.interpol_params(frequencies, pin, baud_rates, pref)
    nf = edfa.nf
    pin = lin2db(pin[0]*1e3)
    osnr_expected = pin - nf[0] + 58

    si = edfa(si)
    pout = array([c.power.signal for c in si.carriers])
    pase = array([c.power.ase for c in si.carriers])
    osnr = lin2db(pout[0] / pase[0]) - lin2db(12.5e9/bw)
    assert pytest.approx(osnr_expected, abs=0.01) == osnr

    trx = setup_trx
    si = trx(si)
    osnr = trx.osnr_ase_01nm[0]
    assert pytest.approx(osnr_expected, abs=0.01) == osnr
