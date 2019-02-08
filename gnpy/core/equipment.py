#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
gnpy.core.equipment
===================

This module contains functionality for specifying equipment.
'''

from numpy import clip, polyval
from sys import exit
from operator import itemgetter
from math import isclose
from pathlib import Path
from json import load
from gnpy.core.utils import lin2db, db2lin, load_json
from collections import namedtuple
from gnpy.core.elements import Edfa

Model_vg = namedtuple('Model_vg', 'nf1 nf2 delta_p')
Model_fg = namedtuple('Model_fg', 'nf0')
Model_openroadm = namedtuple('Model_openroadm', 'nf_coef')
Model_hybrid = namedtuple('Model_hybrid', 'nf_ram gain_ram edfa_variety')
Model_dual_stage = namedtuple('Model_dual_stage', 'preamp_variety booster_variety')

class common:
    def update_attr(self, default_values, kwargs):
        clean_kwargs = {k:v for k,v in kwargs.items() if v !=''}
        for k,v in default_values.items():
            v = clean_kwargs.get(k,v)
            setattr(self, k, v)

class SI(common):
    default_values =\
    {
        "f_min":            191.35e12,
        "f_max":            196.1e12,
        "baud_rate":        32e9,        
        "spacing":          50e9,
        "power_dbm":        0,
        "power_range_db":   [0,0,0.5],
        "roll_off":         0.15,
        "tx_osnr":          45,
        "sys_margins":      0    
    }

    def __init__(self, **kwargs):
        self.update_attr(self.default_values, kwargs)    

class Spans(common):
    default_values = \
    {
        'power_mode':                       True,
        'delta_power_range_db':             None,
        'max_fiber_lineic_loss_for_raman':  0.25,
        'max_length':                       150,
        'length_units':                     'km',
        'max_loss':                         None,
        'padding':                          10,
        'EOL':                              0,
        'con_in':                           0,
        'con_out':                          0
    }
    
    def __init__(self, **kwargs):
        self.update_attr(self.default_values, kwargs)

class Roadms(common):
    default_values = \
    {
        'gain_mode_default_loss':   20,
        'power_mode_pout_target':   -18,
        'add_drop_osnr':            100
    }    

    def __init__(self, **kwargs):
        self.update_attr(self.default_values, kwargs)

class Transceiver(common):
    default_values = \
    {
        'type_variety': None,
        'frequency':    None,
        'mode':         {}
    }

    def __init__(self, **kwargs):
        self.update_attr(self.default_values, kwargs)        

class Fiber(common):
    default_values = \
    {
        'type_variety':  '',
        'dispersion':    None,
        'gamma':         0
    }

    def __init__(self, **kwargs):
        self.update_attr(self.default_values, kwargs)    

class Amp(common):
    default_values = \
    {
        'type_variety':         '',
        'type_def':             '',
        'gain_flatmax':         None,
        'gain_min':             None,
        'p_max':                None,
        'nf_model':             None,
        'dual_stage_model':     None,
        'nf_fit_coeff':         None,
        'nf_ripple':            None,
        'dgt':                  None,
        'gain_ripple':          None,
        'out_voa_auto':         False,
        'allowed_for_design':   False,
        'raman':                False
    }

    def __init__(self, **kwargs):
        self.update_attr(self.default_values, kwargs)

    @classmethod
    def from_json(cls, filename, **kwargs):
        config = Path(filename).parent / 'default_edfa_config.json'

        type_variety = kwargs['type_variety']
        type_def = kwargs.get('type_def', 'variable_gain') #default compatibility with older json eqpt files
        nf_def = None
        dual_stage_def = None

        if type_def == 'fixed_gain':
            try:
                nf0 = kwargs.pop('nf0')
            except KeyError: #nf0 is expected for a fixed gain amp
                print(f'missing nf0 value input for amplifier: {type_variety} in eqpt_config.json')
                exit()
            try: #remove all remaining nf inputs
                del kwargs['nf_min']
                del kwargs['nf_max']
            except KeyError: pass #nf_min and nf_max are not needed for fixed gain amp
            nf_def = Model_fg(nf0)
        elif type_def == 'advanced_model':
            config = Path(filename).parent / kwargs.pop('advanced_config_from_json')
        elif type_def == 'variable_gain':
            gain_min, gain_max = kwargs['gain_min'], kwargs['gain_flatmax']
            try: #nf_min and nf_max are expected for a variable gain amp
                nf_min = kwargs.pop('nf_min')
                nf_max = kwargs.pop('nf_max')
            except KeyError:
                print(f'missing nf_min/max value input for amplifier: {type_variety} in eqpt_config.json')
                exit()
            try: #remove all remaining nf inputs
                del kwargs['nf0']
            except KeyError: pass #nf0 is not needed for variable gain amp
            nf1, nf2, delta_p = nf_model(type_variety, gain_min, gain_max, nf_min, nf_max)
            nf_def = Model_vg(nf1, nf2, delta_p)
        elif type_def == 'openroadm':
            try:
                nf_coef = kwargs.pop('nf_coef')
            except KeyError: #nf_coef is expected for openroadm amp
                print(f'missing nf_coef input for amplifier: {type_variety} in eqpt_config.json')
                exit()
            nf_def = Model_openroadm(nf_coef)
        elif type_def == 'dual_stage':
            try: #nf_ram and gain_ram are expected for a hybrid amp
                preamp_variety = kwargs.pop('preamp_variety')
                booster_variety = kwargs.pop('booster_variety')
            except KeyError:
                print(f'missing preamp/booster variety input for amplifier: {type_variety} in eqpt_config.json')
                exit()
            dual_stage_def = Model_dual_stage(preamp_variety, booster_variety)

        with open(config, encoding='utf-8') as f:
            json_data = load(f)

        return cls(**{**kwargs, **json_data, 
            'nf_model': nf_def, 'dual_stage_model': dual_stage_def})


def nf_model(type_variety, gain_min, gain_max, nf_min, nf_max):
    if nf_min < -10:
        print(f'Invalid nf_min value {nf_min!r} for amplifier {type_variety}')
        exit()
    if nf_max < -10:
        print(f'Invalid nf_max value {nf_max!r} for amplifier {type_variety}')
        exit()

    # NF estimation model based on nf_min and nf_max
    # delta_p:  max power dB difference between first and second stage coils
    # dB g1a:   first stage gain - internal VOA attenuation
    # nf1, nf2: first and second stage coils
    #           calculated by solving nf_{min,max} = nf1 + nf2 / g1a{min,max}
    delta_p = 5
    g1a_min = gain_min - (gain_max - gain_min) - delta_p
    g1a_max = gain_max - delta_p
    nf2 = lin2db((db2lin(nf_min) - db2lin(nf_max)) /
                 (1/db2lin(g1a_max) - 1/db2lin(g1a_min)))
    nf1 = lin2db(db2lin(nf_min) - db2lin(nf2)/db2lin(g1a_max))

    if nf1 < 4:
        print(f'First coil value too low {nf1} for amplifier {type_variety}')
        exit()

    # Check 1 dB < delta_p < 6 dB to ensure nf_min and nf_max values make sense.
    # There shouldn't be high nf differences between the two coils:
    #    nf2 should be nf1 + 0.3 < nf2 < nf1 + 2
    # If not, recompute and check delta_p
    if not nf1 + 0.3 < nf2 < nf1 + 2:
        nf2 = clip(nf2, nf1 + 0.3, nf1 + 2)
        g1a_max = lin2db(db2lin(nf2) / (db2lin(nf_min) - db2lin(nf1)))
        delta_p = gain_max - g1a_max
        g1a_min = gain_min - (gain_max-gain_min) - delta_p
        if not 1 < delta_p < 11:
            print(f'Computed \N{greek capital letter delta}P invalid \
                \n 1st coil vs 2nd coil calculated DeltaP {delta_p:.2f} for \
                \n amplifier {type_variety} is not valid: revise inputs \
                \n calculated 1st coil NF = {nf1:.2f}, 2nd coil NF = {nf2:.2f}')
            exit()
    # Check calculated values for nf1 and nf2
    calc_nf_min = lin2db(db2lin(nf1) + db2lin(nf2)/db2lin(g1a_max))
    if not isclose(nf_min, calc_nf_min, abs_tol=0.01):
        print(f'nf_min does not match calc_nf_min, {nf_min} vs {calc_nf_min} for amp {type_variety}')
        exit()
    calc_nf_max = lin2db(db2lin(nf1) + db2lin(nf2)/db2lin(g1a_min))
    if not isclose(nf_max, calc_nf_max, abs_tol=0.01):
        print(f'nf_max does not match calc_nf_max, {nf_max} vs {calc_nf_max} for amp {type_variety}')
        exit()

    return nf1, nf2, delta_p

def edfa_nf(gain_target, variety_type, equipment):
    amp_params = equipment['Edfa'][variety_type]
    amp = Edfa(
            uid = f'calc_NF',
            params = amp_params.__dict__,
            operational = {
                'gain_target': gain_target,
                'tilt_target': 0
                        }
            )
    amp.pin_db = 0
    amp.nch = 88
    return amp._calc_nf(True)

def trx_mode_params(equipment, trx_type_variety='', trx_mode='', error_message=False):
    """return the trx and SI parameters from eqpt_config for a given type_variety and mode (ie format)"""
    trx_params = {}
    default_si_data = equipment['SI']['default']
    
    try:
        trxs = equipment['Transceiver']
        #if called from path_requests_run.py, trx_mode is filled with None when not specified by user
        #if called from transmission_main.py, trx_mode is ''
        if trx_mode is not None:
            mode_params = next(mode for trx in trxs \
                        if trx == trx_type_variety \
                        for mode in trxs[trx].mode \
                        if mode['format'] == trx_mode)
            trx_params = {**mode_params}
            # sanity check: spacing baudrate must be smaller than min spacing
            if trx_params['baud_rate'] > trx_params['min_spacing'] :
                msg = f'Inconsistency in equipment library:\n Transpoder "{trx_type_variety}" mode "{trx_params["format"]}" '+\
                    f'has baud rate: {trx_params["baud_rate"]*1e-9} GHz greater than min_spacing {trx_params["min_spacing"]*1e-9}.'
                print(msg)
                exit()
        else:
            mode_params = {"format": "undetermined",
                       "baud_rate": None,
                       "OSNR": None,
                       "bit_rate": None,
                       "roll_off": None,
                       "tx_osnr":None,
                       "min_spacing":None,
                       "cost":None}
            trx_params = {**mode_params} 
        trx_params['f_min'] = equipment['Transceiver'][trx_type_variety].frequency['min']
        trx_params['f_max'] = equipment['Transceiver'][trx_type_variety].frequency['max']

        # TODO: novel automatic feature maybe unwanted if spacing is specified
        # trx_params['spacing'] = automatic_spacing(trx_params['baud_rate'])
        # temp = trx_params['spacing']
        # print(f'spacing {temp}')
    except StopIteration :
        if error_message:
            print(f'could not find tsp : {trx_type_variety} with mode: {trx_mode} in eqpt library')
            print('Computation stopped.')
            exit()
        else:
            # default transponder charcteristics
            # mainly used with transmission_main_example.py
            trx_params['f_min'] = default_si_data.f_min
            trx_params['f_max'] = default_si_data.f_max
            trx_params['baud_rate'] = default_si_data.baud_rate
            trx_params['spacing'] = default_si_data.spacing
            trx_params['OSNR'] = None
            trx_params['bit_rate'] = None
            trx_params['cost'] = None
            trx_params['roll_off'] = default_si_data.roll_off
            trx_params['tx_osnr'] = default_si_data.tx_osnr
            trx_params['min_spacing'] = None
            nch = automatic_nch(trx_params['f_min'], trx_params['f_max'], trx_params['spacing'])
            trx_params['nb_channel'] = nch
            print(f'There are {nch} channels propagating')
                
    trx_params['power'] =  db2lin(default_si_data.power_dbm)*1e-3

    return trx_params

def automatic_spacing(baud_rate):
    """return the min possible channel spacing for a given baud rate"""
    # TODO : this should parametrized in a cfg file
    spacing_list = [(33e9,37.5e9), (38e9,50e9), (50e9,62.5e9), (67e9,75e9), (92e9,100e9)] #list of possible tuples
                                                #[(max_baud_rate, spacing_for_this_baud_rate)]
    return min((s[1] for s in spacing_list if s[0] > baud_rate), default=baud_rate*1.2)

def automatic_nch(f_min, f_max, spacing):
    return int((f_max - f_min)//spacing)

def automatic_fmax(f_min, spacing, nch):
    return f_min + spacing * nch

def load_equipment(filename):
    json_data = load_json(filename)
    return equipment_from_json(json_data, filename)

def update_trx_osnr(equipment):
    """add sys_margins to all Transceivers OSNR values"""
    for trx in equipment['Transceiver'].values():
        for m in trx.mode:
            m['OSNR'] = m['OSNR'] + equipment['SI']['default'].sys_margins
    return equipment

def update_hybrid(equipment):
    edfa_dict = equipment['Edfa']
    for edfa in edfa_dict.values():
        if edfa.type_def == 'hybrid':
            edfa_booster = edfa_dict[edfa.raman_model.edfa_variety]
            edfa.nf_model = edfa_booster.nf_model
            edfa.p_max = edfa_booster.p_max
            edfa.gain_flatmax = edfa_booster.gain_flatmax + edfa.raman_model.gain_ram
            edfa.edfa_gain_min = edfa_booster.gain_min
            edfa.edfa_gain_flatmax = edfa_booster.gain_flatmax
    return equipment

def update_dual_stage(equipment):
    edfa_dict = equipment['Edfa']
    for edfa in edfa_dict.values():
        if edfa.type_def == 'dual_stage':
            edfa_preamp = edfa_dict[edfa.dual_stage_model.preamp_variety]
            edfa_booster = edfa_dict[edfa.dual_stage_model.booster_variety]
            for k,v in edfa_preamp.__dict__.items():
                attr_k = 'preamp_'+k
                setattr(edfa, attr_k, v)
            for k,v in edfa_booster.__dict__.items():
                attr_k = 'booster_'+k
                setattr(edfa, attr_k, v)           
            edfa.p_max = edfa_booster.p_max
            edfa.gain_flatmax = edfa_booster.gain_flatmax + edfa_preamp.gain_flatmax
            if edfa.gain_min < edfa_preamp.gain_min:
                print(
                    f'\x1b[1;31;40m'\
                    + f'CRITICAL: dual stage {edfa.type_variety} min gain is lower than its preamp min gain\
                        => please increase its min gain in eqpt_config.json'\
                    + '\x1b[0m'
                    )                        
                exit()
    return equipment


def equipment_from_json(json_data, filename):
    """build global dictionnary eqpt_library that stores all eqpt characteristics:
    edfa type type_variety, fiber type_variety
    from the eqpt_config.json (filename parameter)
    also read advanced_config_from_json file parameters for edfa if they are available:
    typically nf_ripple, dfg gain ripple, dgt and nf polynomial nf_fit_coeff
    if advanced_config_from_json file parameter is not present: use nf_model:
    requires nf_min and nf_max values boundaries of the edfa gain range
    """
    equipment = {}
    for key, entries in json_data.items():
        equipment[key] = {}
        typ = globals()[key]
        for entry in entries:
            subkey = entry.get('type_variety', 'default')           
            if key == 'Edfa':
                equipment[key][subkey] = Amp.from_json(filename, **entry)
            else:                
                equipment[key][subkey] = typ(**entry)
    equipment = update_trx_osnr(equipment)
    equipment = update_hybrid(equipment)
    equipment = update_dual_stage(equipment)
    return equipment
