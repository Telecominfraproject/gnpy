#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
gnpy.core.info
==============

This module contains classes for modelling :class:`SpectralInformation`.
'''


from collections import namedtuple, Sized
from numpy import argsort, mean, array, squeeze, append, ones, ceil, any, zeros
from gnpy.core.utils import automatic_nch, lin2db
from gnpy.core.exceptions import InfoError

BASE_GRID = 12.5e9  # Hz


class Power(namedtuple('Power', 'signal nli ase')):
    """carriers power in W"""


class Channel(namedtuple('Channel', 'channel_number frequency baud_rate roll_off power chromatic_dispersion pmd')):
    """ Class containing the parameters of a WDM signal.

        :param channel_number: channel number in the WDM grid
        :param frequency: central frequency of the signal (Hz)
        :param baud_rate: the symbol rate of the signal (Baud)
        :param roll_off: the roll off of the signal. It is a pure number between 0 and 1
        :param power (gnpy.core.info.Power): power of signal, ASE noise and NLI (W)
        :param chromatic_dispersion: chromatic dispersion (s/m)
        :param pmd: polarization mode dispersion (s)
    """


class Pref(namedtuple('Pref', 'p_span0, p_spani, neq_ch ')):
    """noiseless reference power in dBm:
    p_span0: inital target carrier power
    p_spani: carrier power after element i
    neq_ch: equivalent channel count in dB"""


class SpectralInformation(object):
    def __init__(self, frequency, baud_rate, grid, signal, nli, ase,
                 roll_off, chromatic_dispersion, pmd):
        indices = argsort(frequency)
        self._frequency = frequency[indices]
        self._number_of_channels = len(self._frequency)
        self._grid = grid[indices]
        self._baud_rate = baud_rate[indices]
        if any(self._frequency[:-1] + self._grid[:-1] / 2 > self._frequency[1:] - self._grid[1:] / 2):
            raise InfoError('Spectrum required grid larger than the frequencies spectral distance.')
        elif any(self._baud_rate > self._grid):
            raise InfoError('Spectrum baud rate larger than the grid.')
        self._signal = signal[indices]
        self._nli = nli[indices]
        self._ase = ase[indices]
        self._roll_off = roll_off[indices]
        self._chromatic_dispersion = chromatic_dispersion[indices]
        self._pmd = pmd[indices]
        self._channel_number = [*range(1, len(self._frequency) + 1)]
        pref = lin2db(mean(signal) * 1e3)
        self._pref = Pref(pref, pref, lin2db(len(self._frequency)))

    @property
    def pref(self):
        return self._pref

    @pref.setter
    def pref(self, pref):
        self._pref = pref

    @property
    def frequency(self):
        return self._frequency

    @property
    def grid(self):
        return self._grid

    @property
    def baud_rate(self):
        return self._baud_rate

    @property
    def number_of_channels(self):
        return self._number_of_channels

    @property
    def powers(self):
        powers = zip(self.signal, self.nli, self.ase)
        return [Power(*p) for p in powers]

    @property
    def signal(self):
        return self._signal

    @signal.setter
    def signal(self, signal):
        self._signal = signal

    @property
    def nli(self):
        return self._nli

    @nli.setter
    def nli(self, nli):
        self._nli = nli

    @property
    def ase(self):
        return self._ase

    @ase.setter
    def ase(self, ase):
        self._ase = ase

    @property
    def gsnr(self):
        return lin2db(self.signal/(self.ase+self.nli))

    @property
    def osnr(self):
        return lin2db(self.signal/self.ase)

    @property
    def snr(self):
        return lin2db(self.signal/self.nli)

    @property
    def roll_off(self):
        return self._roll_off

    @property
    def chromatic_dispersion(self):
        return self._chromatic_dispersion

    @chromatic_dispersion.setter
    def chromatic_dispersion(self, chromatic_dispersion):
        self._chromatic_dispersion = chromatic_dispersion

    @property
    def pmd(self):
        return self._pmd

    @pmd.setter
    def pmd(self, pmd):
        self._pmd = pmd

    @property
    def channel_number(self):
        return self._channel_number

    @property
    def carriers(self):
        entries = zip(self.channel_number, self.frequency, self.baud_rate,
                      self.roll_off, self.powers, self.chromatic_dispersion, self.pmd)
        return [Channel(*entry) for entry in entries]

    def __add__(self, si):
        frequency = append(self.frequency, si.frequency)
        baud_rate = append(self.baud_rate, si.baud_rate)
        grid = append(self.grid, si.grid)
        signal = append(self.signal, si.signal)
        nli = append(self.nli, si.nli)
        ase = append(self.ase, si.ase)
        roll_off = append(self.roll_off, si.roll_off)
        chromatic_dispersion = append(self.chromatic_dispersion, si.chromatic_dispersion)
        pmd = append(self.pmd, si.pmd)
        number_of_channels = self.number_of_channels + si.number_of_channels
        power = lin2db(mean(signal) * 1e3)
        pref = Pref(power, power, lin2db(number_of_channels))
        try:
            si = SpectralInformation(frequency=frequency, grid=grid,
                                     signal=signal, nli=nli, ase=ase,
                                     baud_rate=baud_rate, roll_off=roll_off,
                                     chromatic_dispersion=chromatic_dispersion, pmd=pmd)
        except InfoError:
            raise InfoError('Spectra cannot be summed: channels overlapping.')
        return si

    def _replace(self, carriers, pref):
        self.chromatic_dispersion = array([c.chromatic_dispersion for c in carriers])
        self.pmd = array([c.pmd for c in carriers])
        self.signal = array([c.power.signal for c in carriers])
        self.nli = array([c.power.nli for c in carriers])
        self.ase = array([c.power.ase for c in carriers])
        self.pref = pref
        return self


def dimension_reshape(value, dimension, default=None, name=None):
    if value is None:
        if default is not None:
            value = dimension_reshape(value=default, dimension=dimension, name=name)
        else:
            raise InfoError(f'Missing mandatory field: {name}.')
    elif not isinstance(value, Sized):
        value = value * ones(dimension)
    else:
        if len(value) == 1:
            value = value[0] * ones(dimension)
        elif len(value) == dimension:
            value = squeeze(value)
        else:
            raise InfoError(f'Dimension mismatch field: {name}.')
    return value


def create_arbitrary_spectral_information(frequency, grid=None, signal=None, baud_rate=None,
                                          roll_off=None, chromatic_dispersion=None, pmd=None):
    """ Creates an arbitrary spectral information """
    if isinstance(frequency, Sized):
        frequency = squeeze(frequency)
        number_of_channels = len(frequency)
    else:
        frequency = array([frequency])
        number_of_channels = 1
    baud_rate = dimension_reshape(baud_rate, number_of_channels, 'baud rate')
    grid = dimension_reshape(grid, number_of_channels, ceil(baud_rate/BASE_GRID) * BASE_GRID, 'grid')
    signal = dimension_reshape(signal, number_of_channels, 0, 'signal')
    nli = zeros(number_of_channels)
    ase = zeros(number_of_channels)
    roll_off = dimension_reshape(roll_off, number_of_channels, 0, 'roll_off')
    chromatic_dispersion = dimension_reshape(chromatic_dispersion, number_of_channels, 0, 'chromatic dispersion')
    pmd = dimension_reshape(pmd, number_of_channels, 0, 'pmd')
    si = SpectralInformation(frequency=frequency, grid=grid,
                             signal=signal, nli=nli, ase=ase,
                             baud_rate=baud_rate, roll_off=roll_off,
                             chromatic_dispersion=chromatic_dispersion, pmd=pmd)
    return si


def create_input_spectral_information(f_min, f_max, roll_off, baud_rate, power, spacing):
    """ Creates a fixed grid spectral information with flat power """
    nb_channel = automatic_nch(f_min, f_max, spacing)
    frequency = [(f_min + spacing * i) for i in range(1, nb_channel + 1)]
    si = create_arbitrary_spectral_information(
        frequency, grid=spacing, signal=power, baud_rate=baud_rate, roll_off=roll_off
    )
    return si
