#!/usr/bin/env python
'''
Created on Mar 16, 2015

@author: cstoafer
'''
import numpy as np
from hbtep.mdsplus import read_data, signals_from_shot
from hbtep.misc import (init_logging, argparse_sensors_type, extract_equilibrium, 
    calc_q, calc_r_major, BadSensor, argparse_pyfile_type, sensor_blacklist, 
    true_regions, poly_fit)
from hbtep import surfaceAnalysis
from hbtep.misc import cc_info, read_data
import MDSplus
import BandPassFilter as BandFilt
import copy

def filtSignal(times, signal, lowPassFreq):
    if lowPassFreq < 100:
        lowPassFreq*=1e3
    signal = BandFilt.LowPassProcess(signal, times, order=1, freq=lowPassFreq)

    return signal

def zeroSignal(times, trigTime, signal):
    zero_mask = times < trigTime
    return signal - signal[zero_mask].mean()

def smooth(signals, window_len):
    sig_equil = np.zeros(signals.shape)
    window = np.hamming(window_len)
    window = window / window.sum()
    for i in np.arange(signals.shape[0]):
        sig_equil[i,:]=np.convolve(window, signals[i,:],mode='same')

    return sig_equil

def polyFit(times, signals, deg):
    poly_signals = np.polyfit(times, np.transpose(signals), deg)
    sig_equil = np.zeros(signals.shape)
    for i in np.arange(deg+1):
        A
        sig_equil += np.outer(poly_signals[i,:], times**(deg-i))

    return sig_equil

def subtract_background(times, signals, subtr_avg=True, fit='smooth', window_len=720, polyDeg=4, zoomStart=2, zoomEnd=4, fitExtend=0.7):
    sigs_temp = np.copy(signals)
    if subtr_avg:
        avgSigs = np.average(signals, axis=0)
        # Remove n=0 or m=0 component of signals
        sigs_temp = signals-avgSigs

    fitmask = (times > (zoomStart - fitExtend)*1e-3) & (times < (zoomEnd + fitExtend)*1e-3)
    sig_equil = np.zeros(signals.shape)
    if fit == 'smooth':
        sig_equil[:,fitmask] = smooth(sigs_temp[:,fitmask], window_len)
    if fit == 'polynomial':
        sig_equil[:,fitmask] = polyFit(sigs_temp[:,fitmask], polyDeg)
    sigs_temp -= sig_equil
    sigs_temp[:,~fitmask] = 0.0

    return sigs_temp

def detN1(signals, sensors):
    SSet = surfaceAnalysis.SensorSet()
    SSet.setNameList(sensors)
    phi = SSet.loc_Phi
    A = np.zeros([len(sensors), 2])
    A[:,0] = np.cos(-phi)
    A[:,1] = np.sin(-phi)
    B = np.linalg.pinv(A)

    return np.dot(B,signals)

