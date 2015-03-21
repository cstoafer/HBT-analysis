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
from hbtep.plot import (red_green_colormap, get_color, sxr_rainbow_colormap, 
    sxr_rainbow_colormap_w)
from hbtep import surfaceAnalysis
from hbtep.misc import cc_info, read_data
import MDSplus
import BandPassFilter as BandFilt
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator, MultipleLocator
from matplotlib.font_manager import fontManager, FontProperties
import copy
import signalAnalysis

class signalData:
    def __init__(self, times, signal, name):
        self._times = np.copy(times)
        self._signal = np.copy(signal)
        self._name = copy.deepcopy(name)

    @property
    def times(self):
        return self._times

    @times.setter
    def times(self, times):
        self._times = np.copy(times)

    @property
    def data(self):
        return self._signal

    @data.setter
    def data(self, signal):
        self._signal = np.copy(signal)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, sensors):
        self._name = copy.deepcopy(name)

    def plot(self, ax=None, color=None, multiplier=1.0):
        plot_mask = np.isfinite(self._signal)
        plot_times = np.copy(self._times)[plot_mask]*1e3
        plot_signal = multiplier*np.copy(self._signal)[plot_mask]
        if ax is None:
            if color is None:
                plt.plot(plot_times, plot_signal)
            else:
                plt.plot(plot_times, plot_signal, color)
        else:
            if color is None:
                ax.plot(plot_times, plot_signal)
            else:
                ax.plot(plot_times, plot_signal, color)

class magSensorData:
    def __init__(self, times, signals, sensors):
        self._times = np.copy(times)
        self._signals = np.copy(signals)
        self._sensors = copy.deepcopy(sensors)
        self.SSet = surfaceAnalysis.SensorSet()

    @property
    def times(self):
        return self._times

    @times.setter
    def times(self, times):
        self._times = np.copy(times)

    @property
    def data(self):
        return self._signals

    @data.setter
    def data(self, signals):
        self._signals = np.copy(signals)

    @property
    def sensors(self):
        return self._sensors

    @sensors.setter
    def sensors(self, sensors):
        self._sensors = copy.deepcopy(sensors)

    def plotContour(self, startTime, endTime, ax=None, windowLen=720, fitExtend=0.7):
        signals = signalAnalysis.subtract_background(self._times, self._signals, windowLen=windowLen, startTime=startTime, endTime=endTime, fitExtend=fitExtend)
        self.SSet.setNameList(self._sensors)
        if 'PA' in self._sensors[0]:
            angles = 180/np.pi*self.SSet.loc_Theta
        else:
            angles = self.SSet.loc_Phi
            angles[angles<0] += 2*np.pi
            if 'TA' in self._sensors[0]:
                angles = np.roll(180/np.pi*angles,-11)
                signals = np.roll(signals,-11,axis=0)
            elif 'FB' in self._sensors[0]:
                angles = np.roll(180/np.pi*angles,-4)
                signals = np.roll(signals,-4,axis=0)
        plot_mask = (self._times > startTime*1e-3) & (self._times < endTime*1e-3)
        sig_range = np.abs(signals[:,plot_mask]).max()
        if ax is None:
            plt.contourf(self._times[plot_mask]*1e3, angles, signals[:,plot_mask], 25, cmap=red_green_colormap(),
                norm=Normalize(vmin=-sig_range, vmax=sig_range))
        else:
            ax.contourf(self._times[plot_mask]*1e3, angles, signals[:,plot_mask], 25, cmap=red_green_colormap(),
                norm=Normalize(vmin=-sig_range, vmax=sig_range))


class shotData:
    def __init__(self, shotno, startTime, endTime, lowPassMach=None, lowPassModes=None, byrne=False):
        if shotno <= 0:
            self.shotno = shotno + MDSplus.Tree.getCurrent('hbtep2')
        else:
            self.shotno = shotno
        self.startTime = startTime if startTime < 50e-3 else startTime*1e-3
        self.endTime = endTime if endTime < 50e-3 else endTime*1e-3
        self.treeName = 'hbtep2'
        self.tree = MDSplus.Tree(self.treeName, self.shotno)
        self._data = {}
        self.times, _ = read_data(self.tree, '.sensors.rogowskis:ip', t_start=self.startTime, t_end=self.endTime)
        oh_bias_trig_node = self.tree.getNode('.timing.banks:oh_bias_st')
        self.oh_bias_trig = oh_bias_trig_node.data() * 1e-6
        self.lowPassMach = lowPassMach
        self.lowPassModes = lowPassModes
        self.byrne = byrne
        self.SSet = surfaceAnalysis.SensorSet()

    def getSignal(self, sigName, lowPassFreq=None, zeroSignal=False, store=True):
        if sigName not in self._data:
            signal = read_data(self.tree, sigName, times=self.times)
            if zeroSignal:
                signal = signalAnalysis.zeroSignal(self.times, self.oh_bias_trig, signal)
            if lowPassFreq:
                signal = signalAnalysis.filtSignal(self.times, signal, lowPassFreq)
            if store:
                self._data[sigName] = signal
        else:
            signal = self._data[sigName]

        return signalData(self.times, signal, sigName)

    def getMultSignals(self, sigList, lowPassFreq=None, zeroSignal=False, store=False):
        signals = np.empty((len(sigList), len(self.times)))
        for i, sig in enumerate(sigList):
            signals[i] = self.getSignal(sig, lowPassFreq=lowPassFreq, zeroSignal=zeroSignal, store=store).data

        return signalData(self.times, signals, sigList)

    def detMach(self, cal=[1.0, 0.629, .964, 1.324]):
        if 'machT' not in self._data:
            tipA_cur = self.getSignal('.sensors.mach_probe:tipA_c', lowPassFreq = self.lowPassMach, zeroSignal=True).data
            tipB_cur = self.getSignal('.sensors.mach_probe:tipB_c', lowPassFreq = self.lowPassMach, zeroSignal=True).data
            tipC_cur = self.getSignal('.sensors.mach_probe:tipC_c', lowPassFreq = self.lowPassMach, zeroSignal=True).data
            tipD_cur = self.getSignal('.sensors.mach_probe:tipD_c', lowPassFreq = self.lowPassMach, zeroSignal=True).data
            if cal:
                tipA_cur*=cal[0]
                tipB_cur*=cal[1]
                tipC_cur*=cal[2]
                tipD_cur*=cal[3]
            self._data['machT'] = 0.5 * np.log((tipB_cur + tipC_cur) / (tipA_cur + tipD_cur))
            self._data['machP'] = 0.5 * np.log((tipC_cur + tipD_cur) / (tipA_cur + tipB_cur))

        return (signalData(self.times,self._data['machT'],'machT'), signalData(self.times, self._data['machP'], 'machP'))

    def findStatsData(self,sigName,mask):
        '''Finds the average and standard deviation of the shot data in the mask'''
        signal = self.getSignal(sigName).data
        sigAvg = np.average(signal[mask])
        sigStd = np.std(signal[mask])

        return (sigAvg, sigStd)

    # The following properties are used for easier access to common signals
    @property
    def Ip(self):
        Ip = self.getSignal('.sensors.rogowskis:ip', zeroSignal=True)
        Ip.data *= 1e-3
        Ip.name = 'Ip'
        return Ip

    @property
    def MR(self):
        if 'MR' not in self._data:
            #start_calc_r = -1e-3
            _, self._data['MR'] = calc_r_major(self.tree, t_start=self.startTime, t_end=self.endTime, byrne = self.byrne)
        return signalData(self.times, self._data['MR'], 'MR')

    @property
    def q(self):
        if 'q' not in self._data:
            temp_times, temp_r =  calc_r_major(self.tree, t_start=self.startTime, t_end=self.endTime, byrne = self.byrne)
            self._data['q'] = calc_q(self.tree, temp_times, r_major=temp_r, byrne = self.byrne)
        return signalData(self.times, self._data['q'], 'q')

    @property
    def sxrfan(self):
        sxrfan = np.empty((15, len(self.times)))
        for i in np.arange(15):
            sxrfan[i,:] = self.getSignal('devices.west_rack:cpci:input_{0:02d}'.format(i+74)).data

        if self.shotno < 76879:
            sxrfan[9,:] *= 0
            sxrfan[13,:] *= 0
            sxrfan[14,:] *= 0
        elif self.shotno < 79898:
            sxrtemp = np.copy(sxrfan[9,:])
            sxrfan[9,:] = np.copy(sxrfan[14,:])
            sxrfan[14,:] = np.copy(sxrtemp)
        else:
            sxrfan[0,:] /= 8
            sxrtemp = np.copy(sxrfan[9,:])
            sxrfan[9,:] = np.copy(sxrfan[14,:])
            sxrfan[14,:] = np.copy(sxrtemp)

        return signalData(self.times, sxrfan, 'sxrfan')

    @property
    def sxrmid(self):
        sxrmid = self.getSignal('.devices.north_rack:cpci:input_74', zeroSignal=True)
        sxrmid.data *= -1
        sxrmid.name = 'sxrmid'

        return sxrmid

    @property
    def LV(self):
        lv = self.getSignal('.devices.west_rack:cpci:input_69', zeroSignal=True)
        lv.data *= -101.0
        lv.name = 'lv'

        return lv

    @property
    def OH_cur(self):
        oh_current = self.getSignal('.sensors:oh_current', zeroSignal=True)
        oh_current.name = 'oh_current'

        return oh_current

    @property
    def DAlpha(self):
        DAlpha = self.getSignal('.devices.screen_room:a14:input_1', zeroSignal=True)
        DAlpha.name = 'DAlpha'

        return DAlpha

    @property
    def bias_v(self):
        bias_v = self.getSignal('.sensors.bias_probe:voltage')
        bias_v.name = 'bias_v'

        return bias_v

    @property
    def bias_cur(self):
        bias_cur = self.getSignal('.sensors.bias_probe:current')
        bias_cur.name = 'bias_cur'

        return bias_cur

    @property
    def tipA_cur(self):
        tipA_cur = self.getSignal('.sensors.mach_probe:tipA_c', lowPassFreq = self.lowPassMach, zeroSignal=True)
        tipA_cur.name = 'tipA_cur'

        return tipA_cur

    @property
    def tipB_cur(self):
        tipB_cur = self.getSignal('.sensors.mach_probe:tipB_c', lowPassFreq = self.lowPassMach, zeroSignal=True)
        tipB_cur.name = 'tipB_cur'

        return tipB_cur

    @property
    def tipC_cur(self):
        tipC_cur = self.getSignal('.sensors.mach_probe:tipC_c', lowPassFreq = self.lowPassMach, zeroSignal=True)
        tipC_cur.name = 'tipC_cur'

        return tipC_cur

    @property
    def tipD_cur(self):
        tipD_cur = self.getSignal('.sensors.mach_probe:tipD_c', lowPassFreq = self.lowPassMach, zeroSignal=True)
        tipD_cur.name = 'tipD_cur'

        return tipD_cur

    @property
    def isat(self):
        isat = self.tipA_cur.data + self.tipB_cur.data + self.tipC_cur.data + self.tipD_cur.data

        return signalData(self.times, isat, 'isat')

    @property
    def tipE_cur(self):
        tipE_cur = self.getSignal('.sensors.mach_probe:tipE_c', lowPassFreq = self.lowPassMach, zeroSignal=True)
        tipE_cur.name = 'tipE_cur'

        return tipE_cur

    @property
    def tipE_v(self):
        tipE_v = self.getSignal('.sensors.mach_probe:tipE_v', zeroSignal=True)
        tipE_v.name = 'tipE_v'

        return tipE_v

    @property
    def machT(self):
        return self.detMach()[0]

    @property
    def machP(self):
        return self.detMach()[1]

    @property
    def FB1p(self):
        fb_sensors = argparse_sensors_type('FB*1p')
        fb_sensors = self.SSet.filterBadSensors(fb_sensors)
        sigList = ['.sensors.magnetic:{}'.format(sensor) for sensor in fb_sensors]
        fb_signals = self.getMultSignals(sigList, zeroSignal=True, store=False).data

        return magSensorData(self.times, fb_signals, fb_sensors)

    @property
    def FB2p(self):
        fb_sensors = argparse_sensors_type('FB*2p')
        fb_sensors = self.SSet.filterBadSensors(fb_sensors)
        sigList = ['.sensors.magnetic:{}'.format(sensor) for sensor in fb_sensors]
        fb_signals = self.getMultSignals(sigList, zeroSignal=True, store=False).data

        return magSensorData(self.times, fb_signals, fb_sensors)

    @property
    def FB3p(self):
        fb_sensors = argparse_sensors_type('FB*3p')
        fb_sensors = self.SSet.filterBadSensors(fb_sensors)
        sigList = ['.sensors.magnetic:{}'.format(sensor) for sensor in fb_sensors]
        fb_signals = self.getMultSignals(sigList, zeroSignal=True, store=False).data

        return magSensorData(self.times, fb_signals, fb_sensors)

    @property
    def FB4p(self):
        fb_sensors = argparse_sensors_type('FB*4p')
        fb_sensors = self.SSet.filterBadSensors(fb_sensors)
        sigList = ['.sensors.magnetic:{}'.format(sensor) for sensor in fb_sensors]
        fb_signals = self.getMultSignals(sigList, zeroSignal=True, store=False).data

        return magSensorData(self.times, fb_signals, fb_sensors)

    @property
    def FBallp(self):
        fb_sensors = []
        fb_signals = []
        for i in np.arange(4):
            fb_sensors.append(argparse_sensors_type('FB*{}p'.format(i+1)))
            fb_sensors[i] = self.SSet.filterBadSensors(fb_sensors[i])
            sigList = ['.sensors.magnetic:{}'.format(sensor) for sensor in fb_sensors[i]]
            fb_signals.append(self.getMultSignals(sigList, zeroSignal=True, store=False).data)

        return [ magSensorData(self.times, fb_signals[i], fb_sensors[i]) for i in np.arange(4) ]

    @property
    def TAp(self):
        sensors = argparse_sensors_type('TA*p')
        sensors = self.SSet.filterBadSensors(sensors)
        sigList = ['.sensors.magnetic:{}'.format(sensor) for sensor in sensors]
        signals = self.getMultSignals(sigList, zeroSignal=True, store=False).data

        return magSensorData(self.times, signals, sensors)

    @property
    def PA1p(self):
        sensors = argparse_sensors_type('PA1*p')
        sensors = self.SSet.filterBadSensors(sensors)
        sigList = ['.sensors.magnetic:{}'.format(sensor) for sensor in sensors]
        signals = self.getMultSignals(sigList, zeroSignal=True, store=False).data

        return magSensorData(self.times, signals, sensors)

    @property
    def PA2p(self):
        sensors = argparse_sensors_type('PA2*p')
        sensors = self.SSet.filterBadSensors(sensors)
        sigList = ['.sensors.magnetic:{}'.format(sensor) for sensor in sensors]
        signals = self.getMultSignals(sigList, zeroSignal=True, store=False).data

        return magSensorData(self.times, signals, sensors)

    # Radial magnetic sensors -----------------------------
    @property
    def FB1r(self):
        fb_sensors = argparse_sensors_type('FB*1r')
        fb_sensors = self.SSet.filterBadSensors(fb_sensors)
        sigList = ['.sensors.magnetic:{}'.format(sensor) for sensor in fb_sensors]
        fb_signals = self.getMultSignals(sigList, zeroSignal=True, store=False).data

        return magSensorData(self.times, fb_signals, fb_sensors)

    @property
    def FB2r(self):
        fb_sensors = argparse_sensors_type('FB*2r')
        fb_sensors = self.SSet.filterBadSensors(fb_sensors)
        sigList = ['.sensors.magnetic:{}'.format(sensor) for sensor in fb_sensors]
        fb_signals = self.getMultSignals(sigList, zeroSignal=True, store=False).data

        return magSensorData(self.times, fb_signals, fb_sensors)


    @property
    def FB3r(self):
        fb_sensors = argparse_sensors_type('FB*3r')
        fb_sensors = self.SSet.filterBadSensors(fb_sensors)
        sigList = ['.sensors.magnetic:{}'.format(sensor) for sensor in fb_sensors].data
        fb_signals = self.getMultSignals(sigList, zeroSignal=True, store=False)

        return magSensorData(self.times, fb_signals, fb_sensors)


    @property
    def FB4r(self):
        fb_sensors = argparse_sensors_type('FB*4r')
        fb_sensors = self.SSet.filterBadSensors(fb_sensors)
        sigList = ['.sensors.magnetic:{}'.format(sensor) for sensor in fb_sensors]
        fb_signals = self.getMultSignals(sigList, zeroSignal=True, store=False).data

        return magSensorData(self.times, fb_signals, fb_sensors)

    @property
    def FBallr(self):
        fb_sensors = []
        fb_signals = []
        for i in np.arange(4):
            fb_sensors.append(argparse_sensors_type('FB*{}r'.format(i+1)))
            fb_sensors[i] = self.SSet.filterBadSensors(fb_sensors[i])
            sigList = ['.sensors.magnetic:{}'.format(sensor) for sensor in fb_sensors[i]]
            fb_signals.append(self.getMultSignals(sigList, zeroSignal=True, store=False).data)

        return [ magSensorData(self.times, fb_signals[i], fb_sensors[i]) for i in np.arange(4) ]

    @property
    def TAr(self):
        sensors = argparse_sensors_type('TA*r')
        sensors = self.SSet.filterBadSensors(sensors)
        sigList = ['.sensors.magnetic:{}'.format(sensor) for sensor in sensors]
        signals = self.getMultSignals(sigList, zeroSignal=True, store=False).data

        return magSensorData(self.times, signals, sensors)

    @property
    def PA1r(self):
        sensors = argparse_sensors_type('PA1*r')
        sensors = self.SSet.filterBadSensors(sensors)
        sigList = ['.sensors.magnetic:{}'.format(sensor) for sensor in sensors]
        signals = self.getMultSignals(sigList, zeroSignal=True, store=False).data

        return magSensorData(self.times, signals, sensors)

    @property
    def PA2r(self):
        sensors = argparse_sensors_type('PA2*r')
        sensors = self.SSet.filterBadSensors(sensors)
        sigList = ['.sensors.magnetic:{}'.format(sensor) for sensor in sensors]
        signals = self.getMultSignals(sigList, zeroSignal=True, store=False).data

        return magSensorData(self.times, signals, sensors)

    def detFB_amp_freq(self, cutoffFreq=0.3e3, window=50):
        FB_amp_list = []
        FB_freq_list = []
        filt_order = 2
        for FBarray in self.FBallp:
            FB_n1 = signalAnalysis.detN1(FBarray.data, FBarray.sensors)
            FB_n1 = BandFilt.HighPassProcess(FB_n1, self.times, filt_order, cutoffFreq)
            FB_amp_list.append(np.sqrt(FB_n1[0]**2 + FB_n1[1]**2))
            FB_phase = np.arctan2(FB_n1[1], FB_n1[0])
            FB_freq_list.append(np.convolve(np.unwrap(FB_phase), signalAnalysis.slope_fit(window), 'same') / (2*np.pi*2e-3))
        FB_amp = np.average(FB_amp_list, axis=0)*1e3
        FB_freq = np.average(FB_freq_list, axis=0)

        return (FB_amp, FB_freq)

    @property
    def FBamp(self):
        if 'FBamp' not in self._data:
            FB_amp, FB_freq = self.detFB_amp_freq(cutoffFreq=0.4e3, window=50)
            self._data['FBamp'] = FB_amp
            self._data['FBfreq'] = FB_freq

        return signalData(self.times, self._data['FBamp'], 'FBamp')

    @property
    def FBfreq(self):
        if 'FBfreq' not in self._data:
            FB_amp, FB_freq = self.detFB_amp_freq(cutoffFreq=0.1e3, window=50)
            self._data['FBamp'] = FB_amp
            self._data['FBfreq'] = FB_freq

        return signalData(self.times, self._data['FBfreq'], 'FBfreq')
