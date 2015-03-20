#!/usr/bin/env python
'''
Plot summary of one or more shots.
'''

from __future__ import print_function, absolute_import, division
from argparse import ArgumentParser
from hbtep.find_shots import overlaps, BadShot, get_steady_regions
from hbtep.mdsplus import read_data, signals_from_shot
from hbtep.misc import (init_logging, argparse_sensors_type, extract_equilibrium, 
    calc_q, calc_r_major, BadSensor, argparse_pyfile_type, sensor_blacklist, 
    true_regions, poly_fit)
from hbtep.plot import (red_green_colormap, get_color, sxr_rainbow_colormap, 
    sxr_rainbow_colormap_w)
from hbtep import surfaceAnalysis
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator, MultipleLocator
from matplotlib.font_manager import fontManager, FontProperties
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as PathEffects
import MDSplus
import atexit
import logging
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import subprocess
import threading

import BandPassFilter as BandFilt
log = logging.getLogger('')


def parse_args(args=None):
    '''Parse command line'''

    parser = ArgumentParser()

    parser.add_argument("shotnos", metavar='<shot number>', type=int, nargs='+',
                      help="Shots to plot.  The contour plots are generated from the last shot listed")

    parser.add_argument("--end-time", metavar='<milliseconds>', type=float, default=8,
                      help="Plot this many seconds. Default %(default)s")

    parser.add_argument("--zoom-start", metavar='<milliseconds>', type=float, default=2,
                      help="The start of the zoomed window for the stripy plots")

    parser.add_argument("--zoom-end", metavar='<milliseconds>', type=float, default=4,
                      help="The end of the zoomed window for the stripy plots")

    parser.add_argument("--loop", action='store_true', default=False,
                      help="Redraw whenever hbtep2_complete MDS event is received.")
    
    parser.add_argument('--small',action='store_true',default=False,
                        help="Use a smaller window instead of the large display format")

    parser.add_argument('--short-title',action='store_true',default=False,
                        help="Removes the Last Shot Taken part of the title")

    parser.add_argument('--white',action='store_true',default=False,
                        help="Plots with a white background instead of black")

    parser.add_argument('--printable',action='store_true',default=False,
                        help="Creates a small plot with a white background for printing")

    parser.add_argument("--debug", action="store_true", default=False,
                      help="Activate debugging output")
    
    parser.add_argument("--quiet", action="store_true", default=False,
                      help="Be really quiet")

    parser.add_argument("--byrne", action="store_true", default=False,
                        help="Use Pat's subtraction method")

    parser.add_argument("--thomson", action="store_true", default=False,
                        help="Plot Te and Ne from Thomson data")

    parser.add_argument("--rot-probes", action="store_true", default=False,
                        help="Plot voltage and current of bias probe, as well as signals from Mach probe.")

    parser.add_argument("--plotsensors", action="store_true", default=False,
                        help="Plot individual magnetic sensors in separate figures")

    parser.add_argument("--no-avg-subtr", action="store_true", default=False,
                        help="Do not subtract the n=0 or m=0 components from the sensors.  This reduces the visibility of slowed modes.")

    parser.add_argument("--polynomial", action="store_true", default=False,
                        help="Use polynomial for equilibrium subtraction for contour plots")

    parser.add_argument("--plot-n1-sensors", action="store_true", default=False,
                        help="Plot the signal from subtraction of each n=1 pair of sensors.")

    parser.add_argument("--extra-fig", action="store_true", default=False,
                        help="Generate extra figure window with custom plots. Used for getting figures to show.")

    options = parser.parse_args(args)

    return options

def slope_fit(pts_cnt):
    '''Return weights to determine slope from *pts* points'''

    A = np.empty((pts_cnt, 2))
    for i in np.arange(2):
        A[:,i] = np.arange(pts_cnt)**i

    A_i = np.linalg.pinv(A)

    return A_i[1,::-1]

def smoothe(data, window_len):
    window = np.hamming(window_len)
    window = window / window.sum()
    #return np.convolve(window, data, mode='full')[window_len//2:len(data)+window_len//2]
    return np.convolve(window, data, mode='same')

def setup_ax_arr(fig, numV, numH, ylim, sharey=True, hide_xlabels=True, hide_ylabels=True):
    # hide_label option only applies to subplots on the left most and bottom, other labels will always be hided.
    ax_arr = []
    for i in np.arange(numV*numH):
        if i==0:
            ax_arr.append(fig.add_subplot(numV,numH,i+1))
        else:
            if sharey:
                ax_arr.append(fig.add_subplot(numV,numH,i+1, sharex=ax_arr[0], sharey=ax_arr[0]))
            else:
                ax_arr.append(fig.add_subplot(numV,numH,i+1, sharex=ax_arr[0]))
        ax_arr[i].set_ylim(ylim[0],ylim[1])
        if (not hide_xlabels) and ( i > (numV-1)*numH):
            pass
        else:
            for label in ax_arr[i].get_xticklabels():
                label.set_visible(False)
        if (not hide_ylabels) and ( (i-1) % numH == 0):
            pass
        else:
            for label in ax_arr[i].get_yticklabels():
                label.set_visible(False)
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)

    return ax_arr


def setup_axes(axes, ylim, ylabel, aligny=False, xlim=None, xlabel=None, ygrid=False, num_major_yticks=5, xtick_labels=False,
        vlines=[], hlines=[]):
    if ylim: axes.set_ylim(ylim[0], ylim[1])
    if ylabel: axes.set_ylabel(ylabel, multialignment='center')#, ha='center')
    if aligny: axes.yaxis.set_label_coords(-.3, 0.5)
    if xlim: axes.set_xlim(xlim[0], xlim[1])
    if xlabel: axes.set_xlabel(xlabel)
    if ygrid: axes.yaxis.grid(b=True)
    if num_major_yticks: axes.yaxis.set_major_locator(MaxNLocator(num_major_yticks))
    if not xtick_labels:
        for label in axes.get_xticklabels():
            label.set_visible(False)
    linecolor = 'k' if options.white else 'w'
    for vline in vlines:
        axes.axvline(vline, ls='--', color=linecolor, alpha=0.8)
    for hline in hlines:
        axes.axhline(hline, ls='-.', color=linecolor, alpha=0.8)


def get_puff_pressures(thistree):
	nodeloc_pufftrig = '.devices.basement:a14_4:input_6'
	nodeloc_iongauge = '.devices.basement:a14_04:input_5'
	voltagedivider = 2.0286    # ********************************** INCLUDES CORRECTION FOR A14 INPUT IMPEDANCE!
	t_puff_1,t_puff_2 = ((thistree.getNode('.timing:gas_puff')).data())*10**(-6)
	t_basementA14start = ((thistree.getNode('.timing.digitizer:basement_tf')).data())*10**(-6)
	t_tfbankstart = ((thistree.getNode('.timing.banks:TF_ST')).data())*10**(-6)
	p_base_times = [t_puff_1-0.001-1./60.,t_puff_1-0.001]
	p_fill_times = [t_tfbankstart-0.006-1./60.,t_tfbankstart-0.006]
	p_IG_rawsig = (thistree.getNode(nodeloc_iongauge)).data()
	t_p_IG = (thistree.getNode(nodeloc_iongauge)).dim_of().data()
	dt = t_p_IG[1]-t_p_IG[0]
	p_IG_smth = smoothe(p_IG_rawsig,250)
	p_IG_proc = 10**(abs(p_IG_smth)*(voltagedivider) - 2.)
	meanbasepress = np.mean(p_IG_proc[int(round( (p_base_times[0] - t_p_IG[0])/dt ) ):int(round( (p_base_times[1] - t_p_IG[0])/dt ) )])/0.35
	meanfillpress = np.mean(p_IG_proc[int(round( (p_fill_times[0] - t_p_IG[0])/dt ) ):int(round( (p_fill_times[1] - t_p_IG[0])/dt ) )])/0.35
	return (t_puff_2-t_puff_1)*10**6,meanbasepress,meanfillpress  # in nTorr

def determine_background(times, signals):
    deg = 4     # degree for polynomial fit
    sm_win = 720 if not options.no_avg_subtr else 120        # Window used for smoothing

    if options.polynomial:
        poly_signals = np.polyfit(times, np.transpose(signals), deg)
        sig_equil = np.zeros(signals.shape)
        for i in np.arange(deg+1):
            sig_equil += np.outer(poly_signals[i,:], times**(deg-i))
    else:
        sig_equil = np.zeros(signals.shape)
        window = np.hamming(sm_win)
        window = window / window.sum()
        for i in np.arange(signals.shape[0]):
            sig_equil[i,:]=np.convolve(window, signals[i,:],mode='same')#smoothe(sigs_temp[i,fitmask],sm_win)
    
    return sig_equil

def subtract_background(times, signals):
    sigs_temp = signals
    if not options.no_avg_subtr:
        avgSigs = np.average(signals, axis=0)
        # Remove n=0 or m=0 component of signals
        sigs_temp = signals-avgSigs

    sig_equil = determine_background(times, sigs_temp)
    plot_signals = sigs_temp - sig_equil

    plotmask = (times >= (options.zoom_start)/1000)&(times < (options.zoom_end)/1000)

    plot_times = times[plotmask]
    plot_signals = plot_signals[:,plotmask]

    return (plot_times, plot_signals)

def plot_signals(times, signals, ax_arr, color, vlines=[]):
    linecolor = 'k' if options.white else 'w'
    for (i,signal) in enumerate(signals):
        ax_arr[i].plot(times*1e3,signal, color=color)
        for line in vlines:
            ax_arr[i].axvline(line, ls='--',color=linecolor, alpha=0.8)

def plot_sensors(times, signals, ax_arr, avgSigs, fitmask, vlines=[]):
    plotcolor = 'k' if options.white else 'w'
    plot_signals(times, signals, ax_arr, plotcolor, vlines)
    sigs_temp = signals
    if not options.no_avg_subtr:
        sigs_temp = signals-avgSigs
        for i in np.arange(signals.shape[0]):
            ax_arr[i].plot(times*1e3, avgSigs, color='r')
        plot_signals(times, sigs_temp, ax_arr, plotcolor)

    sig_equil = np.zeros(sigs_temp.shape)
    sig_equil[:,fitmask] = determine_background(times[fitmask], sigs_temp[:,fitmask])

    plot_signals(times, sig_equil, ax_arr, color='r')
    plot_signals(times, sigs_temp-sig_equil, ax_arr, color='g')

def extra_fig(SSet):
    shotno = options.shotnos[0] if len(options.shotnos) == 1 else options.shotnos[-1]
    fig_ex = plt.figure(figsize=(7,12))    
    
#    biasv_num   = 3
#    biasc_num   = 4
#    q_num       = 0
#    TAn1_num    = 1
#    FB4n1_num   = 2
#    lv_num      = 5
#    mid_sxr_num = 6
#    sxrfan_num  = 7
    fig_name_list = [\
                     'MR',
                     'q',
                     'mid_sxr',
                     'sxrfan',
                     'lv',
                     'TAn1',
                     'FB4n1',
                     'biasv',
                     'biasc',
                     'macht',
                     'machp',
                     'isat',
                     'tipEv'
                     ]
    
#    fig_name_list = ['q','TAn1','FB4n1','biasv','biasc','lv','mid_sxr','sxrfan','macht','machp','isat','tipEv]

    numV = len(fig_name_list)
    numH = 1
    ax_arr = setup_ax_arr(fig_ex, numV, numH, [0,1], sharey=False, hide_xlabels=False, hide_ylabels=False)
    #for label in ax_arr[numV-1].get_xticklabels():
    #    label.set_visible(True)
    fig_ex.subplots_adjust(hspace=0.04, left=0.14, right=0.96, top=0.98, bottom = 0.12)

    if 'biasv' in fig_name_list:
        biasv_num = fig_name_list.index('biasv')
        setup_axes(axes=ax_arr[biasv_num], ylim=[-125,125], ylabel='Probe\nVoltage',num_major_yticks=3)
    if 'biasc' in fig_name_list:
        biasc_num = fig_name_list.index('biasc')
        setup_axes(axes=ax_arr[biasc_num], ylim=[-10,80], ylabel='Probe\nCurrent',num_major_yticks=3)
    if 'TAn1' in fig_name_list:
        TAn1_num = fig_name_list.index('TAn1')
        setup_axes(axes=ax_arr[TAn1_num], ylim=[-19.9,19.9], ylabel='TA n=1',num_major_yticks=4)
    if 'FB4n1' in fig_name_list:
        FB4n1_num = fig_name_list.index('FB4n1')
        setup_axes(axes=ax_arr[FB4n1_num], ylim=[-19.9,19.9], ylabel='FB4 n=1',num_major_yticks=4)
    if 'lv' in fig_name_list:
        lv_num = fig_name_list.index('lv')
        setup_axes(axes=ax_arr[lv_num], ylim=[0,15], ylabel='Loop\nVoltage',num_major_yticks=4)
    if 'mid_sxr' in fig_name_list:
        mid_sxr_num = fig_name_list.index('mid_sxr')
        setup_axes(axes=ax_arr[mid_sxr_num], ylim=[0,2], ylabel='SXR\nmidplane',num_major_yticks=3)
    if 'sxrfan' in fig_name_list:
        sxrfan_num = fig_name_list.index('sxrfan')
        setup_axes(axes=ax_arr[sxrfan_num], ylim=[1,15], ylabel='SXR Fan', num_major_yticks=4)
    if 'q' in fig_name_list:
        q_num = fig_name_list.index('q')
        setup_axes(axes=ax_arr[q_num], ylim=[2.01,4], ylabel='Edge q',num_major_yticks=4)
    if 'Ip' in fig_name_list:
        Ip_num = fig_name_list.index('Ip')
        setup_axes(axes = ax_arr[Ip_num], ylim=[0,20], ylabel = 'Ip (kA)', num_major_yticks = 4)
    if 'MR' in fig_name_list:
        MR_num = fig_name_list.index('MR')
        setup_axes(axes = ax_arr[MR_num], ylim = [90,97], ylabel = 'MR (cm)', num_major_yticks = 4)
    if 'macht' in fig_name_list:
        macht_num = fig_name_list.index('macht')
        setup_axes(axes=ax_arr[macht_num], ylim=[-0.5,0.5], ylabel='Mach Tor',num_major_yticks=4, ygrid=True)
    if 'machp' in fig_name_list:
        machp_num = fig_name_list.index('machp')
        setup_axes(axes=ax_arr[machp_num], ylim=[-0.5,0.5], ylabel='Mach Pol',num_major_yticks=4, ygrid=True)
    if 'isat' in fig_name_list:
        isat_num = fig_name_list.index('isat')
        setup_axes(axes=ax_arr[isat_num], ylim=[-10,150], ylabel='$I_{sat}$',num_major_yticks=4)
    if 'tipEv' in fig_name_list:
        tipEv_num = fig_name_list.index('tipEv')
        setup_axes(axes=ax_arr[tipEv_num], ylim=[-60,40], ylabel='$V_{tipE}$',num_major_yticks=4)
    
    if (len(ax_arr) != 0):
        ax_arr[0].set_xlim([options.zoom_start,options.zoom_end])    

    for i,label in enumerate(ax_arr[numV-1].get_xticklabels()):
        if i != 0:
            label.set_visible(True)
    ax_arr[numV-1].set_xlabel('Time (ms)')
    tree = MDSplus.Tree('hbtep2', shotno)
    oh_bias_trig_node = tree.getNode('.timing.banks:oh_bias_st')
    oh_bias_trig = oh_bias_trig_node.data() * 1e-6

    if 'biasv' in fig_name_list:
        (times, biasv) = read_data(tree, '.sensors.bias_probe:voltage', t_start=0, t_end=options.end_time)
        #ax_arr[0].plot(times[plot_mask]*1e3, biasv[plot_mask], color='k')
        ax_arr[biasv_num].plot(times*1e3, biasv, color='k')

    if 'biasc' in fig_name_list:
        (times, biasc) = read_data(tree, '.sensors.bias_probe:current', t_start=0, t_end=options.end_time)
        #ax_arr[1].plot(times[plot_mask]*1e3, biasc[plot_mask], color='k')
        ax_arr[biasc_num].plot(times*1e3, biasc, color='k')

    if 'macht' in fig_name_list:
        (times, tipA_c) = read_data(tree, '.sensors.mach_probe:tipa_c', t_start=0, t_end=options.end_time)
        (times, tipB_c) = read_data(tree, '.sensors.mach_probe:tipb_c', t_start=0, t_end=options.end_time)
        (times, tipC_c) = read_data(tree, '.sensors.mach_probe:tipc_c', t_start=0, t_end=options.end_time)
        (times, tipD_c) = read_data(tree, '.sensors.mach_probe:tipd_c', t_start=0, t_end=options.end_time)
        tipA_c = BandFilt.LowPassProcess(tipA_c, times, order=1, freq=5.0e3)
        tipB_c = BandFilt.LowPassProcess(tipB_c, times, order=1, freq=5.0e3)
        tipC_c = BandFilt.LowPassProcess(tipC_c, times, order=1, freq=5.0e3)
        tipD_c = BandFilt.LowPassProcess(tipD_c, times, order=1, freq=5.0e3)
        tipB_c*=0.629
        tipC_c*=0.964
        tipD_c*=1.324
        machT = .5*np.log((tipB_c + tipC_c) / (tipA_c + tipD_c))
        machT_mask = np.isfinite(machT)
        ax_arr[macht_num].plot(times*1e3, machT, color='k')

    if 'machp' in fig_name_list:
        (times, tipA_c) = read_data(tree, '.sensors.mach_probe:tipa_c', t_start=0, t_end=options.end_time)
        (times, tipB_c) = read_data(tree, '.sensors.mach_probe:tipb_c', t_start=0, t_end=options.end_time)
        (times, tipC_c) = read_data(tree, '.sensors.mach_probe:tipc_c', t_start=0, t_end=options.end_time)
        (times, tipD_c) = read_data(tree, '.sensors.mach_probe:tipd_c', t_start=0, t_end=options.end_time)
        tipA_c = BandFilt.LowPassProcess(tipA_c, times, order=1, freq=5.0e3)
        tipB_c = BandFilt.LowPassProcess(tipB_c, times, order=1, freq=5.0e3)
        tipC_c = BandFilt.LowPassProcess(tipC_c, times, order=1, freq=5.0e3)
        tipD_c = BandFilt.LowPassProcess(tipD_c, times, order=1, freq=5.0e3)
        tipB_c*=0.629
        tipC_c*=0.964
        tipD_c*=1.324
        machP = .5*np.log((tipC_c + tipD_c) / (tipA_c + tipB_c))
        machP_mask = np.isfinite(machP)
        ax_arr[machp_num].plot(times*1e3, machP, color='k')

    if 'isat' in fig_name_list:
        (times, isat) = read_data(tree, '.sensors.mach_probe:tipe_c', t_start=0, t_end=options.end_time)
        isat = BandFilt.LowPassProcess(isat, times, order=1, freq=5.0e3)
        ax_arr[isat_num].plot(times*1e3, isat*1e3, color='k')

    if 'tipEv' in fig_name_list:
        (times, tipEv) = read_data(tree, '.sensors.mach_probe:tipe_v', t_start=0, t_end=options.end_time)
        tipEv = BandFilt.LowPassProcess(tipEv, times, order=1, freq=5.0e3)
        ax_arr[tipEv_num].plot(times*1e3, tipEv, color='k')

    if 'lv' in fig_name_list:
        (times, lv) = read_data(tree, '.devices.west_rack:cpci:input_69', t_start=0,
                                   t_end=options.end_time)
        lvmask = times < oh_bias_trig
        lv = -lv*101.0
        lv -= lv[lvmask].mean()
        ax_arr[lv_num].plot(times*1e3, lv, color='k')

    if 'TAn1' in fig_name_list:
        (times, TA01) = read_data(tree, '.sensors.magnetic:TA01_S1P', t_start=0, t_end=options.end_time)
        (times, TA09) = read_data(tree, '.sensors.magnetic:TA03_S3P', t_start=0, t_end=options.end_time)
        (times, TA17) = read_data(tree, '.sensors.magnetic:TA06_S2P', t_start=0, t_end=options.end_time)
        (times, TA25) = read_data(tree, '.sensors.magnetic:TA09_S1P', t_start=0, t_end=options.end_time)
        plot_mask = (times > options.zoom_start*1e-3) & (times < options.zoom_end*1e-3)
        TAn1_sin = (TA01 - TA17)*1e4
        #TAn1_sin -= np.average(TAn1_sin)
        TAn1_sin -= np.average(TAn1_sin[plot_mask])
        TAn1_cos = (TA09 - TA25)*1e4
        TAn1_cos -= np.average(TAn1_cos[plot_mask])
        ax_arr[TAn1_num].plot(times*1e3, TAn1_sin, color='k')
        ax_arr[TAn1_num].plot(times*1e3, TAn1_cos, color='b')

    if 'FB4n1' in fig_name_list:
        (times, FB01) = read_data(tree, '.sensors.magnetic:FB01_S1P', t_start=0, t_end=options.end_time)
        (times, FB04) = read_data(tree, '.sensors.magnetic:FB03_S1P', t_start=0, t_end=options.end_time)
        (times, FB06) = read_data(tree, '.sensors.magnetic:FB06_S1P', t_start=0, t_end=options.end_time)
        (times, FB09) = read_data(tree, '.sensors.magnetic:FB08_S1P', t_start=0, t_end=options.end_time)
        plot_mask = (times > options.zoom_start*1e-3) & (times < options.zoom_end*1e-3)
        FBn1_sin = (FB01 - FB06)*1e4
        #TAn1_sin -= np.average(TAn1_sin)
        FBn1_sin -= np.average(FBn1_sin[plot_mask])
        FBn1_cos = (FB04 - FB09)*1e4
        FBn1_cos -= np.average(FBn1_cos[plot_mask])
        ax_arr[FB4n1_num].plot(times*1e3, FBn1_sin, color='k')
        ax_arr[FB4n1_num].plot(times*1e3, FBn1_cos, color='b')

    if 'mid_sxr' in fig_name_list:
        (times, sxr) = read_data(tree, '.devices.north_rack:cpci:input_74', t_start=0,
                                t_end=options.end_time)
        sxrmask = times < oh_bias_trig
        sxr -= sxr[sxrmask].mean()
        sxr = -sxr
        ax_arr[mid_sxr_num].plot(times*1e3, sxr, color='k')

    if 'sxrfan' in fig_name_list:
        for i in np.arange(15):
            (times, sxr_ch) = read_data(tree, '.devices.west_rack:cpci:input_%02d' %(i+74), t_start=0,
                                t_end=options.end_time)

            if i == 0:
                sxr_array = np.empty((15, len(times)))

            sxr_array[i,:]=sxr_ch

        if shotno < 76879:
            sxr_array[9,:] *=0
            sxr_array[13,:] *=0
            sxr_array[14,:] *=0
        elif shotno > 79898:
            sxr_array[0,:] /= 8
            sxtemp=np.copy(sxr_array[9,:])
            sxr_array[9,:]=np.copy(sxr_array[14,:])
            sxr_array[14,:]=np.copy(sxtemp)
        else:
            sxtemp=np.copy(sxr_array[9,:])
            sxr_array[9,:]=np.copy(sxr_array[14,:])
            sxr_array[14,:]=np.copy(sxtemp)

        avemask = times > 1.5e-3
        plot_mask = (times > options.zoom_start*1e-3) & (times < options.zoom_end*1e-3)
        sig_max = np.abs(sxr_array[:,plot_mask]).max()
        sig_min = 0.03*sig_max
        if options.white:
            ax_arr[sxrfan_num].contourf(times*1e3,np.arange(15)+1,sxr_array,128,cmap=sxr_rainbow_colormap_w(128),
                norm=Normalize(vmin=sig_min, vmax=sig_max))
        else:
            ax_arr[sxrfan_num].contourf(times*1e3,np.arange(15)+1,sxr_array,128,cmap=sxr_rainbow_colormap(128),
                norm=Normalize(vmin=sig_min, vmax=sig_max))

    if 'q' in fig_name_list:
        q_num = fig_name_list.index('q')
        (times, r) = calc_r_major(tree, t_start=-.001, t_end=options.end_time, byrne = options.byrne)
        rmask = ~np.isnan(r)
        q = calc_q(tree, times, r_major=r, byrne = options.byrne)
        qmask = ~np.isnan(q)
        ax_arr[q_num].plot(times[qmask]*1e3, q[qmask], color='k')
        ax_arr[q_num].axhline(3,ls='-.',color='k')

    if 'Ip' in fig_name_list:
        ip_num = fig_name_list.index('Ip')
        (times, ip) = read_data(tree, '.sensors.rogowskis:ip', t_start=0, 
                            t_end=options.end_time)
        ax_arr[ip_num].plot(times*1e3, ip/1e3, label='%d' % (shotno), color='k')
 
    if 'MR' in fig_name_list:
        MR_num = fig_name_list.index('MR')
        (times, r) = calc_r_major(tree, t_start=-.001, t_end=options.end_time, byrne = options.byrne)
        rmask = ~np.isnan(r)
        ax_arr[MR_num].plot(times[rmask]*1e3, r[rmask]*100, color='k')
        ax_arr[MR_num].axhline(92,ls='-.',color = 'k')

    if 'Ip' in fig_name_list:
        ip_num = fig_name_list.index('Ip')
        ax_arr[ip_num].legend(loc = 'best')

def plot_shot(shotno, color, SSet, ip_ax, r_ax, q_ax, oh_ax, amp_ax, freq_ax, lv_ax, sp_ax, sxr_ax, last_shot,
                sxfan_ax, tap_ax, pa1_ax, pa2_ax, fb4_ax, Te_ax, Ne_ax, biasv_ax, biasc_ax, machT_ax, machP_ax, isat_ax, tipEv_ax, TAax_arr,
                PA1ax_arr, PA2ax_arr, FBax_arr, TA_n1ax_arr, FB_n1ax_arr):

    if last_shot:
        color='k' if options.white else 'w'

    tree = MDSplus.Tree('hbtep2', shotno)
    oh_bias_trig_node = tree.getNode('.timing.banks:oh_bias_st')
    oh_bias_trig = oh_bias_trig_node.data() * 1e-6
 
    (times, ip) = read_data(tree, '.sensors.rogowskis:ip', t_start=0, 
                            t_end=options.end_time)

    pufftime,pbase,pfill = get_puff_pressures(tree)
    if not options.short_title:
        ip_ax.plot(times*1e3, ip/1e3, label='%d, $p_{F}$ = %d$\mu$T' % (shotno,pfill/1000.), color=color)
    else:
        ip_ax.plot(times*1e3, ip/1e3, label='%d' % (shotno), color=color)
        #ip_ax.plot(times*1e3, ip/1e3, label='%d' % shotno, color=color)

    (times, oh) = read_data(tree, '.sensors:oh_current', t_start=0,
                            t_end=options.end_time)
    oh_ax.plot(times*1e3, oh*1e-3, color=color)

    #(times, vf) = read_data(tree, '.sensors:vf_current', t_start=0,
    #                        t_end=options.end_time)
    #vf_ax.plot(times*1e3, vf*1e-3, color=color)

    (times, r) = calc_r_major(tree, t_start=-.001, t_end=options.end_time, byrne = options.byrne)
    rmask = ~np.isnan(r)
    if ip.max() < 10:
        rmask = [0]
    r_ax.plot(times[rmask]*1e3, r[rmask]*100, color=color)

    q = calc_q(tree, times, r_major=r, byrne = options.byrne)
    qmask = ~np.isnan(q)
    if ip.max() < 10:
        qmask = [0]
    q_ax.plot(times[qmask]*1e3, q[qmask], color=color)

    if options.rot_probes:
        try:
            (times, biasv) = read_data(tree, '.sensors.bias_probe:voltage', t_start=0, t_end=options.end_time)
            biasv_ax.plot(times*1e3, biasv, color=color)

            (times, biasc) = read_data(tree, '.sensors.bias_probe:current', t_start=0, t_end=options.end_time)
            biasc_ax.plot(times*1e3, biasc, color=color)
        except:
            pass
        try:
            (times, tipA_c) = read_data(tree, '.sensors.mach_probe:tipa_c', t_start=0, t_end=options.end_time)
            (times, tipB_c) = read_data(tree, '.sensors.mach_probe:tipb_c', t_start=0, t_end=options.end_time)
            (times, tipC_c) = read_data(tree, '.sensors.mach_probe:tipc_c', t_start=0, t_end=options.end_time)
            (times, tipD_c) = read_data(tree, '.sensors.mach_probe:tipd_c', t_start=0, t_end=options.end_time)
            (times, tipE_c) = read_data(tree, '.sensors.mach_probe:tipe_c', t_start=0, t_end=options.end_time)
            (times, tipE_v) = read_data(tree, '.sensors.mach_probe:tipe_v', t_start=0, t_end=options.end_time)

            machmask = times < oh_bias_trig
            tipA_c -= tipA_c[machmask].mean()
            tipB_c -= tipB_c[machmask].mean()
            tipC_c -= tipC_c[machmask].mean()
            tipD_c -= tipD_c[machmask].mean()
            tipE_c -= tipE_c[machmask].mean()
            tipE_v -= tipE_v[machmask].mean()

            # Effective probe tip area calibration
            tipB_c*=0.629
            tipC_c*=0.964
            tipD_c*=1.324

            tipA_filt = BandFilt.LowPassProcess(tipA_c, times, order=1, freq=5.0e3)
            tipB_filt = BandFilt.LowPassProcess(tipB_c, times, order=1, freq=5.0e3)
            tipC_filt = BandFilt.LowPassProcess(tipC_c, times, order=1, freq=5.0e3)
            tipD_filt = BandFilt.LowPassProcess(tipD_c, times, order=1, freq=5.0e3)
            tipE_filt = BandFilt.LowPassProcess(tipE_c, times, order=1, freq=5.0e3)
            tipEv_filt = BandFilt.LowPassProcess(tipE_v, times, order=1, freq=5.0e3)

            machT = .5*np.log((tipB_filt + tipC_filt) / (tipA_filt + tipD_filt))
            machT_mask = np.isfinite(machT)
            machP = .5*np.log((tipC_filt + tipD_filt) / (tipA_filt + tipB_filt))
            machP_mask = np.isfinite(machP)
            isat = tipE_filt
            machT_ax.plot(times[machT_mask]*1e3, machT[machT_mask], color=color)
            machP_ax.plot(times[machP_mask]*1e3, machP[machP_mask], color=color)
            isat_ax.plot(times*1e3, isat*1e3, color=color)
            tipEv_ax.plot(times*1e3, tipE_v, color=color)
        except:
            if options.debug:
                raise
            pass


#    (times, m3_sin) = read_data(tree, '.sensors.rogowskis:sin_3a', t_start=0, 
#                           t_end=options.end_time)
#    m3_cos = read_data(tree, '.sensors.rogowskis:cos_3a', times=times) 
#    # m3_cos is zeroed out because the channel is being used by the sxr array
#    m3 = smoothe(np.sqrt((m3_sin)**2 + (m3_cos*0)**2), int(options.smoothe // (times[1]-times[0])))
#    m3_ax.plot(times*1e3, m3, color=color)
    
    #tf = read_data(tree, '.sensors:tf_probe', times=times)
    #tf *= tree.getNode('.sensors:tf_probe:mr').data() / 0.92
    #tf_ax.plot(times*1e3, tf, color=color)

    (times, lv) = read_data(tree, '.devices.west_rack:cpci:input_69', t_start=0,
                               t_end=options.end_time)
    lvmask = times < oh_bias_trig
    lv = -lv*101.0
    lv -= lv[lvmask].mean()
    lv_ax.plot(times*1e3, lv, color=color)

    if shotno >= 70000:
        (times, spec) = read_data(tree, '.devices.screen_room:a14:input_1', t_start=0,
                                    t_end=options.end_time)
        spmask = times < oh_bias_trig
        spec -= spec[spmask].mean()

        sp_ax.plot(times*1e3, spec, color=color)
        spec_max = np.amax(spec)
        if last_shot:
            sp_ax.set_ylim(0,1.2*spec_max)

    (times, sxr) = read_data(tree, '.devices.north_rack:cpci:input_74', t_start=0,
                            t_end=options.end_time)
    sxrmask = times < oh_bias_trig
    sxr -= sxr[sxrmask].mean()
    sxr = -sxr
    sxr_ax.plot(times*1e3, sxr, color=color)

    if amp_ax is not None:
        # Mode amplitude and frequency using feedback sensors 4
        fbMode_list = []
        fbMag = []
        fbPhase = []
        fbFreq = []
        for i in np.arange(4):
            fb_sensors = argparse_sensors_type('FB*{0}p'.format(i+1))
            fb_sensors = SSet.filterBadSensors(fb_sensors)
         
            (times, fbsigs) = signals_from_shot(tree, [ '.sensors.magnetic:%s' % x for x in fb_sensors],
                        t_start=0, t_end=(options.end_time+0.5))

            SSet.setNameList(fb_sensors)
            theta=SSet.loc_Theta
            phi=SSet.loc_Phi
            A=np.zeros([len(fb_sensors),2])
            A[:,0]=np.cos(2*theta-phi)
            A[:,1]=np.sin(2*theta-phi)
            B=np.linalg.pinv(A)
            fbMode=np.dot(B,fbsigs)

            fbmask = (times > 0)&(times < (options.end_time))
            avgmask = (times > 1e-3)&(times < 5e-3)


            fbMode = fbMode[:,fbmask]
            times = times[fbmask]

            fbMode=BandFilt.HighPassProcess(fbMode, times, 2, 0.3e3)
            fbMode_list.append(fbMode)
            fbMag.append(np.sqrt(fbMode[0]**2 + fbMode[1]**2))
            fbPhase.append(np.arctan2(fbMode[1], fbMode[0]))
            fbFreq.append(np.convolve(np.unwrap(fbPhase[i]), slope_fit(50), 'same')/(2*np.pi*2e-3))    #np.convolve(np.unwrap(fbPhase), slope_fit(70), 'same')/(2*np.pi*2e-3)

        amp_ax.plot(times*1e3, np.average(fbMag, axis=0)*1e3, color=color)
        freq_ax.plot(times*1e3, np.average(fbFreq, axis=0), color=color)

    if options.thomson:
        Te = []
        Ne = []
        locations = []
        for polyNum in range(1,11):
            try:
                TeNode = tree.getNode('sensors.thomson.poly_{0:02d}:Te'.format(polyNum))
                NeNode = tree.getNode('sensors.thomson.poly_{0:02d}:Ne'.format(polyNum))
                locNode = tree.getNode('sensors.thomson.poly_{0:02d}:location'.format(polyNum))
                Te.append(TeNode.data())
                Ne.append(NeNode.data())
                locations.append(locNode.data())
            except:
                continue
        Te_ax.scatter(locations, Te, color=color)
        Ne_ax.scatter(locations, Ne, color=color)
        laserTime = tree.getNode('.timing.thomson:fire').data()/1000
        r_ax.axvline(laserTime, ls='--', color=color, alpha=0.8)

    if options.plot_n1_sensors:
        ta_sensors = argparse_sensors_type('TA*p')
        ta_sensors = SSet.filterBadSensors(ta_sensors)
        
        (times, tasigs) = signals_from_shot(tree, [ '.sensors.magnetic:%s' % x for x in ta_sensors],
                    t_start=0, t_end=options.end_time)

        n1_subtr_sigs = np.zeros(tasigs[:15,:].shape)
        for i in np.arange(5):
            for j in np.arange(3):
                if 'TA{0:02}_S{1}P'.format(i+1,j+1) in ta_sensors and 'TA{0:02}_S{1}P'.format(i+6,j+1) in ta_sensors:
                    n1_subtr_sigs[3*i+j,:] = tasigs[3*i+j,:] - tasigs[3*(i+5)+j,:]

        plot_signals(times, n1_subtr_sigs, TA_n1ax_arr, color=color)
        
        fb_sensors = argparse_sensors_type('FB*p')
        (times, fbsigs) = signals_from_shot(tree, [ '.sensors.magnetic:%s' % x for x in fb_sensors],
                    t_start=0, t_end=options.end_time)
        fb_sensors = SSet.filterBadSensors(fb_sensors)

        n1_subtr_sigs = np.zeros(fbsigs[:20,:].shape)
        for i in np.arange(5):
            for j in np.arange(4):
                if 'FB{0:02}_S{1}P'.format(i+1,j+1) in fb_sensors and 'FB{0:02}_S{1}P'.format(i+6,j+1) in fb_sensors:
                    n1_subtr_sigs[4*i+j,:] = fbsigs[4*i+j,:] - fbsigs[4*(i+5)+j,:]

        plot_signals(times, n1_subtr_sigs, FB_n1ax_arr, color=color)


    if not last_shot:
        return

    # SXR array
    for i in np.arange(15):
        (times, sxr_ch) = read_data(tree, '.devices.west_rack:cpci:input_%02d' %(i+74), t_start=0,
                            t_end=options.end_time)

        if i == 0:
            sxr_array = np.empty((15, len(times)))

        sxr_array[i,:]=sxr_ch

    if shotno < 76879:
        sxr_array[9,:] *=0
        sxr_array[13,:] *=0
        sxr_array[14,:] *=0
    elif shotno > 79898:
        sxr_array[0,:] /= 8
        sxtemp=np.copy(sxr_array[9,:])
        sxr_array[9,:]=np.copy(sxr_array[14,:])
        sxr_array[14,:]=np.copy(sxtemp)
    else:
        sxtemp=np.copy(sxr_array[9,:])
        sxr_array[9,:]=np.copy(sxr_array[14,:])
        sxr_array[14,:]=np.copy(sxtemp)

    avemask = times > 1.5e-3
    sig_max = np.abs(sxr_array[:,avemask]).max()
    sig_min = 0.03*sig_max
    if options.white:
        sxfan_ax.contourf(times*1e3,np.arange(15)+1,sxr_array,128,cmap=sxr_rainbow_colormap_w(128),
            norm=Normalize(vmin=sig_min, vmax=sig_max))
    else:
        sxfan_ax.contourf(times*1e3,np.arange(15)+1,sxr_array,128,cmap=sxr_rainbow_colormap(128),
            norm=Normalize(vmin=sig_min, vmax=sig_max))


    fit_extend=0.7
    fitmask = (times > (options.zoom_start - fit_extend)/1000) & (times < (options.zoom_end + fit_extend)/1000)
    # Settings for plotsensors
    plotsensors_color = 'k' if options.white else 'w'
    vline_color = '#888888'



    # Toroidal Array ----------------------------------------------------------------------
    ta_sensors = argparse_sensors_type('TA*p')
    ta_sensors = SSet.filterBadSensors(ta_sensors)
    
    (times, tasigs) = signals_from_shot(tree, [ '.sensors.magnetic:%s' % x for x in ta_sensors], 
            t_start=0, t_end=options.end_time)

    (plot_times, tap) = subtract_background(times[fitmask], tasigs[:,fitmask])

    sig_range = np.abs(tap).max()
    SSet.setNameList(ta_sensors)
    plot_phi=SSet.loc_Phi
    plot_phi[plot_phi<0]+=2*np.pi
    plot_phi=np.roll(180/np.pi*plot_phi,-11)
    plot_tap=np.roll(tap,-11,axis=0)
    tap_ax.contourf(plot_times*1e3,plot_phi,plot_tap,25,cmap=red_green_colormap(),
        norm=Normalize(vmin=-sig_range, vmax=sig_range))

    if options.plotsensors:
        plot_sensors(times, signals = tasigs[[i for i in np.arange(len(ta_sensors)) if 'S2P' not in ta_sensors[i]]], 
                ax_arr = TAax_arr, avgSigs = np.average(tasigs, axis=0), fitmask=fitmask, vlines=[options.zoom_start, options.zoom_end])
    
    # Poloidal Array 1  -----------------------------------------------------------------------
    pa1_sensors = argparse_sensors_type('PA1*p')
    pa1_sensors = SSet.filterBadSensors(pa1_sensors)
    
    (times, pa1sigs) = signals_from_shot(tree, [ '.sensors.magnetic:%s' % x for x in pa1_sensors],
                t_start=0, t_end=options.end_time)

    (plot_times, pa1p) = subtract_background(times[fitmask], pa1sigs[:,fitmask])

    sig_range = np.abs(pa1p).max()
    SSet.setNameList(pa1_sensors)
    plot_theta=180/np.pi*SSet.loc_Theta
    pa1_ax.contourf(plot_times*1e3,plot_theta,pa1p,25,cmap=red_green_colormap(),
        norm=Normalize(vmin=-sig_range, vmax=sig_range))

    if options.plotsensors:
        plot_sensors(times, signals = pa1sigs[0::2], 
                ax_arr = PA1ax_arr, avgSigs = np.average(pa1sigs, axis=0), fitmask=fitmask, vlines=[options.zoom_start, options.zoom_end])

    # Poloidal Array 2  -----------------------------------------------------------------------
    pa2_sensors = argparse_sensors_type('PA2*p')
    pa2_sensors = SSet.filterBadSensors(pa2_sensors)
    
    (times, pa2sigs) = signals_from_shot(tree, [ '.sensors.magnetic:%s' % x for x in pa2_sensors],
                t_start=0, t_end=options.end_time)


    (plot_times, pa2p) = subtract_background(times[fitmask], pa2sigs[:,fitmask])

    sig_range = np.abs(pa2p).max()
    SSet.setNameList(pa2_sensors)
    plot_theta=180/np.pi*SSet.loc_Theta
    pa2_ax.contourf(plot_times*1e3,plot_theta,pa2p,25,cmap=red_green_colormap(),
        norm=Normalize(vmin=-sig_range, vmax=sig_range))

    if options.plotsensors:
        plot_sensors(times, signals = pa2sigs[0::2], 
                ax_arr = PA2ax_arr, avgSigs = np.average(pa2sigs, axis=0), fitmask=fitmask, vlines=[options.zoom_start, options.zoom_end])

    # Feedback Sensors Poloidal 4 -----------------------------------------------------------------------
    fb4_sensors = argparse_sensors_type('FB*4p')
    fb4_sensors = SSet.filterBadSensors(fb4_sensors)
 
    (times, fb4sigs) = signals_from_shot(tree, [ '.sensors.magnetic:%s' % x for x in fb4_sensors],
                t_start=0, t_end=options.end_time)

    (plot_times, fb4) = subtract_background(times[fitmask], fb4sigs[:,fitmask])

    sig_range = np.abs(fb4).max()
    SSet.setNameList(fb4_sensors)
    plot_phi=SSet.loc_Phi
    plot_phi[plot_phi<0]+=2*np.pi
    plot_phi=180/np.pi*np.roll(plot_phi,-4)
    plot_fb4=np.roll(fb4,-4,axis=0)
    fb4_ax.contourf(plot_times*1e3,plot_phi,plot_fb4,25,cmap=red_green_colormap(),
        norm=Normalize(vmin=-sig_range, vmax=sig_range))
 
    if options.plotsensors:
        for j in np.arange(4):
            plot_fb_sensors = argparse_sensors_type('FB*{0}p'.format(j+1))
            plot_fb_sensors = SSet.filterBadSensors(plot_fb_sensors)
            (times, plot_fbsigs) = signals_from_shot(tree, [ '.sensors.magnetic:%s' % x for x in plot_fb_sensors], 
                    t_start=0, t_end=options.end_time)
            
            plot_sensors(times, signals = plot_fbsigs[0::2,:], ax_arr = FBax_arr[5*j:5*j+5], 
                    avgSigs = np.average(plot_fbsigs, axis=0), fitmask=fitmask, vlines=[options.zoom_start, options.zoom_end])


def main(SSet, fig, figTA, figPA1, figPA2, figFB, figTA_n1, figFB_n1):
    bw_color = 'k' if options.white else 'w'
    #tf_ax = fig.add_subplot(1,3,1)
    #tf_ax.set_ylabel('TF at 0.92 (T)')
    #tf_ax.yaxis.set_major_locator(MaxNLocator(4))
    #for label in tf_ax.get_xticklabels():
    #    label.set_visible(False)
 
    #Setup grid of plots using matplotlib.GridSpec
    cols = [None] * 3
    cols_nrows = [None] * 3 # number of rows in each column
    cols_nrows[0] = 5
    if options.short_title:
        #fig.suptitle('Shot '+str(options.shotnos[-1]), fontsize=20)
        grid_top = 0.91
        cols_nrows[0]-=1
        col1_shift = 1
    else:
        grid_top = 0.97
        col1_shift=0

    
    if options.thomson:
        cols_nrows[0]+=2
        col1_row_ratios=[]
        for i in xrange(col1_num_rows+1):
            col1_row_ratios.append(4)
        col1_row_ratios.append(1)
        col1_row_ratios.append(3)
        cols[0] = gridspec.GridSpec(cols_nrows[0], 6, height_ratios=col1_row_ratios)
        Te_ax = fig.add_subplot(cols[0][-1,:3])
        Ne_ax = fig.add_subplot(cols[0][-1,3:])
    else:
        cols[0] = gridspec.GridSpec(cols_nrows[0], 6)

    cols[1] = gridspec.GridSpec(5, 6)
    cols[2] = gridspec.GridSpec(5, 6)


    if not options.short_title:
        shot_info_ax = fig.add_subplot(cols[0][0,:3])
    else:
        shot_info_ax = None

    tap_ax = fig.add_subplot(cols[0][1-col1_shift,:])

    pa1_ax = fig.add_subplot(cols[0][2-col1_shift,:])

    pa2_ax = fig.add_subplot(cols[0][3-col1_shift,:])

    fb4_ax = fig.add_subplot(cols[0][4-col1_shift,:])

    ip_ax = fig.add_subplot(cols[1][0,:])

    r_ax = fig.add_subplot(cols[1][1,:], sharex=ip_ax)
    
    q_ax = fig.add_subplot(cols[1][2,:], sharex=ip_ax)

    amp_ax = fig.add_subplot(cols[1][3,:], sharex=ip_ax)

    freq_ax = fig.add_subplot(cols[1][4,:], sharex=ip_ax)

    sp_ax = fig.add_subplot(cols[2][0,:], sharex=ip_ax)

    sxr_ax = fig.add_subplot(cols[2][1,:], sharex=ip_ax)

    sxfan_ax = fig.add_subplot(cols[2][2,:], sharex=ip_ax)

    lv_ax = fig.add_subplot(cols[2][3,:], sharex=ip_ax)

    oh_ax = fig.add_subplot(cols[2][4,:], sharex=ip_ax)

    m3_ax = None


    if options.rot_probes:
        cols.append(gridspec.GridSpec(6, 6))
        cols[0].update(left=0.06, right=0.26, top=grid_top, bottom=0.06, wspace=0.05, hspace=0.02)
        cols[1].update(left=0.31, right=0.50, top=grid_top, bottom=0.06, wspace=0.05, hspace=0.02)
        cols[2].update(left=0.55, right=0.74, top=grid_top, bottom=0.06, wspace=0.05, hspace=0.02)
        cols[3].update(left=0.79, right=0.98, top=grid_top, bottom=0.06, wspace=0.05, hspace=0.02)

        biasv_ax = fig.add_subplot(cols[3][0,:], sharex=ip_ax)
        biasc_ax = fig.add_subplot(cols[3][1,:], sharex=ip_ax)
        machT_ax = fig.add_subplot(cols[3][2,:], sharex=ip_ax)
        machP_ax = fig.add_subplot(cols[3][3,:], sharex=ip_ax)
        isat_ax = fig.add_subplot(cols[3][4,:], sharex=ip_ax)
        tipEv_ax = fig.add_subplot(cols[3][5,:], sharex=ip_ax)
    else:
        cols[0].update(left=0.06, right=0.32, top=grid_top, bottom=0.06, wspace=0.05, hspace=0.02)
        cols[1].update(left=0.38, right=0.65, top=grid_top, bottom=0.06, wspace=0.05, hspace=0.02)
        cols[2].update(left=0.71, right=0.98, top=grid_top, bottom=0.06, wspace=0.05, hspace=0.02)
        biasv_ax = None
        biasc_ax = None
        machT_ax = None
        machP_ax = None
        isat_ax = None
        tipEv_ax = None

    current_shot = MDSplus.Tree.getCurrent('hbtep2')

    if not options.short_title:
        shot_info_ax.set_frame_on(False)
        shot_info_ax.axes.get_yaxis().set_visible(False)
        shot_info_ax.axes.get_xaxis().set_visible(False)
        current_shot_name = shot_info_ax.text(0.3, 0.75, current_shot, fontsize=70, color='b', va="top", ha="center")
        if options.white:
            shot_info_ax.text(0.3, 0.8, "Last shot taken:", fontsize=24, color='k', va="bottom", ha="center")
            current_shot_name.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="k")])
        else:
            shot_info_ax.text(0.3, 0.8, "Last shot taken:", fontsize=24, color='w', va="bottom", ha="center")
            current_shot_name.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="w")])

    setup_axes(axes=ip_ax, ylim=[0,20], ylabel='Plasma Current (kA)',xlim=[0,options.end_time*1000],  num_major_yticks=4, 
            vlines=[options.zoom_start, options.zoom_end])

    setup_axes(axes=r_ax, ylim=[88,98], ylabel='Major Radius (cm)', num_major_yticks=4,
            vlines=[options.zoom_start, options.zoom_end],
            hlines=[90,92])

    setup_axes(axes=q_ax, ylim=[2,5.1], ylabel='Safety Factor',  num_major_yticks=4,
            vlines=[options.zoom_start, options.zoom_end],
            hlines=[3,])

    if amp_ax: setup_axes(axes=amp_ax, ylim=[0,3], ylabel='Mode Amplitude', num_major_yticks=4,
            vlines=[options.zoom_start, options.zoom_end],
            hlines=[0,])

    if freq_ax: setup_axes(axes=freq_ax, ylim=[-2,10], ylabel='Mode Frequency (kHz)', xlabel='Time (ms)', xtick_labels=True, num_major_yticks=4,
            vlines=[options.zoom_start, options.zoom_end],
            hlines=[0,])

    if biasv_ax: setup_axes(axes=biasv_ax, ylim=[-125,125], ylabel='Bias Probe Voltage', num_major_yticks=5,
            vlines=[options.zoom_start, options.zoom_end],
            hlines=[0,])

    if biasc_ax: setup_axes(axes=biasc_ax, ylim=[-20,90], ylabel='Bias Probe Current', xlabel='Time (ms)', xtick_labels=True, num_major_yticks=5,
            vlines=[options.zoom_start, options.zoom_end],
            hlines=[0,])

    if machT_ax: setup_axes(axes=machT_ax, ylim=[-0.5,0.5], ylabel='Mach Tor.', num_major_yticks=5,
            vlines=[options.zoom_start, options.zoom_end],
            hlines=[0,])

    if machP_ax: setup_axes(axes=machP_ax, ylim=[-0.5,0.5], ylabel='Mach Pol.', num_major_yticks=5,
            vlines=[options.zoom_start, options.zoom_end],
            hlines=[0,])

    if isat_ax: setup_axes(axes=isat_ax, ylim=[-10,150], ylabel='Isat current', num_major_yticks=5,
            vlines=[options.zoom_start, options.zoom_end],
            hlines=[0,])

    if machP_ax: setup_axes(axes=tipEv_ax, ylim=[-60,40], ylabel='tipE V', num_major_yticks=5,
            vlines=[options.zoom_start, options.zoom_end],
            hlines=[0,])

    setup_axes(axes=lv_ax, ylim=[0,15], ylabel='Loop Voltage', num_major_yticks=4,
            vlines=[options.zoom_start, options.zoom_end])
 
    setup_axes(axes=sp_ax, ylim=[0,0.20], ylabel='Spectrometer', num_major_yticks=4,
            vlines=[options.zoom_start, options.zoom_end])

    setup_axes(axes=sxr_ax, ylim=[0,3.5], ylabel='Soft X-ray\nMidplane', num_major_yticks=4,
            vlines=[options.zoom_start, options.zoom_end])

   
    if m3_ax: setup_axes(axes=m3_ax, ylim=[0,5], ylabel='m=3 Rogowski', num_major_yticks=4,
            vlines=[options.zoom_start, options.zoom_end])
    
    #if vf_ax: setup_axes(axes=vf_ax, ylim=None, ylabel='VF Current (kA)', num_major_yticks=4)

    setup_axes(axes=oh_ax, ylim=None, ylabel='OH Current (kA)', xlabel='Time (ms)', xtick_labels=True, num_major_yticks=4,
            vlines=[options.zoom_start, options.zoom_end])

    setup_axes(axes=sxfan_ax, ylim=None, ylabel='SXR Fan Array', num_major_yticks=None)

    tap_ax.set_ylabel('TAp (degrees)')
    fluct_shot = options.shotnos[-1]
    if fluct_shot <= 0:
        fluct_shot += current_shot

    if options.short_title:
        pass
        #tap_ax.set_title('Poloidal Sensor Fluctuations for Shot '+str(fluct_shot),fontsize=20)
    else:
        #tap_ax.set_title('Poloidal Sensor Fluctuations for Shot '+str(fluct_shot),fontsize=20)
        tap_ax.yaxis.set_major_locator(MaxNLocator(4))

    for label in tap_ax.get_xticklabels():
        label.set_visible(False)

    pa1_ax.set_ylabel('PA1p (degrees)')
    pa1_ax.yaxis.set_major_locator(MaxNLocator(4))
    for label in pa1_ax.get_xticklabels():
        label.set_visible(False)

    pa2_ax.set_ylabel('PA2p (degrees)')
    pa2_ax.yaxis.set_major_locator(MaxNLocator(4))
    for label in pa2_ax.get_xticklabels():
        label.set_visible(False)

    fb4_ax.set_ylabel('FB4p (degrees)')
    fb4_ax.set_xlabel('Time (ms)')
    fb4_ax.yaxis.set_major_locator(MaxNLocator(4))
    fb4_ax.xaxis.set_major_locator(MaxNLocator(4))

    if options.thomson:
        #locMajorLocator = MultipleLocator(4)
        TeMajorLocator = MultipleLocator(50)
        TeMinorLocator = MultipleLocator(25)
        NeMajorLocator = MultipleLocator(2)
        Te_ax.set_ylabel('Te (eV)')
        Te_ax.set_xlabel('Radius (cm)')
        Te_ax.set_ylim(0,200)
        Te_ax.set_xlim(78,106)
        #Te_ax.xaxis.set_major_locator(locMajorLocator)
        Te_ax.yaxis.set_major_locator(TeMajorLocator)
        Te_ax.yaxis.set_minor_locator(TeMinorLocator)
        #Te_ax.set_yticks(np.arange(0,175,25))
        Te_ax.yaxis.grid(b=True, which='minor')
        #Te_ax.xaxis.grid(b=True)
        Ne_ax.set_ylabel('Ne (arb)')
        Ne_ax.set_xlabel('Radius (cm)')
        Ne_ax.set_ylim(0,10)
        Ne_ax.set_xlim(78,106)
        Ne_ax.yaxis.tick_right()
        Ne_ax.yaxis.set_label_position("right")
        #Ne_ax.xaxis.set_major_locator(locMajorLocator)
        Ne_ax.yaxis.set_major_locator(NeMajorLocator)
        #Ne_ax.set_yticks(np.arange(0,12,2))
        Ne_ax.yaxis.grid(b=True)
    
    TAax_arr=None
    PA1ax_arr=None
    PA2ax_arr=None
    FBax_arr=None
    if options.plotsensors:
        TAax_arr = setup_ax_arr(figTA, numV=5, numH=4, ylim=[-0.01,0.03])
        PA1ax_arr = setup_ax_arr(figPA1, numV=4, numH=4, ylim=[-0.01,0.03])
        PA2ax_arr = setup_ax_arr(figPA2, numV=4, numH=4, ylim=[-0.01,0.03])
        FBax_arr = setup_ax_arr(figFB, numV=4, numH=5, ylim=[-0.01,0.03])

    TA_n1ax_arr=None
    FB_n1ax_arr=None
    if options.plot_n1_sensors:
        TA_n1ax_arr = setup_ax_arr(figTA_n1, numV=4, numH=4, ylim=[-0.01,0.01])
        FB_n1ax_arr = setup_ax_arr(figFB_n1, numV=4, numH=5, ylim=[-0.01,0.01])

    for (i, shotno) in enumerate(options.shotnos):
        if shotno <= 0:
            shotno += current_shot

        last_shot=1 if i == len(options.shotnos)-1 else 0

        log.info('Processing %d...', shotno)
        try:
            idx = len(options.shotnos)-2-i
            color = bw_color if last_shot else get_color(idx)
            #tree = MDSplus.Tree('hbtep2', shotno)
            #plot_tree_data(oh_ax, tree, '.sensors:oh_current', color, coeff=1e-3)
            plot_shot(shotno, get_color(idx), SSet, ip_ax, r_ax, q_ax, oh_ax, amp_ax, freq_ax, lv_ax, sp_ax, sxr_ax, last_shot,
                sxfan_ax if last_shot else None,
                tap_ax if last_shot else None, 
                pa1_ax if last_shot else None, 
                pa2_ax if last_shot else None, 
                fb4_ax if last_shot else None,
                Te_ax if options.thomson else None,
                Ne_ax if options.thomson else None,
                biasv_ax if options.rot_probes else None,
                biasc_ax if options.rot_probes else None,
                machT_ax if options.rot_probes else None,
                machP_ax if options.rot_probes else None,
                isat_ax if options.rot_probes else None,
                tipEv_ax if options.rot_probes else None,
                TAax_arr, PA1ax_arr, PA2ax_arr, FBax_arr, TA_n1ax_arr, FB_n1ax_arr)
        except BadShot as exc:
            log.debug(str(exc))
            continue
        except BadSensor as exc:
            log.warn("Shot %d seems to have bad sensors, not calculating perturbation amplitude.", shotno)
            log.warn('Worst sensor: %s', options.eq_sensors[exc.index])
            continue
        except Exception as exc:
            log.exception('Error processing shot %d, skipping', shotno)
            continue
   
    handles, labels = ip_ax.get_legend_handles_labels()
    if options.small:
        #legfont = FontProperties(size='medium')
        ip_ax.legend(handles[::-1], labels[::-1], 'lower right')#, prop=legfont)
    else:
        #legfont = FontProperties(size='x-small')
        ip_ax.legend(handles[::-1], labels[::-1], bbox_to_anchor=(-.20,1.12))#, prop=legfont)
#    ip_ax.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1.01,0.67), prop=legfont, ncol=1 )
#    else:
#        ip_ax.legend(handles[::-1], labels[::-1], 'lower right')
#    ip_ax.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1.01,0.67), prop=legfont, ncol=1 )

if __name__ == '__main__':
    options = parse_args()
    init_logging(options.quiet, options.debug)    

    if not options.debug:
        np.seterr(all='ignore')
    # Put end_time in seconds
    options.end_time /= 1000
    if options.printable:
        options.small = True
        options.white = True

    if options.plotsensors or options.plot_n1_sensors:
        options.small = True

    if options.small:
        options.short_title = True
        mpl.rcParams.update({'font.size': 20})
        mpl.rcParams.update({'legend.fontsize': 14})
    else:
        mpl.rcParams.update({'font.size': 20})
        mpl.rcParams.update({'legend.fontsize': 14})

    if options.white:
        mpl.rcParams.update({'figure.facecolor': 'w'})
        #mpl.rcParams['axes.color_cycle'] = get_color_cycle(len(options.shotnos),reverse=True)
    else:
        mpl.rcParams.update({'text.color': 'w'})
        mpl.rcParams.update({'axes.facecolor': 'k'})
        mpl.rcParams.update({'axes.edgecolor': 'w'})
        mpl.rcParams.update({'axes.labelcolor': 'w'})
        mpl.rcParams.update({'xtick.color': 'w'})
        mpl.rcParams.update({'ytick.color': 'w'})
        mpl.rcParams.update({'grid.color': 'w'})
        mpl.rcParams.update({'figure.facecolor': 'k'})
        mpl.rcParams.update({'figure.edgecolor': 'k'})
        #mpl.rcParams['axes.color_cycle'] = get_color_cycle(len(options.shotnos))

    SSet=surfaceAnalysis.SensorSet()    # Contains magnetic sensor information

    fig = plt.figure(figsize=(15,8))
    figTA=None
    figPA1=None
    figPA2=None
    figFB=None
    if options.plotsensors:
        figTA = plt.figure(figsize=(15,8))
        figTA.suptitle('TA sensors', fontsize=20)
        figTA.canvas.set_window_title('TA sensors')
        figPA1 = plt.figure(figsize=(15,8))
        figPA1.suptitle('PA1 sensors', fontsize=20)
        figPA1.canvas.set_window_title('PA1 sensors')
        figPA2 = plt.figure(figsize=(15,8))
        figPA2.suptitle('PA2 sensors', fontsize=20)
        figPA2.canvas.set_window_title('PA2 sensors')
        figFB = plt.figure(figsize=(15,8))
        figFB.suptitle('FB sensors', fontsize=20)
        figFB.canvas.set_window_title('FB sensors')

    figTA_n1=None
    figFB_n1=None
    if options.plot_n1_sensors:
        figTA_n1 = plt.figure(figsize=(15,8))
        figTA_n1.suptitle('n=1 signals of TA sensors', fontsize=20)
        figFB_n1 = plt.figure(figsize=(15,8))
        figFB_n1.suptitle('n=1 signals of FB sensors', fontsize=20)

    if options.loop:
        import socket
        screensaveroff = subprocess.call(['xset','s','off','-dpms'])
        hostname=socket.gethostname()
        if not hostname == 'spitzer':
            subprocess.call(['gnome-screensaver-command','--exit'])

        @atexit.register
        def screensaveron():
            if not hostname == 'spitzer':
                subprocess.call(['gnome-screensaver-command','-d'])
            subprocess.call(['xset','s','on','+dpms'])

        import gobject    
        def loop():
            while True:
                try:
                    log.debug("Waiting for MDSplus event...")
                    MDSplus.event.Event.wfevent('hbtep2_complete')
                    log.debug("Received MDSplus event for shot complete")
                except TypeError:
                    print("Type error in loop.")
                    pass
                except:
                    raise
                gobject.idle_add(fig.clear)
                gobject.idle_add(main, SSet, fig, None, None, None, None, None, None)
                gobject.idle_add(plt.draw)

        gobject.threads_init()
        t = threading.Thread(target=loop)
        t.daemon = True
        t.start()
    
    main(SSet, fig, figTA, figPA1, figPA2, figFB, figTA_n1, figFB_n1)
    if options.extra_fig:
        extra_fig(SSet)
    if not plt.isinteractive():
        if not options.small:
            mng = plt.get_current_fig_manager()
            mng.window.maximize()
        plt.show()
