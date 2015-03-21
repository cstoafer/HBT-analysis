#!/usr/bin/env python
'''
Created on Mar 21, 2015

@author: cstoafer

Purpose is to show example of using shotData package for analyzing and visualizing data from HBT-EP
'''
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator, MultipleLocator
import matplotlib.gridspec as gridspec
import plotSettings
from shotData import shotData

def parse_args(args=None):
    '''Parse command line'''

    parser = ArgumentParser()

    parser.add_argument("shotnos", metavar='<shot number>', type=int, nargs='+',
                      help="Shots to plot.")

    parser.add_argument("--start-time", type=float, default=0, help="Start time for getting data.")

    parser.add_argument("--end-time", type=float, default=6, help="End time for getting data.")

    options = parser.parse_args(args)

    return options

def plotShotFig1(shot_data, color, tipA_cur_ax, tipB_cur_ax, tipC_cur_ax, tipD_cur_ax, tipE_cur_ax, machTOR_ax, machPOL_ax, mach_amp2_ax, mach_dir2_ax, tipE_v_ax):
    print('something')


def main():
    startTime=options.start_time*1e-3 if options.start_time > 0.2e-3 else options.start_time
    endTime=options.end_time*1e-3 if options.end_time > 0.2e-3 else options.end_time
    
    fig = plt.figure(figsize=(8,11))
    
    gs = gridspec.GridSpec(5,1)

    Ip_ax = fig.add_subplot(gs[0,0])
    MR_ax = fig.add_subplot(gs[1,0], sharex=Ip_ax)
    q_ax = fig.add_subplot(gs[2,0], sharex=Ip_ax)
    LV_ax = fig.add_subplot(gs[3,0], sharex=Ip_ax)
    signal_ax = fig.add_subplot(gs[4,0], sharex=Ip_ax)
    
    plotSettings.setup_axes(ax = Ip_ax, ylim=[0,20], ylabel = 'Ip (kA)', num_major_yticks = 4, ygrid=True)
    plotSettings.setup_axes(ax = MR_ax, ylim = [88,97], ylabel = 'MR (cm)', num_major_yticks = 4, ygrid=True)
    plotSettings.setup_axes(ax = q_ax, ylim=[2.01,4], ylabel='Edge q', num_major_yticks=4, ygrid=True)
    plotSettings.setup_axes(ax = LV_ax, ylim=[0,15], ylabel='Loop\nVoltage', num_major_yticks=4, ygrid=True)
    plotSettings.setup_axes(ax = signal_ax, ylabel='any signal', xtick_labels=True, xlabel='Time (ms)', ygrid=True)
    fig.tight_layout()

    fig.subplots_adjust(hspace=0.20, top=0.88)
    
    

    for (i,shotno) in enumerate(options.shotnos):
        shot_data = shotData(shotno, startTime, endTime)

        shot_data.Ip.plot(ax=Ip_ax)
        shot_data.MR.plot(ax=MR_ax, multiplier=1e2)
        shot_data.q.plot(ax=q_ax)
        shot_data.LV.plot(ax=LV_ax)
        shot_data.getSignal('.sensors.bias_probe:voltage').plot(ax=signal_ax)
        
        del shot_data


if __name__ == '__main__':
    plotSettings.customize_mpl(fontsize=18)
    options = parse_args()
    main()
    plt.show()
