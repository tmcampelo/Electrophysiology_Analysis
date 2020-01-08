# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 15:05:33 2019

@author: tiago
"""
# %% IMPORT MODULES NEEDED FOR THE JOB [shift + enter 4run]
import pyabf
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# %% IMPORT FUNCTIONS CREATED BY KEES, ASHLEY (see https://www.biorxiv.org/content/10.1101/652461v3)

#FUNCTION TO DETECT SPIKES FROM ABF RECORDINGS
# find sp_ind, sp_peak_ind, and sp_end_ind at the same time
# thresh is in V/s
# refrac is in ms - blocks detection of multiple spike initiations, peaks
# peak win is in ms - window after the spike inititation when there must be a peak    
def find_sp_ind_v2(Vm, Vm_ts, thresh, refrac, peak_win):
    end_win = 5  # ms after the spike peak to look for the end of the spike
    down_win = refrac  # ms after the spike peak to look for the max down slope
    # make the dVdt trace
    samp_period = np.round(np.mean(np.diff(Vm_ts)), decimals=6)
    dVdt = np.ediff1d(Vm, to_begin=0)
    dVdt = dVdt/(10*1000*samp_period)
    # detect when dVdt exceeds the threshold
    dVdt_thresh = np.array(dVdt > thresh, bool)
    if sum(dVdt_thresh) == 0:
        # there are no spikes
        sp_ind = np.empty(shape=0)
        sp_peak_ind = np.empty(shape=0)
        sp_end_ind = np.empty(shape=0)
    else:
        # keep just the first index per spike
        sp_ind = np.squeeze(np.where(np.diff(dVdt_thresh) == 1))
        # remove any duplicates of spikes that occur within refractory period
        samp_rate = 1/samp_period
        sp_ind = sp_ind[np.ediff1d(sp_ind,
                        to_begin=int(samp_rate*refrac/1000+1)) >
                        samp_rate*refrac/1000]
        # find the potential spike peaks (tallest within the refractory period)
        dist = refrac/(1000*samp_period)
        sp_peak_ind, _ = sp.signal.find_peaks(Vm, distance=dist, prominence=1)
        # find all the peaks, regardless of the refractory period
        sp_peak_ind_all, _ = sp.signal.find_peaks(Vm, prominence=1)
        # keep only sp_ind when there is a sp_peak_ind within the window
        max_lag = peak_win/(1000*samp_period)
        a = np.searchsorted(sp_peak_ind, sp_ind)
        lags = sp_peak_ind[a] - sp_ind
        sp_ind = sp_ind[lags < max_lag]
        sp_peak_ind = sp_peak_ind[np.searchsorted(sp_peak_ind, sp_ind)]
        # if there are any sp_ind that have the same sp_peak_ind, delete the second
        unique, counts = np.unique(sp_peak_ind, return_counts=True)
        repeat_peak_ind = unique[counts>1]
        for k in np.arange(repeat_peak_ind.size):
            temp_ind = np.where((sp_peak_ind == repeat_peak_ind[k]))[0]
            sp_peak_ind = np.delete(sp_peak_ind, temp_ind[1:])
            sp_ind = np.delete(sp_ind, temp_ind[1:])
        # find the end of spikes
        # first find zero crossings of dVdt where the slope of dVdt is positive
        # this is where Vm has maximum downward slope
        dVdt_min_ind = np.where(np.diff(np.signbit(dVdt))&~np.signbit(np.diff(dVdt)))[0]
        # set the windows over which to look for the max neg.slope and
        # near-zero slope after the peak
        win_ind = samp_rate*end_win/1000
        down_win_ind = samp_rate*down_win/1000
        # find potential ends of spikes and choose the best
        end_ind = np.full(sp_ind.size, np.nan)
        for i in np.arange(sp_ind.size):
            start = int(sp_peak_ind[i])  # start at each spike peak
            # if there is another peak before the next spike, reset the start
            j = np.searchsorted(sp_peak_ind_all, sp_peak_ind[i])
            if np.logical_and((i+1 < sp_peak_ind.size), (j+1 < sp_peak_ind_all.size)):
                b = sp_peak_ind_all[j+1] < sp_peak_ind[i+1]
                c = sp_peak_ind_all[j+1] - sp_peak_ind[i] < down_win_ind
                if np.logical_and(b, c):
                    start = int(sp_peak_ind_all[j+1])
            # set potential stop points to look for max downward and
            # near-zero slopes
            stop1 = int(sp_peak_ind[i]+down_win_ind)
            stop = int(sp_peak_ind[i]+win_ind)
            # reset the stop(s) if another spike starts before then
            if i != sp_ind.size-1:
                if stop > sp_ind[i+1]:
                    stop = sp_ind[i+1]
                if stop1 > sp_ind[i+1]:
                    stop1 = sp_ind[i+1]
            # find the minimum dVdt (max down slope Vm) between start and stop1
            min_ind = np.argmin(dVdt[start:stop1]) + start
            # find the next Vm zero-slope after the max down slope
            temp_ind = dVdt_min_ind[np.searchsorted(dVdt_min_ind, start)]+1
            if (temp_ind < stop) & (temp_ind > min_ind):
                # if the zero-slope occurs before stop, keep it
                end_ind[i] = temp_ind
            else:
                # if not, find when the Vm slope is closest to zero
                end_ind[i] = np.argmin(np.abs(dVdt[min_ind:stop]))+min_ind
        sp_end_ind = end_ind.astype('int64')
    return sp_ind, sp_peak_ind, sp_end_ind


# definition to calcuate max rise for each spike
def find_max_rise(Vm, Vm_ts, sp_ind, sp_peak_ind):
    samp_period = np.round(np.mean(np.diff(Vm_ts)), decimals=6)
    max_rise = np.full(sp_ind.size, np.nan)
    for i in np.arange(sp_ind.size):
        try:
            dVdt = np.diff(Vm[sp_ind[i]:sp_peak_ind[i]])/(1000*samp_period)
            max_rise[i] = np.nanmax(dVdt)
        except ValueError:
            max_rise[i] = np.nan
    return max_rise
   

# definition to calculate full-width at half-max for each spike
def find_fwhm(Vm, Vm_ts, sp_ind, sp_end_ind):
    samp_period = np.round(np.mean(np.diff(Vm_ts)), decimals=6)
    fwhm = np.full(sp_ind.size, np.nan)
    for i in np.arange(sp_ind.size):
        try:
            sp_Vm = Vm[sp_ind[i]:sp_end_ind[i]]
            half_max = (np.nanmax(sp_Vm) - sp_Vm[0])/2 + sp_Vm[0]
            inds = np.where(sp_Vm > half_max)[0]
            fwhm[i] = (inds[-1] - inds[0])*samp_period
        except ValueError:
            fwhm[i] = np.nan
    return fwhm

# FUNCTION TO REMOVE THE SPIKES
# definition for removing spikes, but keeping subthreshold components of the
# complex spike
def remove_spikes_v2(Vm, sp_ind, sp_end_ind):
    Vm_nosp = np.copy(Vm)
    # linearly interpolate between start and end of each spike
    for i in np.arange(sp_ind.size):
        start = sp_ind[i]
        stop = sp_end_ind[i]
        start_Vm = Vm[start]
        stop_Vm = Vm[stop]
        Vm_nosp[start:stop] = np.interp(np.arange(start, stop, 1),
                                        [start, stop], [start_Vm, stop_Vm])
    return Vm_nosp    

# %% IMPORT ABF FILE TO BE ANALYZED
abf = pyabf.ABF ("TC1272_CellA_spontaneous.abf")
Vm = abf.sweepY #associate vm to the recorded Vm
Vm_ts = abf.sweepX #associate time to time of recording (sec.)

# %% SPIKE DETECTION
thresh = 5  # V/s (same as 0.25 in old version of spike detection)
refrac = 1.5  # refractory period, in ms
peak_win = 3  # window in which there must be a peak, in ms
sp_ind, sp_peak_ind, sp_end_ind = find_sp_ind_v2(Vm, Vm_ts, thresh, refrac, peak_win)
                                                    
sp_init_Vm = np.empty(0) #the Vm at which the spike occurer --- has to be normalized to the resting Vm
sp_peak_Vm = np.empty(0) #Size of the spike --- depends on the cell opening
sp_fwhm = np.empty(0) #Calculates FWHM of recorded spikes

if sp_ind.size > 0:
        sp_init_Vm = Vm[sp_ind]
        sp_peak_Vm = Vm[sp_peak_ind]
        sp_fwhm = find_fwhm(Vm, Vm_ts, sp_ind, sp_end_ind)

sp_thresh = Vm.min() - sp_init_Vm
sp_num = len (sp_thresh)
sp_freq = len (sp_thresh) / Vm_ts.max()

plt.figure (1)
plt.title ("Spikes Detection")
plt.ylabel('Vm (mV)')
plt.xlabel(' time (sec.)')
plt.plot (Vm_ts, Vm)
plt.scatter(Vm_ts[sp_ind], Vm[sp_ind])
plt.scatter(Vm_ts[sp_peak_ind], Vm[sp_peak_ind])
plt.scatter(Vm_ts[sp_end_ind], Vm[sp_end_ind])
plt.savefig('spiking_spontaneous.png')

#del peak_win, thresh, refrac, sp_peak_ind, abf

# %% SPIKE ELIMINATION AND DOWNSAMPLING
if sp_ind.size > 0:
    Vm_nosp = remove_spikes_v2(Vm, sp_ind, sp_end_ind)
else:
    Vm_nosp = Vm

#Downsampling 20x of time (sec.)
Vm_res = sp.signal.decimate (Vm_nosp, 10, ftype='iir', axis=-1, zero_phase=True)
Vm_ts_res = sp.signal.decimate (Vm_ts, 10, ftype='iir', axis=-1, zero_phase=True)

# Plot to define a 30 sec. window 
plt.figure (2)
plt.title ("Choose a 30 sec frame to slice recording:")
plt.ylabel('Vm (mV)')
plt.xlabel(' time (sec.)')
plt.plot (Vm_ts_res, Vm_res)

#del Vm, Vm_nosp, sp_end_ind, sp_ind, Vm_ts

# %% FFT ANALYSIS
#Slicing Vm in a 10 sec. trace with stable baseline
start = 20  #value in sec
end = 30  # value in sec, make sure that end - start = 30

current_Vm = Vm_res [start*10000:end*10000]
current_Vm_ts = Vm_ts_res [start*10000:end*10000]
current_Vm_ts = current_Vm_ts - current_Vm_ts[0]

current_Vm_norm = (current_Vm - current_Vm.min())
cumulative_Vm_ts = sp.signal.decimate (current_Vm_ts, 40, ftype='iir', axis=-1, zero_phase=True)


cumulativeVm = np.cumsum (current_Vm_norm)/10000 #ERROR OF CODE ---- NEEDS TO BE UPGRATED
P_Upstate = (np.mean(np.amax(cumulativeVm)/np.average(cumulativeVm))/np.amax(current_Vm_ts))
cumulativeVm_graph = sp.signal.decimate (cumulativeVm, 40, ftype='iir', axis=-1, zero_phase=True)

#del Vm_res, Vm_ts_res, current_Vm, end, start, cumulativeVm

# %% PLOT ALL THE DATA
plt.close(2)

plt.figure (2)
plt.title ("Recording with Spike subtraction")
plt.ylabel('Vm (mV)')
plt.xlabel(' time (sec.)')
plt.plot (current_Vm_ts, current_Vm_norm)
plt.savefig('spontaneous_figure.png')

plt.figure (3)
plt.title ("Cumulative Vm Plot")
plt.ylabel('Cumulative Vm')
plt.xlabel('Time (sec.)')
plt.plot (cumulative_Vm_ts, cumulativeVm_graph)
plt.savefig('cumulative_figure.png')
