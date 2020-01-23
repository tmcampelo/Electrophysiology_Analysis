# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 10:42:03 2020

@author: tiago
"""
# %% IMPORT MODULES NEEDED FOR THE JOB [shift + enter 4run]
import pyabf
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import label, generate_binary_structure
from scipy import stats
import os
import seaborn as sns
import pickle
import math

#%% SAVE/LOAD PROCESSED DATA AS VARIABLES IN CASE OF CODE CRASHING :D

f = open('data_spontaneous', 'rb')
data_FWE, data_SWE, data_ts_FWE, data_ts_SWE, Dist_FWE, Dist_SWE = pickle.load(f)
f.close()

"""with open('data_spontaneous', 'wb') as f:
    pickle.dump([data_FWE, data_SWE, data_ts_FWE, data_ts_SWE, Dist_FWE, Dist_SWE], f)
f.close ()"""


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
data_A = []
data_ts_A = []

data_B = []
data_ts_B = []

directory1 = r'Directory/ConditionA'
Vm_A = []
A_ts = []

for filename in os.listdir(directory1):
    if filename.endswith(".abf"):
        abf = pyabf.ABF (os.path.join(directory1, filename))
        Vm_data = abf.sweepY
        Vm_A.append (Vm_data)
        time = abf.sweepX
        A_ts.append (time)
        
directory2 = r'Directory/ConditionB'
Vm_B = []
B_ts = []

for filename in os.listdir(directory2):
    if filename.endswith(".abf"):
        abf = pyabf.ABF (os.path.join(directory2, filename))
        Vm_data = abf.sweepY 
        Vm_B.append (Vm_data)
        time = abf.sweepX
        B_ts.append (time)

# %% SPIKE DETECTION AND ELIMINATION // RECORDING DOWNSAMPLING
thresh = 5  # V/s (same as 0.25 in old version of spike detection)
refrac = 1.5  # refractory period, in ms
peak_win = 3  # window in which there must be a peak, in ms

Vm_nosp_A = []
Vm_ts_A = []

i = 0
a = len (Vm_A) - 1

while i <= a:
    try:
        data_X = A_ts[i]
        data_Y = Vm_A[i]    
        sp_ind, sp_peak_ind, sp_end_ind = find_sp_ind_v2(data_Y, data_X, thresh, refrac, peak_win)
        
        if sp_ind.size > 0:
            Vm_removed = remove_spikes_v2 (data_Y, sp_ind, sp_end_ind)
            Vm_removed2 = sp.signal.decimate (Vm_removed, 10, ftype='iir', axis=-1, zero_phase=True)
            A_ts_res = sp.signal.decimate (data_X, 10, ftype='iir', axis=-1, zero_phase=True)
            Vm_ts_A.append(A_ts_res)
            Vm_nosp_A.append(Vm_removed2)
            print('FWE_done')
            i += 1
            
        else:          
            Vm_removed = data_Y
            Vm_removed2 = sp.signal.decimate (Vm_removed, 10, ftype='iir', axis=-1, zero_phase=True)
            A_ts_res = sp.signal.decimate (data_X, 10, ftype='iir', axis=-1, zero_phase=True)
            Vm_ts_A.append(A_ts_res)
            Vm_nosp_A.append(Vm_removed2)
            print('FWE_done')
            i += 1
            
    except:
            print ('FWE_error')
            i += 1

Vm_nosp_B = []
Vm_ts_B = []

i = 0
a = len (Vm_B) - 1

while i <= a:
    try:
        data_X = B_ts[i]
        data_Y = Vm_B[i]    
        sp_ind, sp_peak_ind, sp_end_ind = find_sp_ind_v2(data_Y, data_X, thresh, refrac, peak_win)
        
        if sp_ind.size > 0:
            Vm_removed = remove_spikes_v2 (data_Y, sp_ind, sp_end_ind)
            Vm_removed2 = sp.signal.decimate (Vm_removed, 10, ftype='iir', axis=-1, zero_phase=True)
            B_ts_res = sp.signal.decimate (data_X, 10, ftype='iir', axis=-1, zero_phase=True)
            Vm_ts_B.append(B_ts_res)
            Vm_nosp_B.append(Vm_removed2)
            print('SWE_done')
            i += 1
            
        else:          
            Vm_removed = data_Y
            Vm_removed2 = sp.signal.decimate (Vm_removed, 10, ftype='iir', axis=-1, zero_phase=True)
            B_ts_res = sp.signal.decimate (data_X, 10, ftype='iir', axis=-1, zero_phase=True)
            Vm_ts_B.append(B_ts_res)
            Vm_nosp_B.append(Vm_removed2)
            print('SWE_done')
            i += 1
            
    except:
            print ('SWE_error')
            i += 1

# %% VISUAL CONTROL OF ALL RECORDED SPONTANEOUS ACTIVITY - Make sure that baseline is stable and there are no spikes
a = 0
b = a

plt.close(1)
plt.figure (1)
plt.title('FWE')
plt.plot (Vm_ts_A[a], Vm_nosp_A[a])

plt.close(2)
plt.figure (2)
plt.title('SWE')
plt.plot (Vm_ts_B[b], Vm_nosp_B[b])

# %% CROP Vm OF RECORDED CELL IN A 30 SEC. WINDOW AND NORMALIZE IT
A_start = 30
A_end = A_start + 30  #crop the full recording in a stable 30 sec. time window
current_Vm_A = Vm_nosp_A[a] [A_start*10000:A_end*10000]
current_Vm_A_norm = current_Vm_A - np.amin (current_Vm_A)

current_Vm_ts_A = Vm_ts_A[a] [A_start*10000:A_end*10000]
current_Vm_ts_A_norm = current_Vm_ts_A - current_Vm_ts_A [0]

data_A.append(current_Vm_A_norm)
data_ts_A.append (current_Vm_ts_A_norm)

B_start = 10
B_end = B_start + 30
current_Vm_B = Vm_nosp_B[a] [B_start*10000:B_end*10000]
current_Vm_B_norm = current_Vm_B - np.amin (current_Vm_B)

current_Vm_ts_B = Vm_ts_B[a] [B_start*10000:B_end*10000]
current_Vm_ts_B_norm = current_Vm_ts_B - current_Vm_ts_B [0]

data_B.append(current_Vm_B_norm)
data_ts_B.append (current_Vm_ts_B_norm)

plt.figure (1)
plt.plot (current_Vm_ts_A, current_Vm_A, color='r')

plt.figure (2)
plt.plot (current_Vm_ts_B, current_Vm_B, color='r')

# %% Vm distribution Analysis and PLOT
Dist_A =  np.array([])
Dist_B =  np.array([])

for i in range (len (data_A)):
    data = data_A[i]
    Dist_A = np.append (Dist_A, data)

for i in range (len (data_B)):
    data = data_B[i]
    Dist_B = np.append (Dist_B, data)

Dist_A = Dist_A [ (Dist_A >= 3)]
Dist_B = Dist_B [ (Dist_B >= 3)]

plt.figure(3)
plt.title ('Vm Distribution')
plt.ylabel('Density Distribution')
sns.distplot(Dist_A, bins = 25, color="#00aced", label="A", axlabel = 'Norm. Vm (Binned)', norm_hist = True)
sns.distplot(Dist_B, bins = 25, color="#224098", label="B", axlabel = 'Norm. Vm (Binned)', norm_hist = True)

del A_end, A_start, B_end, B_start, a, abf, b, current_Vm_A, current_Vm_A_norm, current_Vm_B, current_Vm_B_norm, current_Vm_ts_A, current_Vm_ts_A_norm, current_Vm_ts_B, current_Vm_ts_B_norm, data, i
# %% Cumulative Vm and UP state probability calculation for the complete Vm
A_cumulative = []
A_maxcum = ()

for i in range (len (data_A)):
    data = data_A[i]
    current_Vm_ts = data_ts_A [i]
    cumulative = np.cumsum (data) / 10000
    cum_max = np.amax (cumulative)
    A_cumulative.append (cumulative)
    A_maxcum = np.append (A_maxcum, cum_max)

B_cumulative = []
B_maxcum = ()

for i in range (len (data_B)):
    data = data_B[i]
    current_Vm_ts = data_ts_B [i]
    cumulative = np.cumsum (data) / 10000
    cum_max = np.amax (cumulative)
    B_cumulative.append (cumulative)
    B_maxcum = np.append (B_maxcum, cum_max)

data1 = [A_maxcum, B_maxcum]
names = ['FWE','SWE']
    
plt.figure(4)
plt.title ('Cumulative')
plt.ylabel('Cumulative Vm (mV*sec)')
plt.axis(['x1', 'x2', 0, 150])
qx = sns.boxplot (data = data1, palette=['#00aced', '#224098'])
ax = sns.swarmplot (data = data1, color = 'gray', edgecolor = 'gray')
ax.set(xticklabels=names)

A_Upstate = ()

for i in range (len (A_cumulative)):
    cumulative = A_cumulative [i]
    time_ts = data_ts_A [i]
    Pup_A = ((np.amax(A_cumulative [i]))/np.average(A_cumulative[i]))/np.amax(time_ts)
    A_Upstate = np.append (A_Upstate, Pup_A)
    
B_Upstate = ()

for i in range (len (B_cumulative)):
    cumulative = B_cumulative [i]
    time_ts = data_ts_B [i]
    Pup_B = ((np.amax(B_cumulative [i]))/np.average(B_cumulative[i]))/np.amax(time_ts)
    B_Upstate = np.append (B_Upstate, Pup_B)

data2 = [A_Upstate, B_Upstate]

plt.figure(5)
plt.title ('UP State probability')
plt.ylabel('P (Up State)')
plt.axis(['x1', 'x2', 0, 0.10])
qx = sns.boxplot (data = data2, palette=['#00aced', '#224098'])
ax = sns.swarmplot (data = data2, color = 'gray', edgecolor = 'gray')
ax.set(xticklabels=names)

del ax, cum_max, cumulative, current_Vm_ts, data, data1, data2, i, names, qx

#%% UP States Segmentation (ERASE BASELINE VM and NOISE), Computation and Size Estimation [FWHM]
A_NUP = []
A_UP_Vm = []
A_UP_FWHM = []
UP_A = []
UP_ts_A = []
A_Dist_UP = []

data_cutoff = Dist_A [Dist_A < 5]
A_cutoff = stats.median_absolute_deviation (data_cutoff, scale=1.4826) * 10 # Calculate baseline noise using MAD and *10x higher filtering

for i in range (len (data_A)):
    try:
        data_thresh = (data_A[i] >= A_cutoff).astype(int)
        labeled_array, num_features = label(data_thresh)
        A_NUP = np.append (A_NUP, num_features)
        Average_UP = []
        data_UP = []
        data_FWHM = []
        Average_FWHM = []
        x = 1     
        while x <= num_features:
                try:
                    a = np.where (labeled_array == x)
                    ind_start = min (a[0])
                    ind_end = max (a[0])
                    data = data_A [i]
                    data = data [(data <30)] #Cut-off to avoid detection of spikes
                    data_ts = data_ts_A [i]
                    Vm_cooordinates = data [ind_start - 20 :ind_end + 20]
                    ts_coordinates = data_ts [ind_start - 20 :ind_end + 20]
                    UP_A.append (Vm_cooordinates)
                    UP_ts_A.append (ts_coordinates)
                    Max_Vm_Upstate = np.amax (Vm_cooordinates)
                    data_UP = np.append (data_UP, Max_Vm_Upstate)
                    A_Dist_UP = np.append (A_Dist_UP, Max_Vm_Upstate)
                    FWHM_data = Max_Vm_Upstate/2
                    FWHM_crop = np.where (Vm_cooordinates > FWHM_data)
                    FWHM_value = (max (FWHM_crop[0])) - (min (FWHM_crop[0]))
                    data_FWHM = np.append (data_FWHM, FWHM_value)
                    print ('Job done!'+ ''+ 'A_Current:', i , x)
                    x += 1
                    
                except:
                    print ("Error on A_x:", i , x)
                    x += 1
        else:
            Average_UP = np.average (data_UP)
            A_UP_Vm.append (Average_UP)
            Average_FWHM = np.average (data_FWHM)
            A_UP_FWHM.append (Average_FWHM)
            
    except:
        break


"""plt.figure (20) #Visual control of UP states crop
for i in range (len (UP_A)):
    plt.plot (UP_ts_A[i], UP_A [i])"""


B_NUP = []
B_UP_Vm = []
B_UP_FWHM = []
UP_B = []
UP_ts_B = []
B_Dist_UP = []

data_cutoff = Dist_B [Dist_B < 5]
B_cutoff = stats.median_absolute_deviation(data_cutoff, scale=1.4826) * 10

for i in range (len (data_B)):
    try:
        data_thresh = (data_B[i] >= B_cutoff).astype(int)
        labeled_array, num_features = label(data_thresh)
        B_NUP = np.append (B_NUP, num_features)
        Average_UP = []
        data_UP = []
        data_FWHM = []
        Average_FWHM = []
        x = 1     
        while x <= num_features:
                try:
                    a = np.where (labeled_array == x)
                    ind_start = min (a[0])
                    ind_end = max (a[0])
                    data = data_SWE [i]
                    data = data [(data <30)]
                    data_ts = data_ts_B [i]
                    Vm_cooordinates = data [ind_start - 20 :ind_end + 20]
                    ts_coordinates = data_ts [ind_start - 20 :ind_end + 20]
                    UP_B.append (Vm_cooordinates)
                    UP_ts_B.append (ts_coordinates)
                    Max_Vm_Upstate = np.amax (Vm_cooordinates)
                    data_UP = np.append (data_UP, Max_Vm_Upstate)
                    B_Dist_UP = np.append (B_Dist_UP, Max_Vm_Upstate)
                    FWHM_data = Max_Vm_Upstate/2
                    FWHM_crop = np.where (Vm_cooordinates > FWHM_data)
                    FWHM_value = (max (FWHM_crop[0])) - (min (FWHM_crop[0]))
                    data_FWHM = np.append (data_FWHM, FWHM_value)
                    print ('Job done!'+ ''+ 'B_Current:', i , x)
                    x += 1
                    
                except:
                    print ("Error on B_x:", i , x)
                    x += 1
        else:
            Average_UP = np.average (data_UP)
            B_UP_Vm.append (Average_UP)
            Average_FWHM = np.average (data_FWHM)
            B_UP_FWHM.append (Average_FWHM)
            
    except:
        break

"""plt.figure (21)
for i in range (len (UP_B)):
    plt.plot (UP_ts_B[i], UP_B [i])"""


data3 = [A_UP_Vm, B_UP_Vm]
data4 = [A_UP_FWHM, B_UP_FWHM]
data5 = [A_NUP, B_NUP]
names = ['A','B']
    
plt.figure(6)
plt.title ('Average amplitude of UP State (mV)')
plt.ylabel('Resting Membrane Potential (mV)')
plt.axis(['x1', 'x2', 0, 30])
qx = sns.boxplot (data = data3, palette=['#00aced', '#224098'])
ax = sns.swarmplot (data = data3, color = 'gray', edgecolor = 'gray')
ax.set(xticklabels=names)

plt.figure(7)
plt.title ('Average FWHM of UP State (mV)')
plt.ylabel('FWHM')
plt.axis(['x1', 'x2', 0, 1000])
qx = sns.boxplot (data = data4, palette=['#00aced', '#224098'])
ax = sns.swarmplot (data = data4, color = 'gray', edgecolor = 'gray')
ax.set(xticklabels=names)

plt.figure(8)
plt.title ('Number of UP States')
plt.ylabel('Nr. UP STATES')
plt.axis(['x1', 'x2', 0, 200])
qx = sns.boxplot (data = data5, palette=['#00aced', '#224098'])
ax = sns.swarmplot (data = data5, color = 'gray', edgecolor = 'gray')
ax.set(xticklabels=names)

plt.figure(9)
plt.title ('Distribution of Vm during UP STATES')
plt.ylabel('Density Distribution')
plt.axis([0, 40, 0, 0.4])
sns.distplot(A_Dist_UP, bins = 25, color="#00aced", label="FWE", axlabel = 'Norm. Vm (Binned)', norm_hist = True)
sns.distplot(B_Dist_UP, bins = 25, color="#224098", label="SWE", axlabel = 'Norm. Vm (Binned)', norm_hist = True)

colors = ["#00aced", "#224098"]

plt.figure(10)
plt.ylabel('Vm count')
plt.xlabel('Resting Vm (mV)')
plt.hist([A_Dist_UP, B_Dist_UP], bins = 10, color=colors, alpha=1)

del Average_FWHM, Average_UP, Dist_A, Dist_B, A_Dist_UP, A_Upstate, A_cutoff, A_maxcum, FWHM_crop, FWHM_data, FWHM_value, Max_Vm_Upstate, Pup_A, Pup_B
del B_Dist_UP, B_Upstate, B_cutoff, B_maxcum, Vm_cooordinates, a, colors, data, data3, data4, data5, data_FWHM, data_UP, data_cutoff, data_thresh, data_ts
del i, ind_end, ind_start, labeled_array, names, num_features, time_ts, ts_coordinates, x