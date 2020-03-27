# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 15:45:56 2020

@author: tiago
"""
# %% IMPORT MODULES NEEDED FOR THE JOB [shift + enter 4run]
import pyabf
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import os
import csv

#%% DEFINE FUNCTION NEED TO DO THE JOB

#EXPONENTIAL FUNCTION

def exp_func(x, a, b, c):
    """
    Input: a,b,c,d variables of the exponential fitting
    Returns: bi-exponential function
    """
    return a*np.exp (b*x) + c

#TRANSPONSE FUNCTION
def func_transpose(x):
    """
    Parameters
    ----------
    x : list to be converted to numpy and re-transposed

    Returns Numpy and re-transposed version of the list
    -------
    """
    x = np.asarray (x)
    return np.transpose (x)

# %% IMPORT ABF FILE TO BE ANALYZED
directory = r'D:\AMPAinvivo\Analysis\Ephys\SWE\Inhibition\Evoked_Inhibition\abf_SWE'
data_abf = []

for filename in os.listdir(directory):
    if filename.endswith(".abf"):
        abf = pyabf.ABF (os.path.join(directory, filename))
        data_abf.append(abf)

Vm_ts = abf.sweepX
a = np.searchsorted(abf.sweepX, 0)
b = np.searchsorted(abf.sweepX, 0.06)
c = np.searchsorted(abf.sweepX, 0.1)
d = np.searchsorted(abf.sweepX, 0.14)

del abf
# %% Extract and Analyze whisker-evoked PSPs in UP STATE
file = 12 #FILE TO BE ANALYZED
abf=data_abf[file]
AnimalID=abf.abfID

try:
    plt.close(1)
    plt.close(2)
    plt.close(3)
    plt.close(4)
except:
    pass

Vm_distribution = np.array ([])

for sweepNumber in abf.sweepList:
    abf.setSweep(sweepNumber) 
    data = abf.sweepY [a:c]
    Vm_distribution = np.append (Vm_distribution, data)
    
histogram, bins = np.histogram (Vm_distribution, bins = 50)
ID_hist_max = np.argmax(histogram)
Vm_cutoff = bins[ID_hist_max] + 5 #Arbitary threshold of Vm_max + CUTOFF

sweepsUp = []
sweepsDown = []
sweepstrashed = []
PSPdown = []
PSPup = []
PSPtrashed = []

for sweepNumber in abf.sweepList:
    abf.setSweep(sweepNumber)
    if np.average(abf.sweepY [a:c]) > 0:
        continue
    else:
        if abf.sweepY[a] <= Vm_cutoff and abf.sweepY[b] <= Vm_cutoff and abf.sweepY[c] <= Vm_cutoff and abf.sweepY[a]:
            if abf.sweepY[d] > Vm_cutoff + 5:
                dataDown = abf.sweepY
                PSPdown.append (dataDown)
                sweepsUp.append (sweepNumber)
        
        elif abf.sweepY[b] > Vm_cutoff + 5 and abf.sweepY[c] > Vm_cutoff + 5 and abf.sweepY[d] < abf.sweepY[c]:
            dataUp = abf.sweepY
            PSPup.append (dataUp)
            sweepsUp.append (sweepNumber)
    
        else:
            datatrashed = abf.sweepY
            PSPtrashed.append (datatrashed)
            sweepstrashed.append (sweepNumber)

PSPup = func_transpose (PSPup)
PSPdown = func_transpose (PSPdown)
PSPtrashed = func_transpose (PSPtrashed)

plt.figure (1)
plt.title (abf.abfID +"_UP sorting")
plt.ylabel('Vm (mV)')
plt.xlabel('Time (sec.)')
plt.plot (Vm_ts, PSPdown, color = "b")
plt.plot (Vm_ts, PSPup, color = "r")
plt.savefig(abf.abfID +"_UP_sorting")
filename = abf.abfID +"_UP_sorting"
plt.savefig(os.path.join(directory, filename))

plt.figure (2)
plt.title (abf.abfID +"_Trashed PSPs")
plt.ylabel('Vm (mV)')
plt.xlabel('Time (sec.)')
plt.plot (Vm_ts, PSPtrashed, color = "g")
plt.savefig(abf.abfID +"_Trashed_PSPs")
filename = abf.abfID +"_Trashed_PSPs"
plt.savefig(os.path.join(directory, filename))

ind_start = np.searchsorted(Vm_ts, 0.123) # Adjust to optimize the fitting
ind_end = np.searchsorted(Vm_ts, 0.143)

normalization_factor = []
Up_state_Peak = []

for i in range (len (PSPup.T)):
    normalization = np.amin (PSPup[i].T)
    normalization_factor.append(normalization)

normalization_factor = func_transpose (normalization_factor)    
PSPup_norm = PSPup - normalization_factor

plt.figure (3)
plt.title (abf.abfID +"_PSP normalization")
plt.ylabel('Vm (mV)')
plt.xlabel('Time (sec.)')
plt.plot (Vm_ts, PSPup_norm, color = "b")
filename = abf.abfID +"_PSP_normalization"
plt.savefig(os.path.join(directory, filename))

Vm_up = PSPup_norm[ind_start:ind_end]
Up_ts = Vm_ts[ind_start:ind_end]

Up_state_Peak = []
curves = []
fitvalues = []
chi = []

for i in range (len (Vm_up.T)):
    try:
        c0 = min (Vm_up.T[i]) - 0.001
        a0 = max (Vm_up.T[i]) - c0
        b0 = np.log ((Vm_up.T[i]-c0)/a0)
        bap = (b0[1] - b0 [100])/(Up_ts[1]-Up_ts[100])
        if bap <0:
            popt, pcov = curve_fit (exp_func, Up_ts, Vm_up.T[i], p0=(a0, bap, c0))
            p1 = popt[0] # This is your a
            p2 = popt[1] # This is your b
            p3 = popt[2] # This is your c
            residuals = Vm_up.T[i] - exp_func(Up_ts,p1,p2,p3)
            fres = sum ((residuals**2)/exp_func(Up_ts,p1,p2,p3))
            if fres >= -10 and fres < 10:  #Needs to be improved
                fit = exp_func (Up_ts, p1, p2, p3)
                tau = - 1/p2
                curves.append (fit)
                fitvalues.append (tau)
                chi.append (fres)
                peak = np.amax (Vm_up[i].T)
                Up_state_Peak.append(peak)
    except:
        chi.append ("nan")
        fitvalues.append ('nan')
        Up_state_Peak.append('nan')

curves = func_transpose (curves)
Up_state_Peak = func_transpose (Up_state_Peak)

plt.figure (4)
plt.title (abf.abfID +"_Fitting Accuracy")
plt.ylabel('Vm (mV)')
plt.xlabel('Time (sec.)')
plt.plot (Up_ts, Vm_up, color = "b")
plt.plot (Up_ts, curves, color = "r")
filename = abf.abfID +"_Fitting_Accuracy"
plt.savefig(os.path.join(directory, filename))

with open (os.path.join(directory, filename + '.csv'), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(np.transpose(fitvalues))
    writer.writerow(np.transpose(chi))
    writer.writerow(np.transpose(Up_state_Peak))