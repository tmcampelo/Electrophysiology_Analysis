# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 10:50:35 2019

@author: tiago
"""
# %% IMPORT MODULES NEEDED FOR THE JOB [shift + enter 4run]
import pyabf
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


#%% DEFINE THE EXPONENTIAL FUNCTION
def exp_func(x, a, b, c):
    """
    Input: a,b,c,d variables of the exponential fitting
    Returns: exponential function
    """
    return a*np.exp (b*x) + c

# %% IMPORT ABF FILE TO BE ANALYZED
abf = pyabf.ABF ("Evoked_PSP.abf")
PSPdown = []
Vm_distribution = np.array ([])
Vm_ts = abf.sweepX

# %% Extract sweeps from DOWN STATES
a = np.searchsorted(abf.sweepX, 0)
b = np.searchsorted(abf.sweepX, 0.06)
c = np.searchsorted(abf.sweepX, 0.1)

for sweepNumber in abf.sweepList:
    abf.setSweep(sweepNumber) 
    data = abf.sweepY [a:c]
    Vm_distribution = np.append (Vm_distribution, data)
    
histogram, bins = np.histogram (Vm_distribution, bins = 50)
ID_hist_max = np.argmax(histogram)
Vm_cutoff = bins[ID_hist_max] + 5 #Arbitrary threshold of Vm_max + CUTOFF

sweepsUp = []
sweepsDown = []

for sweepNumber in abf.sweepList:
    abf.setSweep(sweepNumber)
    
    if abf.sweepY[a] <= Vm_cutoff and abf.sweepY[b] <= Vm_cutoff and abf.sweepY[c] <= Vm_cutoff:
        dataDown = abf.sweepY
        PSPdown.append (dataDown)
        sweepsDown.append (sweepNumber)
    else:
        sweepsUp.append (sweepNumber)
    
PSPdown = np.asarray (PSPdown)
PSPdown = np.transpose (PSPdown)

plt.figure (1)
plt.hist(Vm_distribution, bins = 50, color = "b")
plt.axvline (x=Vm_cutoff, color='r')
plt.axvline (x=min(bins), color='r')
plt.axvline (x=bins[ID_hist_max], color='y')
plt.ylabel('Counts')
plt.xlabel('Vm (binned)')

plt.figure (2)
plt.title ("Aaccuracy of DOWN/UP sorting")
plt.ylabel('Vm (mV)')
plt.xlabel('Time (sec.)')
plt.plot (Vm_ts, PSPdown)

del Vm_cutoff, data, dataDown, sweepNumber, histogram, bins, ID_hist_max, Vm_distribution, a, b, c
# %% EXPONENTIAL FITTING TO SEAL TEST
#Determine the "X" coordinates to crop the seal test
ind_start = np.searchsorted(Vm_ts, 0.007)
ind_end = np.searchsorted(Vm_ts, 0.018)

#Crop the evoked response using the aforementioned coordinates
seal_vm = PSPdown[ind_start:ind_end]
seal_ts = Vm_ts[ind_start:ind_end]

sealtest = seal_vm - np.average(seal_vm[0:10], axis = 0)

plt.figure (3)
plt.ylabel('Vm norm. (mV)')
plt.xlabel('Time (sec.)')
plt.plot (seal_ts, sealtest)
plt.savefig(abf.abfID +'_seal_test.png')

curves = []
fitvalues = []
chi = []

for i in range (len (sealtest.T)):
    try:
        c0 = min (sealtest.T[i]) - 0.001
        a0 = max (sealtest.T[i]) - c0
        b0 = np.log ((sealtest.T[i]-c0)/a0)
        bap = (b0[1] - b0 [100])/(seal_ts[1]-seal_ts[100])
        if bap <0:
            popt, pcov = curve_fit (exp_func, seal_ts, sealtest.T[i], p0=(a0, bap, c0))
            p1 = popt[0] # This is your a
            p2 = popt[1] # This is your b
            p3 = popt[2] # This is your c
            residuals = sealtest.T[i] - exp_func(seal_ts,p1,p2,p3)
            fres = sum ((residuals**2)/exp_func(seal_ts,p1,p2,p3))
            if fres >= -15:  #Needs to be improved
                fit = exp_func (seal_ts, p1, p2, p3)
                curves.append (fit)
                fitvalues.append (p3)
                chi.append (fres)
            else:
                fitvalues.append ('nan')
                chi.append ("nan")
        else:
            fitvalues.append ('nan')
            chi.append ("nan")
    except:
        chi.append ("nan")
        fitvalues.append ('nan')

curves = np.asarray (curves)
curves= np.transpose (curves)
cleanedFits = [x for x in fitvalues if str(x) != 'nan']
Average_RM = np.mean(cleanedFits)

plt.figure (4)
plt.ylabel('a*np.exp (b*x) + c')
plt.xlabel('Time (sec.)')
plt.plot (seal_ts, curves)
plt.savefig(abf.abfID + '_fitts.png')
   
del a0, b0, bap, c0, fit, fres, i, ind_end, ind_start, p1, p2, p3, pcov, popt, residuals, seal_vm, cleanedFits, PSPdown, Vm_ts

# TimePlots of C value and Chi of the fitting
plt.figure (5)
plt.ylabel("c value (mV)")
plt.xlabel("Sweeps")
plt.scatter (sweepsDown, fitvalues)
plt.savefig(abf.abfID +'_c value (mV).png')

plt.figure (6)
plt.ylabel("Fitting accuracy (Chi square)")
plt.xlabel("Sweeps")
plt.scatter (sweepsDown, chi)
plt.savefig(abf.abfID + '_fitting_accuracy.png')