**Extract membrane resistance from a hyperpolarizing square pulse before a whisker-evoked PSP.**


**Try the script on the example "Evoked_PSP.abf". A short description of the code is given below:**

1) Define exponential function used to extract membrane resistance;


2) Import ".abf" file into python using the pyabf module;


3) Sorting of DOWN state whisker-evoked PSPs. Membrane resistance is calculated exclusively from those traces.
The cut-off UP/DOWN states is determined from the histogram distribution of the Vm. A higher count of Vm is expected to occur during the DOWN state (anaesthesia). The cut-off is the Vm where this count is max (bins[ID_hist_max]) + "5" (arbitrary value)  *[Under optimization - Gaussian fitting would be prefered]*

<p align="center">
  <img src="https://github.com/tmcampelo/Electrophysiology_Analysis/blob/master/Membrane_Resistance/Example_Figures/Rin_UpStatesRemoved5.jpg">
</p>



4) Crop the seal test from the entire trace (given by ind_start / ind_end) and normalize to baseline;


5) Iterate over the complet list of sweeps to fit the exponential fitting to the seal test;
This is given by the function *popt, pcov = curve_fit (exp_func, seal_ts, sealtest.T[i], p0=(a0, bap, c0))*
a0, bap, c0 are the initial guesses calculated to each independent sweep.


<p align="center">
  <img src="https://github.com/tmcampelo/Electrophysiology_Analysis/blob/master/Membrane_Resistance/Example_Figures/Rin_Fitting.jpg">
</p>


6) The fitting gives the "c value" (see exponential fitting) in mV. The membrane resistance can be then calculated accordingly to Ohm's law, where the Electric current (I) is derived from the "c value" accordingly to the gain of recording system.


IMPORTANT: the Chi square of the exponential fitting can be used to keep the sweeps with appropriated fitting.


<p align="center">
  <img src="https://github.com/tmcampelo/Electrophysiology_Analysis/blob/master/Membrane_Resistance/Example_Figures/Rin_Cvalue.jpg">
</p>
