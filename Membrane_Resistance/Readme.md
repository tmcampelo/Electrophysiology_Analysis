**Extract membrane resistance from a hyperpolarizing square pulse before a whisker-evoked PSP.**


**Try the script on the example "Evoked_PSP.abf". A short description of the code is given below:**

1) Define exponential function *(a*np.exp (b*x) + c)* used to extract membrane resistance;


2) Import ".abf" file into python using the pyabf module;


3) Sorting of DOWN state whisker-evoked PSPs. Membrane resistance is calculated exclusively from those traces.
The cut-off UP/DOWN states is determined from the histogram distribution of the Vm. A higher count of Vm is expected to occur during the DOWN state (anaesthesia). The cut-off is the Vm where this count is max (bins[ID_hist_max]) + "5" (arbitrary value)  *[Under optimization - Gaussian fitting would be prefered]*

<p align="center">
  <img src="https://github.com/tmcampelo/Electrophysiology_Analysis/blob/master/Membrane_Resistance/Example_Figures/Rin_UpStatesRemoved5.jpg">
</p>



4)


