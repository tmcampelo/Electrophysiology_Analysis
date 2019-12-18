#Python code to extract membrane resistance from a 100-ms long-lasting hyperpolarizing square pulse (400 ms) before a whisker-evoked PSP (current-clamp mode).
#You can run it on the example "Evoked_PSP.abf" file. A short description of the code is given below:

**1) Define an function for the exponential fitting that will be used to extract the membrane resistance from the seal test;**
**2) Import ".abf" file into python using the pyabf module;**
**3) Sorting of DOWN state whisker-evoked PSPs. The script will only analyze sweeps where Vm is stable during test seal**


Note: UP and DOWN states sweeps are defined accordingly to the distribution of Vm before whisker deflection (higher count correspond to the down state).

![alt text](https://github.com/tmcampelo/Electrophysiology_Analysis/blob/master/Membrane_Resistance/Example_Figures/Histogram.png)

# UNDER CONSTRUCTION
