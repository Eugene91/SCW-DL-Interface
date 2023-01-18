# README
The project contains Python code for asymptotic secret key simulation for SCW-QKD under collective beam-splitter attack that is used
in the manuscript [arXiv:2209.11719](https://arxiv.org/abs/2209.11719).


## Structure of the projects

File MeasurementScheme.py contains classes for estimating asymptotic secret key for conventional SCW-QKD scheme and the one-way SCW-QKD with the proposed interface being used two state discriminator.
The simulation takes into account parameters such as losses, amplitude, modulation depth, detector's dark counting rate and efficiency, gating time and repetition period and spectral filter transmission.  

File KeyRateOptimization.py finds optimal experimental control parameters, such as carrier amplitude and modulation depth at given range of losses. The optimization is done by the non-gradient method with the help of the [nlopt](https://nlopt.readthedocs.io/en/latest/) library.

Jupyter notebook Result-Visualisation.ipynb is used to call the optimization procedure and visualise the results, i.e. the key rate for  the both scheme and the set of optimal parameters. 

## Requirements
numpy >= 1.18.5
scipy >= 1.5.0
nlopt >= 2.6.2
matplotlib >= 3.2.2
for latex rendering 
texlive-latex-base >= 2019
texlive-latex-picture >= 2019