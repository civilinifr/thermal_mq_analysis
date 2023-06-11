# LSPE Thermal Moonquake Analysis 
[![DOI](https://zenodo.org/badge/652235845.svg)](https://zenodo.org/badge/latestdoi/652235845)

Supporting information for JGR-Planets paper "Thermal moonquake characterization and cataloging using frequency-based algorithms and stochastic gradient descent"

Most of this repository contains code can be used to convert seismic data, run analysis, and reproduce every figure displayed in the paper. The rest are miscellaneous data required to run the code. 

Data and catalog produced by this workflow is available on CaltechData at:
https://doi.org/10.22002/gc871-q2a25

Code:
- *P00_sum_evts.py*: Sums the events in the Civilini 2021 catalog and displays them as a daily bar plot. 
- *P01_ascii2sac.py*: Converts daily ASCII data to hourly SAC data in decompressed volts. 
- *P02_plot_thermal_mq.py*: Plots the thermal moonquakes detected by the Civilini et al. 2021 catalog and saves their data as PKL files for easy implementation in other steps.
- *P03_pick_events.py*: Pick the start of the trace for each geophone of the LSPE for a random sampling of impulsive events.
- *P04_test_fine_tune_hyperparameters.py*: Checks finetuning hyperparameters based on the results obtained in P03
- *P05A_fine_tune_detection.py*: Finetunes the original detections based on parameters obtain in P04. 
- *P05B_test_xcorr_finetuning.py*: Assesses the potential of using cross-correlation to finetune the arrival times.  
- *P06_convert_sac_to_physunits.py*: Converts the hourly SAC data from decompressed volts to physical units (nm/s)
- *P07_calculate_PGV.py*: Computes Peak Ground Velocity (PGV) from the converted data for each seismic event. 
- *P08_combine_into_catalog.py*: Combines the finetuned arrival times and PGV measurements into a catalog. 
- *P09_check_wavespeed.py*: Computes the seismic wave speed for each finetuned event. 
- *P10_find_synthetic_azimuth.py*: Tests a range of Stochastic Gradient Descent (SGD) parameters for azimuth determination using a synthetic test. 
- *P10B_find_synthetic_azimuth_velvar_change.py*: Tests the preferred SGD parameters from P10 to assess their accuracy when the actual seismic velocity is different from the assumed velocity. 
- *P11_find_azimuth.py*: Computes the azimuth for the real finetuned events. Each event runs SGD multiple times. 
- *P12_combine_azimuth_results.py*: Combines all the azimuth results obtained in P11 and computes mean azimuth and variance. 
- *P13_combine_all_catalog.py*: Creates a final catalog combining all obtained results. 
- *P14A_check_PGV_values.py*: Plots PGV information for the dataset.
- *P14B_check_PGV_values_noLM.py*: Plots PGV information of events NOT from the lunar module (LM).
- *P14C_check_PGV_values_onlyLM.py*: Plots PGV information of events ONLY from the lunar module (LM).
- *P15_assess_catalog.py*: Do a statistical analysis of the moonquake catalog.
- *P16_investigate_daynight_evids.py*: Compares day and night impulsive and emergent events.
- *P17_plot_longterm_timeseries.py*: Check continuous spectrogram segements for changes during night-day transitions.
- *P18A_sunrise_burst_analysis.py*: Visual way to pick the start and end time of the two bursts occurring at sunrise for each cycle.
- *P18B_sunrise_burst_analysis_computation.py*: Assesses the moonquake start times picked in P18A.

Misc Data:
- *conda_thermal_env_list.txt*: Anaconda package list for the environment used to run the code. 
- *fc_lspe_dl_catalog.csv*: Original thermal moonquake catalog from Civilini et al. [2021] (DOI: https://doi.org/10.1093/gji/ggab083).
- *longterm_thermal_data.txt*: Temperature data for the Apollo 17 landing site from Molaro et al. [2017] (DOI: http://dx.doi.org/10.1016/j.icarus.2017.03.008) .
- *nightfall_burst.txt*: Datetimes of when the nightfall burst starts. Obtained from visual inspection using code P17.
- *resp.pkl*: Instrument response curves obtained from Nunn et al. [2020] (DOI: https://doi.org/10.1007/s11214-020-00709-3).
- *sunrise_burst.txt*: Datetimes of when the sunrise burst starts. Obtained from visual inspection using code P17.