# Comparing Leakage Detectors

This folder stores the results of a comparison between a leakage detector
trained on realistic demands and a detector trained on toy demands. Leaks were
maximized until detection for randomly selected junctions. In each case, the
leak was placed 5 days, 5 hours and 5 minutes after the start of the L-Town
time series. A BetweenSensorInterpolator with k=1 was used. In the case were
it was trained on real data, it used the old sensor configuration (see
`Data/L-Town/old_pressure_sensors.csv`) and a global threshold of 0.3. In the
second case were it was trained on toy data, it used the new sensor
configuration (see `Data/L-Town/pressure_sensors.csv`) and a threshold of 1.2.

- `picked_junctions.npy`: Junctions used for the analysis (selected randomly)
- `trained_on_real.npy`: Maximal undetected leak areas when the detector was trained on realistic data. This may be zipped with `picked_junctions` to get the associated junctions for each maximum.
- `trained_on_toy.npy`: Maximum undetected leak area when trained on toy data.
- `result_dataframe.py`: Script to create a result dataframe to view the different results in comparison. Run  
``` python -i result_dataframe.py```  
and then enter  
``` >>> df ```  
in the python prompt to view the dataframe.
