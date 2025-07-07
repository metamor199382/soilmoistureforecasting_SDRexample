# soilmoistureforecasting_SDRexample

This is the code for the manuscript "Short-term Forecasting and Fine-resolution Mapping of Multi-layer Soil Moisture Using Remote Sensing Image Fusion and Ensemble Learning" submitted to Computers and Geosciences.

Ruizhen Wang, Weitao Chen, Pingkang Wang, Hao Chen, Xuwen Qin

All code mentioned in this paper is either already open-source and on github. 

The Python code developed by Nietupski et al. to perform image fusion on Google Earth Engine can be found on Github (https://github.com/tytupski/GEE-Image-Fusion), which was used to derive long-term NDVI and NSDSI in this study (Times 10000 for storing as int16 type). 

The dataset for this study was developed by extracting pixel value of variables using locations of 34 SDR-SMN stations. And the extracted information are in the file SMC-database-lag5-4.28.xlsx with the pre-processes, which can be used for performing the model training and evaluation.

Prediction example gives the input 30-m resolution raster data for generating the soil moisture for the example date by using Model prediction.py.


Other Reference
Nietupski, T. C., Kennedy, R. E., Temesgen, H., & Kerns, B. K. (2021). Spatiotemporal image fusion in Google Earth Engine for annual estimates of land surface phenology in a heterogenous landscape. International journal of applied earth observation and geoinformation, 99, 102323.
