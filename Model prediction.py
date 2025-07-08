# -*- coding: utf-8 -*-
"""

@author: ruizhen_wang
"""

import pandas as pd
import numpy as np
import os
import rasterio
from xgboost import XGBRegressor
from datetime import datetime
df = pd.read_excel('D:/SMC_retrieval/predictors_dataset_lag5/SMC_database_lag5_4.28.xlsx')
#the prediction soil layer depth, selecting from below:
#05, 515, 1530, 3060
depth = '3060'
date_str = "2019-08-08"
batch_size_30m = 100000


predictor_depths = ['05', '515', '1530', '3060']
target_depths = {'05':'5cm', '515':'10cm', '1530':'20cm', '3060':'50cm'}
current_idx = predictor_depths.index(depth)
deeper_layers = predictor_depths[current_idx+1:]  # 当前深度之后的
for layer in deeper_layers:
    for item in df.columns:
        if layer in item:
            df.drop(columns=item, inplace=True)
predictor_depths.remove(depth)
for layer in predictor_depths:
    for item in df.columns:
        if target_depths[layer] in item:
        #if layer in item or target_depths[layer] in item:
            df.drop(columns = item, inplace = True)
            
df['date'] = pd.to_datetime(df['date'])
df.drop(columns = ['station', 'date'], inplace = True)

#参考影像信息提取，用于生成预测土壤属性图
nodata_value = -9999
with rasterio.open('D:/SMC_retrieval/Shapefile/Study_area_reference.tif') as src:
    target_data = src.read(1)
    target_transform = src.transform
    height, width = target_data.shape
#预测数据导入
predictor_path_30m = 'D:/SMC_retrieval/prediction example/'
predictor_dataset_30m = pd.DataFrame([])
for root, dirs, files in os.walk(predictor_path_30m):
    for dir in dirs:
        dir_path = os.path.join(root, dir)
        print(f"Contents of {dir_path}:")
        for file in os.listdir(dir_path):
            print(file + ' is under processing.')
            with rasterio.open(dir_path + '/' + file) as src:
                arr = src.read(1).flatten()
            predictor_dataset_30m[file.split('.')[0]] = arr
raster_shape_30m = src.shape
rows_30m, cols_30m, = src.shape
for layer in deeper_layers:
    for item in predictor_dataset_30m.columns:
        if layer in item:
            predictor_dataset_30m.drop(columns=item, inplace=True)
for layer in predictor_depths:
    for item in predictor_dataset_30m.columns:
        if target_depths[layer] in item:
        #if layer in item or target_depths[layer] in item:
            predictor_dataset_30m.drop(columns = item, inplace = True)
date = datetime.strptime(date_str, "%Y-%m-%d")            
predictor_dataset_30m['doy'] = date.timetuple().tm_yday
print('Data loading complete')

#地表覆盖研磨
with rasterio.open('D:/SMC_retrieval/landcover/CLCD_v01_2019.tif') as src:
    landcover = pd.DataFrame(src.read(1).flatten(), columns=['landcover'])


####data filtering (different depths)
X_train = df.iloc[:,:-3]
Y_train = df.iloc[:,-3:]
predictor_dataset_30m.columns=X_train.columns
rf = XGBRegressor(n_estimators=100, random_state=42, learning_rate=0.07) 

 
#day1
X_train1 = X_train.drop(columns = ['wind_2', 'wind_3', 'pre_2', 'pre_3', 'temp_2', 'temp_3'])
Y_train1 = Y_train.iloc[:,0]
rf.fit(X_train1, Y_train1)
predictor_dataset_30m1 = predictor_dataset_30m.drop(columns = ['wind_2', 'wind_3', 'pre_2', 'pre_3', 'temp_2', 'temp_3'])
prediction1 = rf.predict(predictor_dataset_30m1)
zero_indices1 = predictor_dataset_30m1[predictor_dataset_30m1['NDVI'] == 0].index
zero_indices2 = landcover[landcover['landcover'] == 5].index
zero_indices3 = landcover[landcover['landcover'] == 8].index
prediction1[zero_indices1] = -9999
prediction1[zero_indices2] = -9999
prediction1[zero_indices3] = -9999
output1 = np.array(prediction1).reshape(target_data.shape)
with rasterio.open(predictor_path_30m + 'SM_' + target_depths[depth]+'_'+ date_str + '_day1.tif', 'w', driver='GTiff',  count=1, dtype=output1.dtype, 
                   height=height, width=width, crs=src.crs,nodata=-9999, transform=target_transform) as dst:
    dst.write(output1, 1)



# #day2
X_train2 = X_train.drop(columns = ['wind_3', 'pre_3', 'temp_3'])
Y_train2 = Y_train.iloc[:,1]  
rf.fit(X_train2, Y_train2)
predictor_dataset_30m2 = predictor_dataset_30m.drop(columns = ['wind_3', 'pre_3', 'temp_3'])
prediction2 = rf.predict(predictor_dataset_30m2)
zero_indices1 = predictor_dataset_30m1[predictor_dataset_30m1['NDVI'] == 0].index
zero_indices2 = landcover[landcover['landcover'] == 5].index
zero_indices3 = landcover[landcover['landcover'] == 8].index
prediction2[zero_indices1] = -9999
prediction2[zero_indices2] = -9999
prediction2[zero_indices3] = -9999
output2 = np.array(prediction2).reshape(target_data.shape)
with rasterio.open(predictor_path_30m + 'SM_' + target_depths[depth]+'_'+ date_str + '_day2.tif', 'w', driver='GTiff',  count=1, dtype=output1.dtype, 
                   height=height, width=width, crs=src.crs,nodata=-9999, transform=target_transform) as dst:
    dst.write(output2, 1)


# #day3
X_train3 = X_train
Y_train3 = Y_train.iloc[:,2]    
rf.fit(X_train3, Y_train3)
predictor_dataset_30m3 = predictor_dataset_30m
prediction3 = rf.predict(predictor_dataset_30m3)
zero_indices1 = predictor_dataset_30m1[predictor_dataset_30m1['NDVI'] == 0].index
zero_indices2 = landcover[landcover['landcover'] == 5].index
zero_indices3 = landcover[landcover['landcover'] == 8].index
prediction3[zero_indices1] = -9999
prediction3[zero_indices2] = -9999
prediction3[zero_indices3] = -9999
output3 = np.array(prediction3).reshape(target_data.shape)
with rasterio.open(predictor_path_30m + 'SM_' + target_depths[depth]+'_'+ date_str + '_day3.tif', 'w', driver='GTiff',  count=1, dtype=output1.dtype, 
                   height=height, width=width, crs=src.crs, nodata=-9999, transform=target_transform) as dst:
    dst.write(output3, 1)


