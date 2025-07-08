# -*- coding: utf-8 -*-
"""

@author: ruizhen_wang
"""

import pandas as pd
import numpy as np
import os
from scipy.stats import pearsonr
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error

df = pd.read_excel('E:/SMC_retrieval/predictors_dataset_lag5/SMC_database_lag5_4.28.xlsx')

#the prediction soil layer depth, selecting from below:
#05, 515, 1530, 3060
depth = '3060'
    
####data filtering (different depths)
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

output = pd.DataFrame([])

stations = list(df['station'].unique())
for station in stations:
    df_test =  df[df['station'] == station]
    df_train= df[df['station'] != station]

    df_train.drop(columns = ['station', 'date'], inplace = True)
    df_test.drop(columns = ['station', 'date'], inplace = True)
    X_train = df_train.iloc[:,:-3]
    Y_train = df_train.iloc[:,-3:]
    X_test = df_test.iloc[:,:-3]
    Y_test = df_test.iloc[:,-3:]
    multi_output_rf = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, learning_rate=0.07)  
    #day1    
    X_train1 = X_train.drop(columns = ['wind_2', 'wind_3', 'pre_2', 'pre_3', 'temp_2', 'temp_3'])
    X_test1 = X_test.drop(columns = ['wind_2', 'wind_3', 'pre_2', 'pre_3', 'temp_2', 'temp_3'])    
    Y_train1 = Y_train.iloc[:,0]
    Y_test1 = Y_test.iloc[:,0]     
    multi_output_rf.fit(X_train1, Y_train1)
    Y_pred = multi_output_rf.predict(X_test1)
    rmse1 = np.sqrt(mean_squared_error(Y_test['SM_'+ target_depths[depth] + '_1'], Y_pred))
    r1,_ = pearsonr(Y_test['SM_'+ target_depths[depth] + '_1'], Y_pred)
    bias1 = sum(Y_pred-Y_test['SM_'+ target_depths[depth] + '_1'])/len(Y_test['SM_'+ target_depths[depth] + '_1'])
    ubRMSE1 = (rmse1**2-bias1**2)**0.5
    print(f'RMSE1: {rmse1}')
    print(f'R Score 1: {r1}')
    print('Bias1:' + str(bias1))
    print('ubRMSE1:' + str(ubRMSE1))
    print('                ')
    
    
    
    #day2
    X_train2 = X_train.drop(columns = ['wind_3', 'pre_3', 'temp_3'])
    X_test2 = X_test.drop(columns = ['wind_3', 'pre_3','temp_3'])    
    Y_train2 = Y_train.iloc[:,1]
    Y_test2 = Y_test.iloc[:,1]
    multi_output_rf.fit(X_train2, Y_train2)
    Y_pred = multi_output_rf.predict(X_test2)
    rmse2 = np.sqrt(mean_squared_error(Y_test['SM_'+ target_depths[depth] + '_2'], Y_pred))
    r2,_ = pearsonr(Y_test['SM_'+ target_depths[depth] + '_2'], Y_pred)
    bias2 = sum(Y_pred-Y_test['SM_'+ target_depths[depth] + '_2'])/len(Y_test['SM_'+ target_depths[depth] + '_2'])
    ubRMSE2 = (rmse2**2-bias2**2)**0.5
    print(f'RMSE2: {rmse2}')
    print(f'R Score 2: {r2}')
    print('Bias2:' + str(bias2))
    print('ubRMSE2:' + str(ubRMSE2))
    print('                ')
    
    #day3
    X_train3 = X_train
    X_test3 = X_test
    Y_train3 = Y_train.iloc[:,2]
    Y_test3 = Y_test.iloc[:,2]
    multi_output_rf.fit(X_train3, Y_train3)
    Y_pred=multi_output_rf.predict(X_test3)
    rmse3 = np.sqrt(mean_squared_error(Y_test['SM_'+ target_depths[depth] + '_3'], Y_pred))
    r3,_ = pearsonr(Y_test['SM_'+ target_depths[depth] + '_3'], Y_pred)
    bias3 = sum(Y_pred-Y_test['SM_'+ target_depths[depth] + '_3'])/len(Y_test['SM_'+ target_depths[depth] + '_3'])
    ubRMSE3 = (rmse3**2-bias3**2)**0.5
    print(f'RMSE3: {rmse3}')
    print(f'R Score 3: {r3}')
    print('Bias3:' + str(bias3))
    print('ubRMSE3:' + str(ubRMSE3))
    
    temp = pd.DataFrame(np.array([station, r1, rmse1, bias1, ubRMSE1,  r2, rmse2, bias2, ubRMSE2, r3, rmse3, bias3, ubRMSE3]).reshape(1,-1),
              columns=['station', 'r1', 'rmse1','bias1','ubrmse1', 'r2', 'rmse2','bias2','ubrmse2', 'r3', 'rmse3','bias3','ubrmse3'])
    output = pd.concat([output,temp]) 
    
    print(station + ' has been processed.')

output.to_excel('E:/SMC_retrieval/predictors_site_4.28/site_independent_'+ target_depths[depth] +'.xlsx')



