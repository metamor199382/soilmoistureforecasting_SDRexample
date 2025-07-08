# -*- coding: utf-8 -*-
"""

@author: ruizhen_wang
"""

import pandas as pd
import numpy as np
import os
from xgboost import XGBRegressor
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 22

df = pd.read_excel('E:/SMC_retrieval/predictors_dataset_lag5/SMC_database_lag5_4.28.xlsx')

#the prediction soil layer depth, selecting from below:
#05, 515, 1530, 3060
depth = '05'
    
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



#df.drop(columns = 'SM_' +target_depths[depth], inplace = True)
df['date'] = pd.to_datetime(df['date'])

df_train= df[df['date'].dt.year.isin([2018,2019])].copy()
df_test = df[df['date'].dt.year == 2020].copy()

#seasonal evaluation
#df_train = df[~((df['date'].dt.year == 2020) & (df['date'].dt.month.isin([9, 10, 11])))].copy()
#df_test = df[(df['date'].dt.year == 2020) & (df['date'].dt.month.isin([7,8,9]))].copy()

#记录测试点位与日期
df_test_id = df_test[['station', 'date', 'SM_' +target_depths[depth] + '_1', 'SM_' +target_depths[depth] + '_2',
                      'SM_' +target_depths[depth] + '_3']].copy()

df_test_id = df_test_id.copy()

df_train.drop(columns = ['station', 'date'], inplace = True)
df_test.drop(columns = ['station', 'date'], inplace = True)
X_train = df_train.iloc[:,:-3]
Y_train = df_train.iloc[:,-3:]
X_test = df_test.iloc[:,:-3]
Y_test = df_test.iloc[:,-3:]
rf = XGBRegressor(n_estimators=100, random_state=42, learning_rate=0.07)       
X_train1 = X_train.drop(columns = ['wind_2', 'wind_3', 'pre_2', 'pre_3', 'temp_2', 'temp_3'])
X_test1 = X_test.drop(columns = ['wind_2', 'wind_3', 'pre_2', 'pre_3', 'temp_2', 'temp_3'])    
Y_train1 = Y_train.iloc[:,0]
Y_test1 = Y_test.iloc[:,0]

rf.fit(X_train1, Y_train1)
Y_pred = rf.predict(X_test1)
N=len(Y_pred)
df_test_id['Predict_SM_'+ target_depths[depth] + '_1'] = Y_pred
#day1
rmse1 = np.sqrt(mean_squared_error(Y_test['SM_'+ target_depths[depth] + '_1'], Y_pred))
r1,_ = pearsonr(Y_test['SM_'+ target_depths[depth] + '_1'], Y_pred)
bias1 = sum(Y_pred-Y_test['SM_'+ target_depths[depth] + '_1'])/len(Y_test['SM_'+ target_depths[depth] + '_1'])
ubRMSE1 = (rmse1**2-bias1**2)**0.5
print(f'RMSE1: {rmse1}')
print(f'R Score 1: {r1}')
print('Bias1:' + str(bias1))
print('ubRMSE1:' + str(ubRMSE1))
print('                ')

plt.figure(figsize=(8, 6), dpi=400)

sns.kdeplot(x=Y_test['SM_'+ target_depths[depth] + '_1'], 
            y=Y_pred, 
            fill=True, cmap='Blues', thresh=0.05, levels=100)

# 添加 y=x 参考线
plt.plot([Y_test['SM_'+ target_depths[depth] + '_1'].min(), Y_test['SM_'+ target_depths[depth] + '_1'].max()], 
         [Y_test['SM_'+ target_depths[depth] + '_1'].min(), Y_test['SM_'+ target_depths[depth] + '_1'].max()], 
         'k--', label='y=x')

# 设置轴标签
plt.xlabel('Observed SM (cm$^3$/cm$^3$)', fontsize=24)
plt.ylabel('Predicted SM (cm$^3$/cm$^3$)', fontsize=24)

# 设置刻度和范围
plt.xticks(np.arange(0, 0.5, 0.1), fontsize=24)
plt.yticks(np.arange(0, 0.5, 0.1), fontsize=24)
plt.xlim(0, 0.5)
plt.ylim(0, 0.5)

textstr = '\n'.join((
    f'N = {N}',
    f'R = {r1:.3f}',
    f'RMSE = {rmse1:.3f}',
    f'Bias = {bias1:.3f}',
    f'ubRMSE = {ubRMSE1:.3f}'
))

# 设置文本框的位置，左上角坐标 (0.05, 0.95)
plt.gcf().text(0.15, 0.85, textstr, fontsize=22, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

# 手动添加色条：从 plt.cm 获取颜色映射，设定范围为0-1之间
sm = plt.cm.ScalarMappable(cmap='Blues')
sm.set_array([])  # 手动设置空数据
cbar =plt.colorbar(sm)
cbar.ax.tick_params(labelsize=24)
plt.grid(True, linestyle='--', alpha=0.6)
plt.title('SM at ' + target_depths[depth] +' layer (Day 1)', fontsize=24)
plt.savefig('E:/SMC_retrieval/成图文件/'+ 'SM at ' + target_depths[depth] +' layer (Day 1)'+'.png', dpi=400, bbox_inches='tight')
plt.show()


#day2
X_train2 = X_train.drop(columns = ['wind_3', 'pre_3', 'temp_3'])
X_test2 = X_test.drop(columns = ['wind_3', 'pre_3','temp_3'])    
Y_train2 = Y_train.iloc[:,1]
Y_test2 = Y_test.iloc[:,1]
    
rf.fit(X_train2, Y_train2)
Y_pred = rf.predict(X_test2)   
df_test_id['Predict_SM_'+ target_depths[depth] + '_2'] = Y_pred   

rmse2 = np.sqrt(mean_squared_error(Y_test['SM_'+ target_depths[depth] + '_2'], Y_pred))
r2,_ = pearsonr(Y_test['SM_'+ target_depths[depth] + '_2'], Y_pred)
bias2 = sum(Y_pred-Y_test['SM_'+ target_depths[depth] + '_2'])/len(Y_test['SM_'+ target_depths[depth] + '_2'])
ubRMSE2 = (rmse2**2-bias2**2)**0.5
print(f'RMSE2: {rmse2}')
print(f'R Score 2: {r2}')
print('Bias2:' + str(bias2))
print('ubRMSE2:' + str(ubRMSE2))
print('                ')

plt.figure(figsize=(8, 6), dpi=400)

sns.kdeplot(x=Y_test['SM_'+ target_depths[depth] + '_1'], 
            y=Y_pred, 
            fill=True, cmap='Blues', thresh=0.05, levels=100)

# 添加 y=x 参考线
plt.plot([Y_test['SM_'+ target_depths[depth] + '_1'].min(), Y_test['SM_'+ target_depths[depth] + '_1'].max()], 
         [Y_test['SM_'+ target_depths[depth] + '_1'].min(), Y_test['SM_'+ target_depths[depth] + '_1'].max()], 
         'k--', label='y=x')

# 设置轴标签
plt.xlabel('Observed SM (cm$^3$/cm$^3$)', fontsize=24)
plt.ylabel('Predicted SM (cm$^3$/cm$^3$)', fontsize=24)

# 设置刻度和范围
plt.xticks(np.arange(0, 0.5, 0.1), fontsize=24)
plt.yticks(np.arange(0, 0.5, 0.1), fontsize=24)
plt.xlim(0, 0.5)
plt.ylim(0, 0.5)

textstr = '\n'.join((
    f'N = {N}',
    f'R = {r2:.3f}',
    f'RMSE = {rmse2:.3f}',
    f'Bias = {bias2:.3f}',
    f'ubRMSE = {ubRMSE2:.3f}'
))

# 设置文本框的位置，左上角坐标 (0.05, 0.95)
plt.gcf().text(0.15, 0.85, textstr, fontsize=22, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

# 手动添加色条：从 plt.cm 获取颜色映射，设定范围为0-1之间
sm = plt.cm.ScalarMappable(cmap='Blues')
sm.set_array([])  # 手动设置空数据
cbar =plt.colorbar(sm)
cbar.ax.tick_params(labelsize=24)
plt.grid(True, linestyle='--', alpha=0.6)
plt.title('SM at ' + target_depths[depth] +' layer (Day 2)', fontsize=24)
plt.savefig('E:/SMC_retrieval/成图文件/'+ 'SM at ' + target_depths[depth] +' layer (Day 2)'+'.png', dpi=400, bbox_inches='tight')
plt.show()


#day3
X_train3 = X_train
X_test3 = X_test
Y_train3 = Y_train.iloc[:,2]
Y_test3 = Y_test.iloc[:,2]
    
rf.fit(X_train3, Y_train3)
Y_pred = rf.predict(X_test3)  
df_test_id['Predict_SM_'+ target_depths[depth] + '_3'] = Y_pred
 
rmse3 = np.sqrt(mean_squared_error(Y_test['SM_'+ target_depths[depth] + '_3'], Y_pred))
r3,_ = pearsonr(Y_test['SM_'+ target_depths[depth] + '_3'], Y_pred)
bias3 = sum(Y_pred-Y_test['SM_'+ target_depths[depth] + '_3'])/len(Y_test['SM_'+ target_depths[depth] + '_3'])
ubRMSE3 = (rmse3**2-bias3**2)**0.5
print(f'RMSE3: {rmse3}')
print(f'R Score 3: {r3}')
print('Bias3:' + str(bias3))
print('ubRMSE3:' + str(ubRMSE3))

plt.figure(figsize=(8, 6), dpi=400)

sns.kdeplot(x=Y_test['SM_'+ target_depths[depth] + '_1'], 
            y=Y_pred, 
            fill=True, cmap='Blues', thresh=0.05, levels=100)

# 添加 y=x 参考线
plt.plot([Y_test['SM_'+ target_depths[depth] + '_1'].min(), Y_test['SM_'+ target_depths[depth] + '_1'].max()], 
         [Y_test['SM_'+ target_depths[depth] + '_1'].min(), Y_test['SM_'+ target_depths[depth] + '_1'].max()], 
         'k--', label='y=x')

# 设置轴标签
plt.xlabel('Observed SM (cm$^3$/cm$^3$)', fontsize=24)
plt.ylabel('Predicted SM (cm$^3$/cm$^3$)', fontsize=24)

# 设置刻度和范围
plt.xticks(np.arange(0, 0.5, 0.1), fontsize=24)
plt.yticks(np.arange(0, 0.5, 0.1), fontsize=24)
plt.xlim(0, 0.5)
plt.ylim(0, 0.5)

textstr = '\n'.join((
    f'N = {N}',
    f'R = {r3:.3f}',
    f'RMSE = {rmse3:.3f}',
    f'Bias = {bias3:.3f}',
    f'ubRMSE = {ubRMSE3:.3f}'
))

# 设置文本框的位置，左上角坐标 (0.05, 0.95)
plt.gcf().text(0.15, 0.85, textstr, fontsize=22, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

# 手动添加色条：从 plt.cm 获取颜色映射，设定范围为0-1之间
sm = plt.cm.ScalarMappable(cmap='Blues')
sm.set_array([])  # 手动设置空数据
cbar =plt.colorbar(sm)
cbar.ax.tick_params(labelsize=24)
plt.grid(True, linestyle='--', alpha=0.6)
plt.title('SM at ' + target_depths[depth] +' layer (Day 3)', fontsize=24)
plt.savefig('E:/SMC_retrieval/成图文件/'+ 'SM at ' + target_depths[depth] +' layer (Day 3)'+'.png', dpi=400, bbox_inches='tight')
plt.show()


