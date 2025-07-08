# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 14:12:51 2024

@author: ruizhen_wang
"""

import pandas as pd
import numpy as np
import os
import rasterio
from xgboost import XGBRegressor
from datetime import datetime
import matplotlib.pyplot as plt
import shap  # 新增SHAP库
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Times New Roman'
df = pd.read_excel('D:/SMC_retrieval/predictors_dataset_lag5/SMC_database_lag5_4.28.xlsx')
#the prediction soil layer depth, selecting from below:
#05, 515, 1530, 3060
depth = '3060'



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


#地表覆盖研磨
with rasterio.open('D:/SMC_retrieval/landcover/CLCD_v01_2019.tif') as src:
    landcover = pd.DataFrame(src.read(1).flatten(), columns=['landcover'])


####data filtering (different depths)
X_train = df.iloc[:,:-3]
Y_train = df.iloc[:,-3:]
#predictor_dataset_30m.columns=X_train.columns
rf = XGBRegressor(n_estimators=100, random_state=42, learning_rate=0.07) 
def save_shap_importance(shap_values, feature_names, output_file, X_train):
    """生成并保存beeswarm图"""
    plt.figure(figsize=(12, 8))
    
    shap.summary_plot(shap_values, 
                     X_train,
                     plot_type="dot",  # beeswarm图类型
                     max_display=15,   # 显示前20个特征
                     color=plt.get_cmap("viridis"),  # 使用Viridis色系
                     alpha=0.7,        # 点透明度
                     show=False)
    
    # 优化样式
    plt.title(f"SHAP Beeswarm Plot - {output_file.split('_')[0]}", fontsize=20, pad=20)
    plt.xlabel("SHAP Value (Impact on Prediction)", fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=20)
    plt.gcf().set_facecolor('white')  # 设置背景为白色
    
    # 保存图像
    plt.savefig(output_file.replace('.csv', '_beeswarm.png'), 
               dpi=300, 
               bbox_inches='tight')
    plt.close()

# 修改analyze_with_shap函数
def analyze_with_shap(X_train, Y_train, depth_label, day_suffix):
    model = XGBRegressor(n_estimators=100, random_state=42, learning_rate=0.07)
    model.fit(X_train, Y_train)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    
    # 保存beeswarm图
    save_shap_importance(shap_values, X_train.columns, 
                        f"{depth_label}_shap_{day_suffix}", X_train)
    
    return model
 
#day1
X_train1 = X_train.drop(columns=['wind_2', 'wind_3', 'pre_2', 'pre_3', 'temp_2', 'temp_3'])
Y_train1 = Y_train.iloc[:, 0]
rf1 = analyze_with_shap(X_train1, Y_train1, target_depths[depth], "day1")

# Day2
X_train2 = X_train.drop(columns=['wind_3', 'pre_3', 'temp_3'])
Y_train2 = Y_train.iloc[:, 1]
rf2 = analyze_with_shap(X_train2, Y_train2, target_depths[depth], "day2")

# Day3
X_train3 = X_train
Y_train3 = Y_train.iloc[:, 2]
rf3 = analyze_with_shap(X_train3, Y_train3, target_depths[depth], "day3")

