import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from lifelines import KaplanMeierFitter
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import re

import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    df = pd.read_excel('./python_code/附件.xlsx',sheet_name=0)

    def weeks_to_days(weeks_str):
        match = re.match(r'(\d+)[wW](?:\+(\d+))?', str(weeks_str), re.IGNORECASE)
        if match:
            weeks = int(match.group(1))
            days = int(match.group(2)) if match.group(2) else 0
            return weeks * 7 + days
        else:
            print(f"无法解析: {weeks_str}")
            return None
        
    df['检测孕周_天数'] = df['检测孕周'].apply(weeks_to_days)
    male_data = df[df['Y染色体浓度'].notna()].copy()
    print(f"男胎数据量: {len(male_data)}")
    
    features = male_data[['孕妇代码', '年龄', '身高', '体重', '孕妇BMI', 'Y染色体浓度', 'Y染色体的Z值', '检测孕周_天数']].copy()
    
    for col in features.columns:
        if col != '孕妇代码': 
            if features[col].isnull().sum() > 0:
                mean_val = features[col].mean()
                features[col].fillna(mean_val, inplace=True)
                print(f"列 '{col}' 有 {features[col].isnull().sum()} 个缺失值，已用均值 {mean_val:.2f} 填充")
    
    for col in features.columns:
        if col != '孕妇代码': 
            features[col] = pd.to_numeric(features[col], errors='coerce')
            if features[col].isnull().sum() > 0:
                mean_val = features[col].mean()
                features[col].fillna(mean_val, inplace=True)
                print(f"列 '{col}' 转换后有缺失值，已用均值 {mean_val:.2f} 填充")
    
    # 使用Z-score标准化特征
    scaler = StandardScaler()  
    feature_cols = ['年龄', '身高', '体重', '孕妇BMI', '检测孕周_天数']
    features_scaled = scaler.fit_transform(features[feature_cols])
    features_scaled_df = pd.DataFrame(features_scaled, 
                                     columns=[f'{col}_标准化' for col in feature_cols])
    
    final_data = pd.concat([
        features[['孕妇代码', 'Y染色体浓度', 'Y染色体的Z值']], 
        features[feature_cols], 
        features_scaled_df
    ], axis=1)
    
    final_data['能否检测'] = final_data['Y染色体浓度'].apply(lambda x: 1 if x > 0.04 else 0)

    can_code = []
    df_codes = df['孕妇代码'].unique()
    for code in df_codes:
        data = df[df['孕妇代码'] == code].sort_values(by='检测孕周_天数')
        first_ok_idx = None

        for idx, row in data.iterrows():
            if row['Y染色体浓度'] >= 0.04:
                first_ok_idx = idx
                break
        
        if first_ok_idx is not None:
            later_data = data.loc[first_ok_idx:]
            if (later_data['Y染色体浓度'] < 0.04).any():
                continue
            else:
                can_code.append(code)

    final_data = final_data[final_data['孕妇代码'].isin(can_code)]


    try:
        final_data.to_excel('./python_code/Q3/Q3数据预处理.xlsx', index=False)
        print("预处理后的数据已保存至: ./python_code/Q3/Q3数据预处理.xlsx")
    except Exception as e:
        print(f"保存文件出错: {e}")
        return None
    
    return final_data

processed_data = load_and_preprocess_data()