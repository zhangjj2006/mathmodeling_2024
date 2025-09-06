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
    """
    加载附件数据并进行预处理
    1. 提取男胎数据（Y染色体浓度非空）
    2. 选择特征：BMI、身高、体重、年龄
    3. 处理缺失值：均值插补
    4. 使用Z-score标准化特征
    """
    df = pd.read_excel('./python_code/附件.xlsx',sheet_name=0)

    def weeks_to_days(weeks_str):
        match = re.match(r'(\d+)[wW](?:\+(\d+))?', str(weeks_str), re.IGNORECASE)
        if match:
            weeks = int(match.group(1))
            days = int(match.group(2)) if match.group(2) else 0
            return weeks * 7 + days
        else:
            print(f"无法解析孕周格式: {weeks_str}")
            return None
        
    df['检测孕周_天数'] = df['检测孕周'].apply(weeks_to_days)
    # 提取男胎数据（Y染色体浓度非空）
    male_data = df[df['Y染色体浓度'].notna()].copy()
    print(f"男胎数据量: {len(male_data)}")
    
    # 选择特征：BMI、身高、体重、年龄
    features = male_data[['孕妇代码', '年龄', '身高', '体重', '孕妇BMI', 'Y染色体浓度', 'Y染色体的Z值', '检测孕周_天数']].copy()
    
    # 处理缺失值：均值插补
    for col in features.columns:
        if col != '孕妇代码':  # 跳过非数值列
            if features[col].isnull().sum() > 0:
                mean_val = features[col].mean()
                features[col].fillna(mean_val, inplace=True)
                print(f"列 '{col}' 有 {features[col].isnull().sum()} 个缺失值，已用均值 {mean_val:.2f} 填充")
    
    # 检查并处理可能的非数值数据
    for col in features.columns:
        if col != '孕妇代码':  # 跳过非数值列
            # 尝试转换为数值类型
            features[col] = pd.to_numeric(features[col], errors='coerce')
            # 再次检查并填充可能出现的NaN
            if features[col].isnull().sum() > 0:
                mean_val = features[col].mean()
                features[col].fillna(mean_val, inplace=True)
                print(f"列 '{col}' 转换后有缺失值，已用均值 {mean_val:.2f} 填充")
    
    # 使用Z-score标准化特征（排除孕妇代码、Y染色体浓度和Y染色体的Z值）
    scaler = StandardScaler()  # StandardScaler默认使用Z-score标准化
    feature_cols = ['年龄', '身高', '体重', '孕妇BMI', '检测孕周_天数']
    features_scaled = scaler.fit_transform(features[feature_cols])
    features_scaled_df = pd.DataFrame(features_scaled, 
                                     columns=[f'{col}_标准化' for col in feature_cols])
    
    # 只保留需要的列：孕妇代码、原始特征、标准化特征、Y染色体浓度和Y染色体的Z值
    final_data = pd.concat([
        features[['孕妇代码', 'Y染色体浓度', 'Y染色体的Z值']], 
        features[feature_cols], 
        features_scaled_df
    ], axis=1)
    
    # 根据Y染色体浓度判断能否检测
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


    # 保存处理后的数据
    try:
        final_data.to_excel('./python_code/Q3/Q3数据预处理.xlsx', index=False)
        print("预处理后的数据已保存至: ./python_code/Q3/Q3数据预处理.xlsx")
    except Exception as e:
        print(f"保存文件时出错: {e}")
        return None
    
    # 打印数据基本信息
    print("\n=== 数据预处理完成 ===")
    print(f"处理后的数据形状: {final_data.shape}")
    print("\n特征统计信息:")
    print(final_data.describe())
    
    # 返回处理后的数据
    return final_data

# 运行数据预处理
processed_data = load_and_preprocess_data()

# # 如果成功处理数据，显示前几行
# if processed_data is not None:
#     print("\n处理后的数据前5行:")
#     print(processed_data.head())