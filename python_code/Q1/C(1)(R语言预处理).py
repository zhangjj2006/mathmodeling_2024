import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import linregress
import re

def best_bmi_Y():  # 将y染色体达标时候的孕妇bmi挑出来
    def weeks_to_days(weeks_str):
        match = re.match(r'(\d+)[wW](?:\+(\d+))?', str(weeks_str), re.IGNORECASE)
        if match:
            weeks = int(match.group(1))
            days = int(match.group(2)) if match.group(2) else 0
            return weeks * 7 + days
        else:
            print(f"无法解析孕周格式: {weeks_str}")
            return None
    
    df = pd.read_excel('./python_code/附件.xlsx', sheet_name=0)
    df = df[['孕妇代码', '检测孕周', '孕妇BMI', 'Y染色体浓度']]
    bmi_avg = df.groupby('孕妇代码')['孕妇BMI'].mean()
    df['孕妇平均BMI'] = df['孕妇代码'].map(bmi_avg)
    df['检测孕周_天数'] = df['检测孕周'].apply(weeks_to_days)

    detection_counts = df['孕妇代码'].value_counts()
    valid_ids = detection_counts[detection_counts >= 4].index
    df_filtered = df[df['孕妇代码'].isin(valid_ids)]
    
    # 找出符合条件的孕妇ID
    valid_patient_ids = []
    regression_params = {}  # 存储每个孕妇的回归参数
    
    for patient_id in df_filtered['孕妇代码'].unique():
        patient_data = df_filtered[df_filtered['孕妇代码'] == patient_id]
        x = patient_data['检测孕周_天数']
        y = patient_data['Y染色体浓度']
        
        # 进行线性回归
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        r_squared = r_value ** 2
        
        # 只有斜率为正的孕妇才被认为是符合条件的
        # if slope > 0:
        valid_patient_ids.append(patient_id)
            # 保存回归参数供后续使用
        regression_params[patient_id] = {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_squared,
                'x_values': 0.04 - intercept / slope if slope != 0 else np.nan
            }
    
    # 筛选出所有符合条件的孕妇的所有记录
    df_valid_patients = df_filtered[df_filtered['孕妇代码'].isin(valid_patient_ids)].copy()
    
    # 为每个符合条件的孕妇添加回归分析结果和BMI增长速率指标
    regression_results = []
    
    for patient_id in valid_patient_ids:
        patient_data = df_valid_patients[df_valid_patients['孕妇代码'] == patient_id]
            
        # 为该孕妇的每一条记录添加回归分析结果和BMI增长速率指标
        for idx, record in patient_data.iterrows():
            result_row = record[['孕妇代码', '孕妇BMI', 'Y染色体浓度', '检测孕周_天数']]

            regression_results.append(result_row)
    
    df_result = pd.DataFrame(regression_results)
        
    # 保存结果到Excel文件
    df_result.to_excel('./python_code/Q1/C(1)R语言预处理.xlsx', index=False)

if __name__ == "__main__":
    best_bmi_Y()