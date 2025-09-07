import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress
import re


def best_bmi_Y():  # 将y染色体达标时候的孕妇bmi挑出来
    def weeks_to_days(weeks_str):
        match = re.match(r"(\d+)[wW](?:\+(\d+))?", str(weeks_str), re.IGNORECASE)
        if match:
            weeks = int(match.group(1))
            days = int(match.group(2)) if match.group(2) else 0
            return weeks * 7 + days
        else:
            print(f"无法解析孕周格式: {weeks_str}")
            return None

    df = pd.read_excel("./python_code/附件.xlsx", sheet_name=0)
    df = df[["孕妇代码", "检测孕周", "孕妇BMI", "Y染色体浓度"]]
    bmi_avg = df.groupby("孕妇代码")["孕妇BMI"].mean()
    df["孕妇平均BMI"] = df["孕妇代码"].map(bmi_avg)
    df["检测孕周_天数"] = df["检测孕周"].apply(weeks_to_days)

    detection_counts = df["孕妇代码"].value_counts()
    valid_ids = detection_counts[detection_counts >= 4].index
    df_filtered = df[df["孕妇代码"].isin(valid_ids)]
    
    regression_results = []

    for patient_id in df_filtered["孕妇代码"].unique():
        patient_data = df_filtered[df_filtered["孕妇代码"] == patient_id]
        x = patient_data["检测孕周_天数"]
        y = patient_data["Y染色体浓度"]

        # 进行Y染色体浓度的线性回归
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        r_squared = r_value**2
        x_values = 0.04 - intercept / slope if slope != 0 else np.nan
        
        # if slope > 0:
            # 对BMI进行线性回归
        x_bmi = patient_data["检测孕周_天数"]
        y_bmi = patient_data["孕妇BMI"]
        slope_bmi, intercept_bmi, r_value_bmi, p_value_bmi, std_err_bmi = linregress(x_bmi, y_bmi)
            
        # 获取第一个记录
        first_record = patient_data.iloc[0]
        
        # 计算BMI增长速率 (使用线性拟合的斜率)
        bmi_growth_rate = slope_bmi
        # 标准化BMI增长速率 (斜率/平均BMI)
        normalized_bmi_growth_rate = slope_bmi / first_record["孕妇平均BMI"]
            
        # 添加回归分析结果和BMI增长速率指标
        result_row = first_record.copy()
        result_row["斜率(a)"] = slope
        result_row["截距(b)"] = intercept
        result_row["R方"] = r_squared
        result_row["BMI增长速率"] = slope_bmi
        result_row["标准化BMI增长速率"] = normalized_bmi_growth_rate

        regression_results.append(result_row)

    df_result = pd.DataFrame(regression_results)

    # 保存结果到Excel文件
    df_result.to_excel("./python_code/Q1/C(1)预处理.xlsx", index=False)


if __name__ == "__main__":
    best_bmi_Y()