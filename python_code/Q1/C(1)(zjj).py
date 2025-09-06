import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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
    # df_filtered.to_excel('./python_code/Week_Y_filtered.xlsx', index=False)
    regression_results = []

    for patient_id in df_filtered["孕妇代码"].unique():
        patient_data = df_filtered[df_filtered["孕妇代码"] == patient_id]
        x = patient_data["检测孕周_天数"]
        y = patient_data["Y染色体浓度"]

        # 进行线性回归
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        r_squared = r_value**2
        x_values = 0.04 - intercept / slope if slope != 0 else np.nan
        if slope > 0:
            # if slope > 0 :
            # 获取第一个满足条件的记录
            first_record = patient_data.iloc[0]

            # 计算BMI增长速率相关指标
            # 获取第一个和最后一个记录来计算BMI变化
            first_bmi_record = patient_data.iloc[0]
            last_bmi_record = patient_data.iloc[-1]

            # 计算BMI变化和时间间隔
            bmi_change = last_bmi_record["孕妇BMI"] - first_bmi_record["孕妇BMI"]
            days_change = (
                last_bmi_record["检测孕周_天数"] - first_bmi_record["检测孕周_天数"]
            )

            # 避免除零错误
            if days_change != 0:
                # BMI增长速率 (ΔBMI/Δ天数)
                bmi_growth_rate = bmi_change / days_change
                # 标准化BMI增长速率 (ΔBMI/Δ天数)/平均BMI
                normalized_bmi_growth_rate = (
                    bmi_growth_rate / first_bmi_record["孕妇平均BMI"]
                )
            else:
                bmi_growth_rate = 0
                normalized_bmi_growth_rate = 0
            # if bmi_growth_rate > 0:
            # 添加回归分析结果和BMI增长速率指标
            result_row = first_record.copy()
            result_row["斜率(a)"] = slope
            result_row["截距(b)"] = intercept
            result_row["R方"] = r_squared
            result_row["BMI增长速率"] = bmi_growth_rate
            result_row["标准化BMI增长速率"] = normalized_bmi_growth_rate
            result_row["BMI值(y = 0.04)"] = x_values

            regression_results.append(result_row)

    df_result = pd.DataFrame(regression_results)

    # 保存结果到Excel文件
    df_result.to_excel("文件//complete_Week_Y_BMI_0.04_result.xlsx", index=False)


if __name__ == "__main__":
    best_bmi_Y()
