import pandas as pd
import numpy as np  # 导入 numpy 库并使用 np 作为别名


def convert_gestational_week(week_str):
    if pd.isna(week_str):
        return np.nan
    try:
        week_str = week_str.lower()  # 将字符串转换为小写
        if "w" in week_str:
            parts = week_str.split("w")
            weeks = float(parts[0])
            if "+" in parts[1]:
                days = float(parts[1].split("+")[1])
                return weeks + days / 7
            return weeks
        return float(week_str)
    except:
        return np.nan


# 读取数据
df = pd.read_excel("python_code//附件.xlsx", sheet_name="男胎检测数据")

# 将检测孕周转换为数值格式
df["检测孕周数值"] = df["检测孕周"].apply(convert_gestational_week)

# 提取需要的列数据
columns = ["Y染色体浓度", "检测孕周数值", "孕妇BMI"]
data = df[columns]

# 计算相关系数矩阵
correlation_matrix = data.corr(method="spearman")

print(correlation_matrix)
