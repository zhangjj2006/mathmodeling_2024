import pandas as pd
from scipy import stats

# 读取 Excel 文件
excel_file = pd.ExcelFile("文件//版本3_归一化_Week_Y_BMI_004_result.xlsx")

# 获取指定工作表中的数据
df = excel_file.parse("编码数据_版本3_归一化")

# 假设两组连续型数据分别在 'column1' 和 'column2' 列
data1 = df["斜率(a)"]
data2 = df["BMI增长速率"]

# 计算斯皮尔曼等级相关系数和 p 值
corr, p_value = stats.spearmanr(data1, data2)
print(f"斯皮尔曼等级相关系数: {corr}")
print(f"p 值: {p_value}")
if p_value < 0.05:
    print("两组数据的相关是显著的。")
else:
    print("两组数据的相关不显著。")
