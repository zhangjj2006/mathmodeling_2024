import pandas as pd
from scipy import stats

# 读取文件
excel_file = pd.ExcelFile("文件//版本3_归一化_Week_Y_BMI_004_result.xlsx")
# 获取指定工作表中的数据
df = excel_file.parse("编码数据_版本3_归一化")

# 选取两组配对数据
data1 = df["斜率(a)_归一化"]
data2 = df["BMI增长速率_归一化"]

# 进行配对样本 t 检验
t_statistic, p_value = stats.ttest_rel(data1, data2)

print(f"t 统计量: {t_statistic}")
print(f"p 值: {p_value}")

if p_value < 0.05:
    print("两组数据的均值存在显著差异。")
else:
    print("两组数据的均值不存在显著差异。")
