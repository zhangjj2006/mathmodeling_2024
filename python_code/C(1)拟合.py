import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# 定义指数函数
def exponential_func(x, a, b):
    return a * np.exp(b * x)


# 读取 Excel 文件（请替换为你的文件路径）
excel_file = pd.ExcelFile("文件//Week_Y_BMI_Line_result.xlsx")

# 获取指定工作表中的数据（请替换为你的工作表名称）
df = excel_file.parse("Sheet1")

# 假设两组数据分别在 'x_column' 和 'y_column' 列（请替换为实际列名）
x_data = df["检测孕周_天数"].values
y_data = df["Y染色体浓度"].values

# 进行非线性拟合
popt, _ = curve_fit(exponential_func, x_data, y_data)

# 提取拟合得到的参数
a_fit, b_fit = popt

# 计算拟合值
y_fit = exponential_func(x_data, a_fit, b_fit)

# 计算残差平方和
residuals = y_data - y_fit
ss_res = np.sum(residuals**2)

# 计算总离差平方和
ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)

# 计算决定系数 R^2
r_squared = 1 - (ss_res / ss_tot)

print(f"拟合得到的参数 a: {a_fit:.4f}, b: {b_fit:.4f}")
print(f"残差平方和: {ss_res:.4f}")
print(f"决定系数 R^2: {r_squared:.4f}")

# 绘制原始数据和拟合曲线
plt.scatter(x_data, y_data, label="原始数据")
plt.plot(x_data, y_fit, "r-", label="拟合曲线")
plt.xlabel("x")
plt.ylabel("y")
plt.title("非线性拟合结果")
plt.legend()
plt.show()
