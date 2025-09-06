import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 读取Excel文件
file_path = "python_code//C(1)_total.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1")

# 提取需要拟合的两组数据
# 这里以"斜率(a)_归一化"和"BMI增长速率_归一化"为例
x_data = df["孕妇BMI"].values
x_data = df["检测孕周"].values

y_data = df["Y染色体浓度"].values


# 定义非线性拟合函数（这里以二次函数为例）
def nonlinear_func(x, a, b, c):
    return a * x**2 + b * x + c


# 执行非线性拟合
popt, pcov = curve_fit(nonlinear_func, x_data, y_data)

# 提取拟合参数
a_fit, b_fit, c_fit = popt

# 生成拟合曲线
x_fit = np.linspace(min(x_data), max(x_data), 100)
y_fit = nonlinear_func(x_fit, a_fit, b_fit, c_fit)

# 计算R²
residuals = y_data - nonlinear_func(x_data, *popt)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
r_squared = 1 - (ss_res / ss_tot)

# 绘制结果
plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, label="原始数据", alpha=0.6)
plt.plot(
    x_fit,
    y_fit,
    "r-",
    label=f"拟合曲线: y = {a_fit:.4f}x² + {b_fit:.4f}x + {c_fit:.4f}",
)
plt.xlabel("斜率(a)_归一化")
plt.ylabel("BMI增长速率_归一化")
plt.title(f"非线性拟合结果 (R² = {r_squared:.4f})")
plt.legend()
plt.grid(True)
plt.show()

# 打印拟合参数
print(f"拟合参数: a = {a_fit:.6f}, b = {b_fit:.6f}, c = {c_fit:.6f}")
print(f"R² = {r_squared:.6f}")
