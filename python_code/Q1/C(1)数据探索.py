import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 正确显示负号

# 读取处理好的数据
df = pd.read_excel("./python_code/Q1/C(1)预处理.xlsx")

# 数据清洗：移除无穷大或缺失值
df_clean = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['标准化BMI增长速率', '斜率(a)'])

# 提取需要的数据
x = df_clean['标准化BMI增长速率']  # 标准化BMI增长速率
y = df_clean['斜率(a)']           # Y染色体增长速率

# 创建画布
plt.figure(figsize=(12, 5))

# 第一张图：基础散点图
plt.subplot(1, 2, 1)
plt.scatter(x, y, alpha=0.6, color='steelblue')
plt.xlabel('标准化BMI增长速率')
plt.ylabel('Y染色体增长速率')
plt.title('Y染色体增长速率 vs 标准化BMI增长速率')
plt.grid(True, linestyle='--', alpha=0.7)

# 第二张图：带拟合直线的散点图
plt.subplot(1, 2, 2)
plt.scatter(x, y, alpha=0.6, color='steelblue', label='数据点')

# 计算线性回归
slope, intercept, r_value, p_value, std_err = linregress(x, y)
line_x = np.linspace(min(x), max(x), 100)
line_y = slope * line_x + intercept

# 绘制拟合直线
plt.plot(line_x, line_y, color='red', linewidth=2, 
         label=f'拟合直线: y = {slope:.4f}x + {intercept:.4f}\nR² = {r_value**2:.4f}')

plt.xlabel('标准化BMI增长速率')
plt.ylabel('Y染色体增长速率')
plt.title('带拟合直线的散点图')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# 调整布局并显示
plt.tight_layout()
plt.savefig('./python_code/Q1/散点图分析.png', dpi=300, bbox_inches='tight')
plt.show()

# 打印回归统计信息
print(f"回归方程: y = {slope:.6f}x + {intercept:.6f}")
print(f"R²值: {r_value**2:.6f}")
print(f"p值: {p_value:.6f}")