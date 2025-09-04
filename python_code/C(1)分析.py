import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

# 创建散点图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Y染色体浓度 vs 孕周
scatter1 = ax1.scatter(df_clean["检测孕周数值"], df_clean["Y染色体浓度"], alpha=0.6)
ax1.set_xlabel("孕周")
ax1.set_ylabel("Y染色体浓度")
ax1.set_title("Y染色体浓度 vs 孕周")
ax1.grid(True, linestyle="--", alpha=0.7)

# Y染色体浓度 vs BMI
scatter2 = ax2.scatter(
    df_clean["孕妇BMI"], df_clean["Y染色体浓度"], alpha=0.6, c="orange"
)
ax2.set_xlabel("孕妇BMI")
ax2.set_ylabel("Y染色体浓度")
ax2.set_title("Y染色体浓度 vs BMI")
ax2.grid(True, linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()
