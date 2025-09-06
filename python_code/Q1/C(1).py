import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_excel("python_code//附件.xlsx")

counts = df["孕妇代码"].value_counts()

valid_ids = counts[counts > 3].index

filtered_df = df[df["孕妇代码"].isin(valid_ids)]
filtered_df = filtered_df[['孕妇代码', '检测孕周','孕妇BMI', 'Y染色体浓度']]
filtered_df.to_excel("python_code//附件_剔除后数据.xlsx", index=False)

def calculate_regression_coefficients(group):
    """
    计算线性回归的斜率和截距
    """

    x = group['孕周']
    y = group['头围']
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    return pd.Series({
        '斜率': slope,
        '截距': intercept
    })

regression_coefficients = filtered_df.groupby('孕妇代码').apply(calculate_regression_coefficients)

result_df = filtered_df.merge(regression_coefficients, on='孕妇代码')

result_df.to_excel("python_code//附件_剔除后数据_含回归系数.xlsx", index=False)