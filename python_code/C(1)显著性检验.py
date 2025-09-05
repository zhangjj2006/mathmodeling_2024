import pandas as pd
from scipy import stats

# 读取Excel文件中的两组数据
# 请替换为你的Excel文件路径
file_path = "文件//版本3_归一化_Week_Y_BMI_004_result.xlsx"
# 请替换为实际的工作表名称
sheet_name = "编码数据_版本3_归一化"
# 请替换为实际的列名（第一组数据：初始BMI；第二组数据：Y染色体浓度日增长率k_i）
col1 = "标准化BMI增长速率"
col2 = "斜率(a)"

# 读取数据并处理缺失值
try:
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    # 提取两组数据并删除缺失值
    data1 = df[col1].dropna()
    data2 = df[col2].dropna()

    # 确保两组数据长度一致（取交集）
    common_indices = data1.index.intersection(data2.index)
    data1 = data1.loc[common_indices]
    data2 = data2.loc[common_indices]

    # 计算Pearson相关系数和p值
    r, p_value = stats.pearsonr(data1, data2)

    # 输出结果
    print(f"Pearson相关系数 r = {r:.4f}")
    print(f"p值 = {p_value:.6f}")

    # 判断显著性类型
    if p_value < 0.01:
        if r > 0:
            print("显著性类型：显著正相关（p < 0.01）")
        else:
            print("显著性类型：显著负相关（p < 0.01）")
    elif p_value < 0.05:
        if r > 0:
            print("显著性类型：显著正相关（p < 0.05）")
        else:
            print("显著性类型：显著负相关（p < 0.05）")
    else:
        print("显著性类型：无显著线性相关（p ≥ 0.05）")

except FileNotFoundError:
    print(f"错误：找不到文件 {file_path}")
except KeyError as e:
    print(f"错误：数据中不存在列 {e}，请检查列名是否正确")
except Exception as e:
    print(f"计算过程中发生错误：{str(e)}")
