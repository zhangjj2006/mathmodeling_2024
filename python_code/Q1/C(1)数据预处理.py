import pandas as pd
import numpy as np
from scipy import stats

# 读取Excel文件
df = pd.read_excel("python_code//附件.xlsx", sheet_name="男胎检测数据")


# 1. 数据清洗 - 提取相关变量
def clean_data(df):
    """
    数据清洗函数，提取相关变量并确保只保留男胎样本
    """
    # 选择需要的列
    columns_to_keep = [
        "孕妇代码",
        "年龄",
        "身高",
        "体重",
        "检测孕周",
        "孕妇BMI",
        "Y染色体浓度",
        "13号染色体的Z值",
        "18号染色体的Z值",
        "21号染色体的Z值",
        "X染色体的Z值",
        "胎儿是否健康",
        "怀孕次数",
        "生产次数",
    ]

    # 保留需要的列
    df_clean = df[columns_to_keep].copy()

    # 转换孕周为数值格式（例如："11w+6" → 11.857）
    def convert_gestational_week(week_str):
        if pd.isna(week_str):
            return np.nan
        try:
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

    df_clean["检测孕周"] = df_clean["检测孕周"].apply(convert_gestational_week)

    return df_clean


# 2. 处理缺失值
def handle_missing_values(df):
    """
    处理缺失值函数
    """
    # 检查缺失值
    missing_values = df.isnull().sum()
    print("缺失值统计:")
    print(missing_values[missing_values > 0])

    # 对于数值列，使用中位数填充缺失值
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(
                f"列 '{col}' 的 {df[col].isnull().sum()} 个缺失值已用中位数 {median_val:.4f} 填充"
            )

    return df


# 3. 处理多次检测
def handle_multiple_tests(df):
    """
    处理同一孕妇多次检测的函数
    对于同一孕妇，取Y染色体浓度的平均值
    """
    # 按孕妇代码分组，计算Y染色体浓度的平均值
    y_chromosome_avg = df.groupby("孕妇代码")["Y染色体浓度"].mean().reset_index()
    y_chromosome_avg.rename(columns={"Y染色体浓度": "Y染色体浓度_平均"}, inplace=True)

    # 合并回原数据框
    df = df.merge(y_chromosome_avg, on="孕妇代码", how="left")

    # 删除重复的孕妇记录，只保留每个孕妇的一条记录
    df_unique = df.drop_duplicates(subset="孕妇代码", keep="first").copy()
    df_unique["Y染色体浓度"] = df_unique["Y染色体浓度_平均"]
    df_unique.drop("Y染色体浓度_平均", axis=1, inplace=True)

    print(f"处理多次检测后，数据从 {len(df)} 行减少到 {len(df_unique)} 行")

    return df_unique


# 4. 异常值处理
def handle_outliers(df):
    """
    处理异常值函数，使用Z-score方法
    """
    # 选择需要检查异常值的数值列
    numeric_cols = ["年龄", "身高", "体重", "检测孕周", "孕妇BMI", "Y染色体浓度"]

    # 计算Z-score并识别异常值
    z_scores = np.abs(stats.zscore(df[numeric_cols]))
    outliers = (z_scores > 3).any(axis=1)

    print(f"检测到 {outliers.sum()} 个异常值")

    # 移除异常值
    df_clean = df[~outliers].copy()
    print(f"移除异常值后，数据从 {len(df)} 行减少到 {len(df_clean)} 行")

    return df_clean


# 执行数据预处理
def preprocess_data(df):
    """
    完整的数据预处理流程
    """
    print("开始数据预处理...")

    # 1. 数据清洗
    print("\n1. 数据清洗...")
    df_clean = clean_data(df)
    print(f"数据清洗后形状: {df_clean.shape}")

    # 2. 处理缺失值
    print("\n2. 处理缺失值...")
    df_no_missing = handle_missing_values(df_clean)
    print(f"处理缺失值后形状: {df_no_missing.shape}")

    # 3. 处理多次检测
    print("\n3. 处理多次检测...")
    df_unique = handle_multiple_tests(df_no_missing)
    print(f"处理多次检测后形状: {df_unique.shape}")

    # 4. 异常值处理
    print("\n4. 处理异常值...")
    df_final = handle_outliers(df_unique)
    print(f"处理异常值后形状: {df_final.shape}")

    print("\n数据预处理完成!")
    return df_final


# 执行预处理
processed_data = preprocess_data(df)

# 显示预处理后的数据前几行
print("\n预处理后的数据前5行:")
print(processed_data.head())

# 保存预处理后的数据
processed_data.to_excel("文件//预处理后的男胎检测数据.xlsx", index=False)
print("\n预处理后的数据已保存为 '预处理后的男胎检测数据.xlsx'")
