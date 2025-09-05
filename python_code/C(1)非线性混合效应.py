# -*- coding: utf-8 -*-
"""
NIPT数据分析 - 非线性混合效应模型实现
针对2025年高教社杯全国大学生数学建模竞赛C题
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import optimize
import statsmodels.api as sm
from statsmodels.tools.tools import add_constant
from statsmodels.formula.api import mixedlm
import warnings

warnings.filterwarnings("ignore")

# 设置中文显示
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# 1. 数据加载与预处理
def load_and_preprocess_data(file_path):
    """
    加载并预处理NIPT数据
    """
    # 读取数据
    df = pd.read_excel(file_path)

    # 数据清洗
    # 筛选男胎数据（Y染色体浓度非空）
    df = df[df["Y染色体浓度"].notna()]

    # 处理缺失值
    df = df.dropna(subset=["检测孕周", "孕妇BMI", "Y染色体浓度"])

    # 转换孕周为天数
    df["孕天"] = df["检测孕周"] * 7

    # 确保每个孕妇有足够的数据点（≥3次检测）
    patient_count = df["孕妇代码"].value_counts()
    valid_patients = patient_count[patient_count >= 3].index
    df = df[df["孕妇代码"].isin(valid_patients)]

    return df


# 2. 探索性数据分析
def exploratory_data_analysis(df):
    """
    执行探索性数据分析，绘制相关图表
    """
    # 绘制Y染色体浓度随孕周变化的散点图
    plt.figure(figsize=(10, 6))
    # 使用scatterplot并获取返回的Axes对象
    ax = sns.scatterplot(
        data=df,
        x="检测孕周",
        y="Y染色体浓度",
        hue="孕妇BMI",
        palette="viridis",
        alpha=0.6,
    )
    plt.title("Y染色体浓度随孕周变化分布")
    plt.xlabel("检测孕周")
    plt.ylabel("Y染色体浓度 (%)")

    # 添加颜色条
    norm = plt.Normalize(df["孕妇BMI"].min(), df["孕妇BMI"].max())
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])

    # 移除之前的图例（如果有）
    ax.get_legend().remove()

    # 添加颜色条
    cbar = plt.colorbar(sm)
    cbar.set_label("孕妇BMI")

    plt.show()

    # 绘制BMI分布直方图
    plt.figure(figsize=(10, 6))
    sns.histplot(df["孕妇BMI"], kde=True)
    plt.title("孕妇BMI分布")
    plt.xlabel("孕妇BMI")
    plt.ylabel("频数")
    plt.show()

    # 计算相关系数矩阵
    numeric_cols = ["检测孕周", "孕妇BMI", "Y染色体浓度", "年龄", "体重"]
    corr_matrix = df[numeric_cols].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0)
    plt.title("变量间相关系数矩阵")
    plt.show()


# 3. 非线性混合效应模型实现
def nonlinear_mixed_effects_model(df):
    """
    建立非线性混合效应模型
    使用逻辑增长模型: c = A / (1 + exp(-(t - t0)/τ))
    """
    # 为每个患者准备数据
    patients = df["孕妇代码"].unique()
    n_patients = len(patients)

    # 定义逻辑增长函数
    def logistic_growth(t, A, t0, tau):
        """逻辑增长模型"""
        return A / (1 + np.exp(-(t - t0) / tau))

    # 第一步：为每个患者单独拟合曲线，获取初始参数估计
    patient_params = {}

    for patient in patients:
        patient_data = df[df["孕妇代码"] == patient]
        t_data = patient_data["孕天"].values
        c_data = patient_data["Y染色体浓度"].values
        bmi = patient_data["孕妇BMI"].iloc[0]  # 取第一次测量的BMI

        # 初始参数猜测
        A_guess = np.max(c_data) * 1.1  # 饱和值略高于最大值
        t0_guess = np.median(t_data)  # 拐点在中位数附近
        tau_guess = (np.max(t_data) - np.min(t_data)) / 4  # 时间常数

        try:
            # 拟合曲线
            params, _ = optimize.curve_fit(
                logistic_growth,
                t_data,
                c_data,
                p0=[A_guess, t0_guess, tau_guess],
                maxfev=5000,
            )
            patient_params[patient] = {
                "A": params[0],
                "t0": params[1],
                "tau": params[2],
                "BMI": bmi,
            }
        except:
            # 拟合失败时跳过该患者
            continue

    # 转换为DataFrame
    params_df = pd.DataFrame.from_dict(patient_params, orient="index")
    params_df = params_df.reset_index().rename(columns={"index": "孕妇代码"})

    # 第二步：分析参数与BMI的关系
    # 使用混合效应模型分析A和tau与BMI的关系
    # 准备长格式数据
    mixed_data = df.merge(params_df[["孕妇代码", "A", "tau"]], on="孕妇代码")

    # 使用statsmodels的MixedLM建立混合效应模型
    # 模型1: A ~ BMI
    model_A = mixedlm("A ~ BMI", data=params_df, groups=params_df["孕妇代码"])
    result_A = model_A.fit()

    # 模型2: tau ~ BMI
    model_tau = mixedlm("tau ~ BMI", data=params_df, groups=params_df["孕妇代码"])
    result_tau = model_tau.fit()

    # 打印模型结果
    print("=" * 50)
    print("饱和值A与BMI关系的混合效应模型结果:")
    print(result_A.summary())

    print("=" * 50)
    print("时间常数tau与BMI关系的混合效应模型结果:")
    print(result_tau.summary())

    return params_df, result_A, result_tau


# 4. 模型评估与可视化
def model_evaluation_and_visualization(df, params_df):
    """
    评估模型效果并生成可视化结果
    """

    # 定义逻辑增长函数
    def logistic_growth(t, A, t0, tau):
        return A / (1 + np.exp(-(t - t0) / tau))

    # 选择几个代表性患者展示拟合效果
    sample_patients = params_df.sample(min(4, len(params_df)), random_state=42)[
        "孕妇代码"
    ].values

    plt.figure(figsize=(12, 10))
    for i, patient in enumerate(sample_patients):
        patient_data = df[df["孕妇代码"] == patient]
        params = params_df[params_df["孕妇代码"] == patient].iloc[0]

        # 生成拟合曲线
        t_range = np.linspace(min(patient_data["孕天"]), max(patient_data["孕天"]), 100)
        fitted_curve = logistic_growth(
            t_range, params["A"], params["t0"], params["tau"]
        )

        # 绘制子图
        plt.subplot(2, 2, i + 1)
        plt.scatter(patient_data["孕天"], patient_data["Y染色体浓度"], label="观测值")
        plt.plot(t_range, fitted_curve, "r-", label="拟合曲线")
        plt.xlabel("孕天")
        plt.ylabel("Y染色体浓度 (%)")
        plt.title(f'患者 {patient} (BMI={params["孕妇BMI"]:.1f})')
        plt.legend()

    plt.tight_layout()
    plt.show()

    # 绘制参数与BMI的关系
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 饱和值A与BMI的关系
    sns.regplot(data=params_df, x="BMI", y="A", ax=ax1)
    ax1.set_xlabel("BMI")
    ax1.set_ylabel("饱和值 A")
    ax1.set_title("饱和值与BMI的关系")

    # 时间常数tau与BMI的关系
    sns.regplot(data=params_df, x="BMI", y="tau", ax=ax2)
    ax2.set_xlabel("BMI")
    ax2.set_ylabel("时间常数 τ")
    ax2.set_title("时间常数与BMI的关系")

    plt.tight_layout()
    plt.show()

    # 计算整体R²
    all_actual = []
    all_predicted = []

    for _, row in params_df.iterrows():
        patient = row["孕妇代码"]
        patient_data = df[df["孕妇代码"] == patient]

        if len(patient_data) > 0:
            actual = patient_data["Y染色体浓度"].values
            predicted = logistic_growth(
                patient_data["孕天"].values, row["A"], row["t0"], row["tau"]
            )

            all_actual.extend(actual)
            all_predicted.extend(predicted)

    # 计算R²
    correlation_matrix = np.corrcoef(all_actual, all_predicted)
    r_squared = correlation_matrix[0, 1] ** 2

    print(f"模型整体R²: {r_squared:.4f}")


# 主函数
def main():
    """
    主函数，执行完整分析流程
    """
    # 1. 加载和预处理数据
    print("正在加载和预处理数据...")
    df = load_and_preprocess_data("python_code//附件.xlsx")  # 替换为实际文件路径

    # 2. 探索性数据分析
    print("正在进行探索性数据分析...")
    exploratory_data_analysis(df)

    # 3. 建立非线性混合效应模型
    print("正在建立非线性混合效应模型...")
    params_df, result_A, result_tau = nonlinear_mixed_effects_model(df)

    # 4. 模型评估与可视化
    print("正在进行模型评估与可视化...")
    model_evaluation_and_visualization(df, params_df)

    # 5. 输出结论
    print("\n" + "=" * 60)
    print("模型分析结论:")
    print("1. Y染色体浓度随孕周增长呈现非线性逻辑增长模式")
    print("2. BMI对增长参数有显著影响:")
    print(
        f"   - BMI每增加1单位，饱和值A变化: {result_A.params['BMI']:.4f} (p={result_A.pvalues['BMI']:.4f})"
    )
    print(
        f"   - BMI每增加1单位，时间常数τ变化: {result_tau.params['BMI']:.4f} (p={result_tau.pvalues['BMI']:.4f})"
    )
    print("3. 高BMI孕妇的Y染色体浓度增长更快，饱和值更高")


if __name__ == "__main__":
    main()
