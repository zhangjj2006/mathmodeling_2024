import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import numpy as np

# 读取 Excel 文件，创建 excel_file 对象（请将文件名替换为实际文件名）
excel_file = pd.ExcelFile("文件//Week_Y_BMI_Line_result.xlsx")

# 获取对应工作表数据
df = excel_file.parse("Sheet1")

# 提取相关列
features = df[["斜率(a)", "截距(b)"]]

# 设置聚类的类别数
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features)

# 获取每个样本的聚类标签
df["cluster_label"] = kmeans.labels_

# 对每个聚类分别计算平均斜率、平均截距以及评估拟合程度
for i in range(n_clusters):
    cluster_df = df[df["cluster_label"] == i]
    combined_slope = cluster_df["斜率(a)"].mean()
    combined_intercept = cluster_df["截距(b)"].mean()

    # 计算平均 R^2
    avg_r_squared = cluster_df["R方"].mean()

    # 计算残差平方和 (SSR)
    x = cluster_df["检测孕周_天数"]
    y_true = cluster_df["Y染色体浓度"]
    y_pred = combined_slope * x + combined_intercept
    ssr = ((y_true - y_pred) ** 2).sum()

    # 计算调整后的 R^2
    n = len(cluster_df)
    p = 1
    adjusted_r2 = 1 - (1 - avg_r_squared) * (n - 1) / (n - p - 1)

    print(
        f"第 {i + 1} 类整合后的线性方程为: y = {combined_slope:.4f} * x + {combined_intercept:.4f}"
    )
    print(f"该类平均 R^2: {avg_r_squared:.4f}")
    print(f"该类残差平方和 (SSR): {ssr:.4f}")
    print(f"该类调整后的 R^2: {adjusted_r2:.4f}\n")

# 计算整体的拟合指标
y_pred_all = []
for _, row in df.iterrows():
    cluster_index = row["cluster_label"]
    relevant_cluster = df[df["cluster_label"] == cluster_index]
    combined_slope = relevant_cluster["斜率(a)"].mean()
    combined_intercept = relevant_cluster["截距(b)"].mean()
    x_value = row["检测孕周_天数"]
    y_pred = combined_slope * x_value + combined_intercept
    y_pred_all.append(y_pred)

y_true_all = df["Y染色体浓度"].values
# 计算整体残差平方和
ssr_all = ((y_true_all - np.array(y_pred_all)) ** 2).sum()
# 计算总离差平方和
ss_tot_all = np.sum((y_true_all - np.mean(y_true_all)) ** 2)
# 计算整体决定系数 R^2
r_squared_all = 1 - (ssr_all / ss_tot_all)

print(f"整体残差平方和 (SSR): {ssr_all:.4f}")
print(f"整体决定系数 R^2: {r_squared_all:.4f}")
