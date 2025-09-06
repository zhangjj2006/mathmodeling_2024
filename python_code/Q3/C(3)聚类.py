
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 执行聚类分析
def perform_clustering(df, n_clusters=5):
    """
    执行K-means聚类分析
    """
    # 提取BMI数据并标准化
    X = df[['孕妇BMI']].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 执行K-means聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # 将聚类结果添加到数据框中
    df['Cluster'] = clusters
    
    # 计算聚类中心（原始尺度）
    centers_scaled = kmeans.cluster_centers_
    centers_original = scaler.inverse_transform(centers_scaled)
    
    return df, kmeans, scaler, centers_original, X_scaled

# 评估聚类效果
def evaluate_clustering(X, clusters):
    """
    计算聚类评估指标
    """
    silhouette_avg = silhouette_score(X, clusters)
    dbi = davies_bouldin_score(X, clusters)
    ch_index = calinski_harabasz_score(X, clusters)
    
    print("\n聚类效果评估:")
    print(f"轮廓系数 (Silhouette Score): {silhouette_avg:.4f}")
    print(f"DBI指数 (Davies-Bouldin Index): {dbi:.4f}")
    print(f"CH指数 (Calinski-Harabasz Index): {ch_index:.4f}")
    
    return silhouette_avg, dbi, ch_index

# 打印分类的BMI范围
def print_cluster_ranges(df):
    """
    打印每个聚类的BMI范围
    """
    print("\n各聚类的BMI范围:")
    print("-" * 50)
    for i in range(5):
        cluster_data = df[df['Cluster'] == i]['孕妇BMI']
        min_bmi = cluster_data.min()
        max_bmi = cluster_data.max()
        mean_bmi = cluster_data.mean()
        count = len(cluster_data)
        
        # 确定BMI分类标签
        if mean_bmi < 18.5:
            category = "偏瘦"
        elif mean_bmi < 24:
            category = "正常"
        elif mean_bmi < 28:
            category = "偏胖"
        else:
            category = "肥胖"
            
        print(f"聚类 {i}: {count:3d} 个样本, BMI范围: {min_bmi:.2f}-{max_bmi:.2f}, "
              f"平均值: {mean_bmi:.2f} ({category})")
    
    print("-" * 50)

# 可视化结果
def visualize_results(df, centers_original, silhouette_avg, dbi, ch_index):
    """
    可视化聚类结果
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 聚类分布直方图
    for i in range(5):
        cluster_data = df[df['Cluster'] == i]['孕妇BMI']
        axes[0, 0].hist(cluster_data, alpha=0.6, bins=20, label=f'Cluster {i}')
    
    axes[0, 0].axvline(x=18.5, color='r', linestyle='--', label='偏瘦阈值')
    axes[0, 0].axvline(x=24, color='g', linestyle='--', label='正常阈值')
    axes[0, 0].axvline(x=28, color='orange', linestyle='--', label='偏胖阈值')
    axes[0, 0].set_xlabel('BMI')
    axes[0, 0].set_ylabel('频数')
    axes[0, 0].set_title('各聚类BMI分布')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 聚类中心比较
    cluster_labels = [f'Cluster {i}' for i in range(5)]
    axes[0, 1].bar(cluster_labels, centers_original.flatten(), color=['blue', 'green', 'red', 'purple', 'orange'])
    axes[0, 1].set_ylabel('BMI值')
    axes[0, 1].set_title('各聚类中心BMI值')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 达标天数与BMI的关系
    for i in range(5):
        cluster_data = df[df['Cluster'] == i]
        axes[1, 0].scatter(cluster_data['孕妇BMI'], cluster_data['检测孕周_天数'], 
                          alpha=0.6, label=f'Cluster {i}')
    
    axes[1, 0].set_xlabel('BMI')
    axes[1, 0].set_ylabel('检测孕周_天数')
    axes[1, 0].set_title('BMI与检测孕周_天数的关系')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)


# 保存结果
def save_results(df, centers_original, silhouette_avg, dbi, ch_index):
    """
    保存聚类结果到Excel文件
    """
    # 创建结果摘要
    summary_data = {
        '聚类编号': range(5),
        '聚类中心(BMI)': centers_original.flatten(),
        '样本数量': [len(df[df['Cluster'] == i]) for i in range(5)],
        'BMI最小值': [df[df['Cluster'] == i]['孕妇BMI'].min() for i in range(5)],
        'BMI最大值': [df[df['Cluster'] == i]['孕妇BMI'].max() for i in range(5)],
        'BMI平均值': [df[df['Cluster'] == i]['孕妇BMI'].mean() for i in range(5)]
    }
    summary_df = pd.DataFrame(summary_data)
    
    # 添加评估指标
    eval_df = pd.DataFrame({
        '评估指标': ['轮廓系数', 'DBI指数', 'CH指数'],
        '值': [silhouette_avg, dbi, ch_index]
    })
    
    # 保存到Excel
    with pd.ExcelWriter('聚类分析结果.xlsx') as writer:
        df.to_excel(writer, sheet_name='原始数据与聚类结果', index=False)
        summary_df.to_excel(writer, sheet_name='聚类摘要', index=False)
        eval_df.to_excel(writer, sheet_name='评估指标', index=False)
    
    print("\n结果已保存到 '聚类分析结果.xlsx'")

# 主函数
def main():
    # 请替换为您的Excel文件路径
    file_path = "python_code\Q3\Q3数据预处理.xlsx" 
    df = pd.read_excel(file_path)
    # 执行聚类分析
    df_result, kmeans, scaler, centers_original, X_scaled = perform_clustering(df)
    
    # 评估聚类效果
    silhouette_avg, dbi, ch_index = evaluate_clustering(X_scaled, df_result['Cluster'].values)
    
    # 打印分类的BMI范围
    print_cluster_ranges(df_result)
    
    # 可视化结果
    visualize_results(df_result, centers_original, silhouette_avg, dbi, ch_index)
    
    # 保存结果
    save_results(df_result, centers_original, silhouette_avg, dbi, ch_index)

if __name__ == "__main__":
    main()
