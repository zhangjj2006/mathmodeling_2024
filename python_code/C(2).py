import pandas as pd
import numpy as np
import os

# def process_pregnancy_data():
#     excel_path = os.path.join(".", "python_code","附件.xlsx")
#     df = pd.read_excel(excel_path) 
#     # groupby：按“孕妇代码”分组
#     # agg：对“孕周BMI”和“Y染色体浓度”分别求均值
#     # reset_index：将“孕妇代码”从“分组索引”转为普通列
#     result = (
#         df.groupby("孕妇代码")
#         .agg(
#             {
#                 "孕妇BMI": "mean",  # 对“孕周BMI”列求平均
#                 "Y染色体浓度": "mean",  # 对“Y染色体浓度”列求平均（列名需与Excel完全一致）
#             }
#         )
#         .reset_index()
#     )
#     output_path = os.path.join(".", "python_code","孕妇数据_平均值结果.xlsx")
#     result.to_excel(output_path, index=False)  # index=False：不保存行索引
#     print(f"处理完成！结果已保存至：{output_path}")


# if __name__ == "__main__":
#     process_pregnancy_data()
    
import pandas as pd
import numpy as np
import os

def process_male_pregnancy_data():
    # 读取数据
    excel_path = os.path.join(".", "python_code", "附件.xlsx")
    df = pd.read_excel(excel_path)
    
    # 列名配置（根据实际数据调整）
    code_col = '孕妇代码'
    week_col = '孕周'
    bmi_col = '孕妇BMI'
    y_col = 'Y染色体浓度'
    
    # 筛选男胎孕妇：至少有一次Y浓度≥4%
    male_pregnant_codes = df[df[y_col] >= 4][code_col].unique()
    male_df = df[df[code_col].isin(male_pregnant_codes)]
    

    # 计算每位男胎孕妇的平均BMI和最早达标孕周
    results = []
    for code in male_pregnant_codes:
        data = male_df[male_df[code_col] == code]
        mean_bmi = data[bmi_col].mean()
        ok_data = data[data[y_col] >= 0.04]
        min_day = ok_data[week_col].min()
        results.append({code_col: code, 'BMI': mean_bmi, '最早达标天数': min_day})
    
    result_df = pd.DataFrame(results)
    
    # 根据平均BMI进行等频分组（5组）
    try:
        result_df['BMI_group'] = pd.qcut(result_df['BMI'], q=5, duplicates='drop')
    except ValueError:
        bins = np.percentile(result_df['BMI'], [0, 20, 40, 60, 80, 100])
        result_df['BMI_group'] = pd.cut(result_df['BMI'], bins=bins, include_lowest=True)

    # 计算每组的最佳NIPT时点（最早达标天数的第95百分位数）
    grouped = result_df.groupby('BMI_group')['最早达标天数']
    best_weeks = grouped.quantile(0.95)
    
    # 输出结果
    print("男胎孕妇BMI分组及最佳NIPT时点（确保95%孕妇已达标）：")
    for group, week in best_weeks.items():
        print(f"BMI区间: {group}, 最佳NIPT时点: {week:.2f} 周")
    
    # 保存结果
    output_df = pd.DataFrame({'BMI_group': best_weeks.index, '最佳NIPT时点': best_weeks.values})
    output_path = os.path.join(".", "python_code", "男胎孕妇BMI分组最佳NIPT时点.csv")
    output_df.to_csv(output_path, index=False)
    print(f"\n结果已保存至: {output_path}")
    
    return result_df, best_weeks

def analyze_error_impact(result_df, best_weeks, error_std=0.5):
    """
    分析检测误差对结果的影响（模拟Y染色体浓度的测量误差）
    error_std: 假设测量误差的标准差（单位：%）
    """
    print("\n===== 检测误差分析 =====")
    print(f"假设Y染色体浓度测量误差服从正态分布N(0, {error_std})")
    
    # 模拟误差：对每位孕妇的最早达标时间重新计算
    simulated_weeks = []
    for _, row in result_df.iterrows():
        # 假设实际Y浓度有误差，但这里简化处理：直接在最达标时间上引入误差?
        # 更准确的方法是模拟每次测量，但数据不足，我们假设最早达标时间有偏移
        # 由于误差对达标时间的影响复杂，这里仅示意性分析
        original_week = row['最早达标孕周']
        # 误差导致达标时间可能提前或延后，我们假设误差对孕周的影响为±1周（根据误差大小调整）
        error_effect = np.random.normal(0, 0.5)  # 误差对孕周的影响系数，假设0.5周/%
        simulated_week = original_week + error_effect
        simulated_weeks.append(simulated_week)
    
    result_df['模拟最早达标孕周'] = simulated_weeks
    
    # 重新计算每组的最佳NIPT时点
    grouped_simulated = result_df.groupby('BMI_group')['模拟最早达标孕周']
    best_weeks_simulated = grouped_simulated.quantile(0.95)
    
    print("误差模拟后最佳NIPT时点变化:")
    for group, week in best_weeks_simulated.items():
        original_week = best_weeks[group]
        change = week - original_week
        print(f"BMI区间: {group}, 原时点: {original_week:.2f} 周, 模拟时点: {week:.2f} 周, 变化: {change:+.2f} 周")
    
    return best_weeks_simulated

if __name__ == "__main__":
    # 处理数据并获取分组结果
    result_df, best_weeks = process_male_pregnancy_data()
    
    # 分析检测误差对结果的影响（假设误差标准差为0.5%）
    analyze_error_impact(result_df, best_weeks, error_std=0.5)