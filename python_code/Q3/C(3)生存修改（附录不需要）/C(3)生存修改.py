import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize_scalar
import statsmodels.api as sm
from lifelines import KaplanMeierFitter

# 设置中文字体支持
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

def analyze_nipt_optimal_time_with_factors():
    """
    分析不同BMI分组中年龄、身高、体重对NIPT最佳时点的影响
    """
    # 读取预处理数据
    file_path = "python_code/Q3/Q3数据预处理.xlsx" 
    df = pd.read_excel(file_path)
    
    # 定义BMI分组函数
    def categorize_bmi(bmi):
        if bmi < 30.01:
            return "聚类0 (<30.01)"
        elif 30.01 <= bmi < 32.18:
            return "聚类1 (30.01-32.18)"
        elif 32.18 <= bmi < 34.63:
            return "聚类2 (32.18-34.63)"
        elif 34.63 <= bmi < 37.93:
            return "聚类3 (34.63-37.93)"
        else:
            return "聚类4 (≥37.93)"
    
    # 应用BMI分组
    df['bmi_category'] = df['孕妇BMI'].apply(categorize_bmi)
    
    # 准备生存分析数据
    survival_data = []
    
    for code, group in df.groupby('孕妇代码'):
        # 按检测孕周排序
        group = group.sort_values('检测孕周_天数')
        
        # 找到首次达标的时间
        first_reach = group[group['Y染色体浓度'] >= 0.04]
        
        if len(first_reach) > 0:
            # 有达标记录
            event_time = first_reach.iloc[0]['检测孕周_天数']
            event_observed = 1
        else:
            # 未达标，使用最后一次检测时间作为删失时间
            event_time = group.iloc[-1]['检测孕周_天数']
            event_observed = 0
        
        # 获取孕妇的基本信息（取第一次检测的值）
        base_info = group.iloc[0]
        
        survival_data.append({
            '孕妇代码': code,
            '年龄': base_info['年龄'],
            '身高': base_info['身高'],
            '体重': base_info['体重'],
            '孕妇BMI': base_info['孕妇BMI'],
            'bmi_category': base_info['bmi_category'],
            '达标时间': event_time,
            '达标状态': event_observed
        })
    
    survival_df = pd.DataFrame(survival_data)
    
    # 定义效用函数
    def get_rearly(t):
        """早期检测效用函数：随时间指数衰减"""
        return 2.0 * np.exp(-t / 50)
    
    def get_rlate(t):
        """晚期检测风险函数：分段函数"""
        if t < 84:  # 12周
            return 0.1
        elif t <= 189:  # 27周
            normalized_t = (t - 84) / (189 - 84)
            return 0.1 + 0.9 * (normalized_t**2)
        else:  # 超过27周
            return 1.0
    
    def calculate_utility(t, survival_func):
        """计算总效用函数"""
        if t < 0:
            return float("inf")
        
        # 找到最接近的时间点
        closest_time_idx = np.abs(survival_func.index - t).argmin()
        actual_time = survival_func.index[closest_time_idx]
        
        # 计算达标概率
        p_t = 1 - survival_func.loc[actual_time, "s_t"]
        
        # 计算总效用
        return (1 - p_t) * get_rearly(t) + p_t * get_rlate(t)
    
    # 分析每个BMI分组
    bmi_categories = [
        "聚类0 (<30.01)", 
        "聚类1 (30.01-32.18)", 
        "聚类2 (32.18-34.63)", 
        "聚类3 (34.63-37.93)", 
        "聚类4 (≥37.93)"
    ]
    
    results_table = []
    
    for bmi_cat in bmi_categories:
        df_bmi = survival_df[survival_df['bmi_category'] == bmi_cat].copy()
        
        if df_bmi.empty:
            results_table.append({
                "BMI分组": bmi_cat,
                "最优时点(天)": "无数据",
                "最小效用值": "无数据",
                "风险水平": "无数据",
                "样本量": 0,
            })
            continue
        
        print(f"\n=== 正在分析 BMI 分组: {bmi_cat} ===")
        print(f"样本量: {len(df_bmi)}")
        
        # 计算KM生存曲线
        kmf = KaplanMeierFitter()
        kmf.fit(durations=df_bmi["达标时间"], event_observed=df_bmi["达标状态"])
        
        s_t = kmf.survival_function_.rename(columns={"KM_estimate": "s_t"})
        
        # 优化找到最佳时点
        def objective(t):
            return calculate_utility(t, s_t)
        
        result = minimize_scalar(
            objective, bounds=(0, df_bmi["达标时间"].max()), method="bounded"
        )
        
        optimal_time = result.x
        optimal_utility = result.fun
        risk_level = 1 / optimal_utility if optimal_utility > 0 else float("inf")
        sample_size = len(df_bmi)
        
        print(f"\n最优预测时间点分析:")
        print(f"  - 最优时间点: {optimal_time:.1f} 天 (约{optimal_time/7:.1f}周)")
        print(f"  - 最小效用值: {optimal_utility:.4f}")
        print(f"  - 风险水平: {risk_level:.4f}")
        
        # 分析年龄、身高、体重对达标时间的影响
        print("\n影响因素分析:")
        
        # 重置索引以确保一致性
        df_bmi_reset = df_bmi.reset_index(drop=True)
        
        # 准备回归数据
        X = df_bmi_reset[['年龄', '身高', '体重']]
        y = df_bmi_reset['达标时间']
        
        # 标准化特征
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=['年龄', '身高', '体重'])
        
        # 重置索引以确保一致性
        X_scaled = X_scaled.reset_index(drop=True)
        y = y.reset_index(drop=True)
        
        # 添加常数项
        X_scaled = sm.add_constant(X_scaled)
        
        # 检查样本量是否足够进行回归分析
        if len(X_scaled) < 5:  # 至少需要5个样本
            print("样本量不足，无法进行回归分析")
            age_coef = np.nan
            height_coef = np.nan
            weight_coef = np.nan
        else:
            try:
                # 拟合OLS模型
                model = sm.OLS(y, X_scaled).fit()
                
                # 打印回归结果
                print(model.summary())
                
                # 提取系数
                age_coef = model.params['年龄']
                height_coef = model.params['身高']
                weight_coef = model.params['体重']
                
                print(f"\n影响因素对达标时间的影响:")
                print(f"年龄: 每增加1个标准差，达标时间变化 {age_coef:.2f} 天")
                print(f"身高: 每增加1个标准差，达标时间变化 {height_coef:.2f} 天")
                print(f"体重: 每增加1个标准差，达标时间变化 {weight_coef:.2f} 天")
            except Exception as e:
                print(f"回归分析出错: {e}")
                age_coef = np.nan
                height_coef = np.nan
                weight_coef = np.nan
        
        # 可视化影响因素
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 生存曲线和效用函数
        ax = axes[0, 0]
        (1 - kmf.survival_function_).plot(
            ax=ax,
            label=f"累积达标函数 F(t) ({bmi_cat})",
            color="seagreen",
            linewidth=2.5,
        )
        
        time_points = np.linspace(0, df_bmi["达标时间"].max(), 200)
        utilities = [calculate_utility(t, s_t) for t in time_points]
        ax_twin = ax.twinx()
        ax_twin.plot(
            time_points, utilities, "--", color="red", label="效用函数 E(t)", alpha=0.6
        )
        
        ax_twin.axvline(x=optimal_time, color="purple", linestyle="--", linewidth=2)
        ax_twin.text(
            optimal_time + 2,
            min(utilities) + 0.1,
            f"最优时间点: {optimal_time:.1f}天",
            color="purple",
            fontsize=10,
        )
        
        ax.axvspan(0, 84, facecolor="green", alpha=0.15, label="早期阶段 (<12周)")
        ax.axvspan(84, 189, facecolor="orange", alpha=0.15, label="中期阶段 (12-27周)")
        ax.axvspan(
            189,
            df_bmi["达标时间"].max() + 10,
            facecolor="red",
            alpha=0.15,
            label="晚期阶段 (>27周)",
        )
        
        ax.set_title(f"BMI分组 {bmi_cat} 达标情况与效用分析", fontsize=14)
        ax.set_xlabel("天数", fontsize=12)
        ax.set_ylabel("累积达标比例", fontsize=12)
        ax_twin.set_ylabel("效用值", fontsize=12)
        
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        ax_twin.set_ylim(0, max(utilities) * 1.2)
        ax.set_xlim(-5, df_bmi["达标时间"].max() + 10)
        
        # 2. 年龄与达标时间的关系
        axes[0, 1].scatter(df_bmi['年龄'], df_bmi['达标时间'], alpha=0.6)
        if not np.isnan(age_coef):
            z = np.polyfit(df_bmi['年龄'], df_bmi['达标时间'], 1)
            p = np.poly1d(z)
            axes[0, 1].plot(df_bmi['年龄'], p(df_bmi['年龄']), "r--", alpha=0.8)
            axes[0, 1].text(0.05, 0.95, f'斜率: {z[0]:.2f}', transform=axes[0, 1].transAxes, 
                           fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[0, 1].set_xlabel('年龄')
        axes[0, 1].set_ylabel('达标时间(天)')
        axes[0, 1].set_title('年龄与达标时间的关系')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 身高与达标时间的关系
        axes[1, 0].scatter(df_bmi['身高'], df_bmi['达标时间'], alpha=0.6)
        if not np.isnan(height_coef):
            z = np.polyfit(df_bmi['身高'], df_bmi['达标时间'], 1)
            p = np.poly1d(z)
            axes[1, 0].plot(df_bmi['身高'], p(df_bmi['身高']), "r--", alpha=0.8)
            axes[1, 0].text(0.05, 0.95, f'斜率: {z[0]:.2f}', transform=axes[1, 0].transAxes, 
                           fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 0].set_xlabel('身高(cm)')
        axes[1, 0].set_ylabel('达标时间(天)')
        axes[1, 0].set_title('身高与达标时间的关系')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 体重与达标时间的关系
        axes[1, 1].scatter(df_bmi['体重'], df_bmi['达标时间'], alpha=0.6)
        if not np.isnan(weight_coef):
            z = np.polyfit(df_bmi['体重'], df_bmi['达标时间'], 1)
            p = np.poly1d(z)
            axes[1, 1].plot(df_bmi['体重'], p(df_bmi['体重']), "r--", alpha=0.8)
            axes[1, 1].text(0.05, 0.95, f'斜率: {z[0]:.2f}', transform=axes[1, 1].transAxes, 
                           fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 1].set_xlabel('体重(kg)')
        axes[1, 1].set_ylabel('达标时间(天)')
        axes[1, 1].set_title('体重与达标时间的关系')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        filename = f'./python_code/BMI_{bmi_cat.replace("<", "lt").replace("≥", "ge").replace(" ", "_").replace("(", "").replace(")", "")}_analysis.png'
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"\n图表已保存至: {filename}")
        
        results_table.append({
            "BMI分组": bmi_cat,
            "最优时点(天)": f"{optimal_time:.1f}",
            "最优时点(周)": f"{optimal_time/7:.1f}",
            "最小效用值": f"{optimal_utility:.4f}",
            "风险水平": f"{risk_level:.4f}",
            "样本量": sample_size,
            "年龄影响(天/标准差)": f"{age_coef:.2f}" if not np.isnan(age_coef) else "N/A",
            "身高影响(天/标准差)": f"{height_coef:.2f}" if not np.isnan(height_coef) else "N/A",
            "体重影响(天/标准差)": f"{weight_coef:.2f}" if not np.isnan(weight_coef) else "N/A",
        })
        
        print("\n" + "="*60 + "\n")
    
    # 打印和保存结果表格
    print("\n\n=== 各BMI分组NIPT时点计算结果 ===")
    results_df = pd.DataFrame(results_table)
    print(results_df.to_string(index=False))
    
    results_df.to_excel("./python_code/NIPT_optimal_times_with_factors.xlsx", index=False)
    print(f"\n结果表格已保存至: ./python_code/NIPT_optimal_times_with_factors.xlsx")
    
    # 创建影响因素汇总图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 提取影响系数
    age_effects = []
    height_effects = []
    weight_effects = []
    labels = []
    
    for _, row in results_df.iterrows():
        if row['年龄影响(天/标准差)'] != 'N/A':
            age_effects.append(float(row['年龄影响(天/标准差)']))
            height_effects.append(float(row['身高影响(天/标准差)']))
            weight_effects.append(float(row['体重影响(天/标准差)']))
            labels.append(row['BMI分组'].split(' ')[0])
    
    # 绘制年龄影响
    if age_effects:
        axes[0].bar(range(len(age_effects)), age_effects, color='skyblue')
        axes[0].set_xlabel('BMI分组')
        axes[0].set_ylabel('影响系数(天/标准差)')
        axes[0].set_title('年龄对达标时间的影响')
        axes[0].set_xticks(range(len(age_effects)))
        axes[0].set_xticklabels(labels)
    
    # 绘制身高影响
    if height_effects:
        axes[1].bar(range(len(height_effects)), height_effects, color='lightgreen')
        axes[1].set_xlabel('BMI分组')
        axes[1].set_ylabel('影响系数(天/标准差)')
        axes[1].set_title('身高对达标时间的影响')
        axes[1].set_xticks(range(len(height_effects)))
        axes[1].set_xticklabels(labels)
    
    # 绘制体重影响
    if weight_effects:
        axes[2].bar(range(len(weight_effects)), weight_effects, color='lightcoral')
        axes[2].set_xlabel('BMI分组')
        axes[2].set_ylabel('影响系数(天/标准差)')
        axes[2].set_title('体重对达标时间的影响')
        axes[2].set_xticks(range(len(weight_effects)))
        axes[2].set_xticklabels(labels)
    
    plt.tight_layout()
    plt.savefig('./python_code/factors_impact_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n影响因素汇总图已保存至: ./python_code/factors_impact_summary.png")
    
    return results_df

# 运行分析
results = analyze_nipt_optimal_time_with_factors()