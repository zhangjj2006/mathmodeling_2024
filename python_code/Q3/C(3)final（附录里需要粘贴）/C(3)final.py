import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
from scipy.optimize import minimize_scalar
from lifelines import KaplanMeierFitter
import shap
from statsmodels.nonparametric.smoothers_lowess import lowess
import warnings
from scipy.stats import norm
import re
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 建模分析不同BMI分组中年龄、身高、体重对NIPT最佳时点的影响
def analyze_nipt_optimal_time_with_advanced_modeling():
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
    
    df['bmi_category'] = df['孕妇BMI'].apply(categorize_bmi)
    
    # 准备生存分析数据
    survival_data = []
    
    for code, group in df.groupby('孕妇代码'):
        group = group.sort_values('检测孕周_天数')
        # 找到首次达标的时间
        first_reach_idx = group[group['Y染色体浓度'] >= 0.04].index
        last_below_idx = group[group['Y染色体浓度'] < 0.04].index
        
        if len(first_reach_idx) > 0:
            # 有达标记录
            first_reach_time = group.loc[first_reach_idx[0], '检测孕周_天数']
            first_reach_concentration = group.loc[first_reach_idx[0], 'Y染色体浓度']
            
            # 找到最后一次未达标的时间
            if len(last_below_idx) > 0 and last_below_idx[-1] < first_reach_idx[0]:
                last_below_time = group.loc[last_below_idx[-1], '检测孕周_天数']
                last_below_concentration = group.loc[last_below_idx[-1], 'Y染色体浓度']
                
                # 使用线性插值法计算达标时间
                # 公式: t = t0 + (t1 - t0) * (0.04 - c0) / (c1 - c0)
                event_time = last_below_time + (first_reach_time - last_below_time) * \
                            (0.04 - last_below_concentration) / (first_reach_concentration - last_below_concentration)
            else:
                # 如果没有之前的未达标记录，使用首次达标时间
                event_time = first_reach_time
            
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
    model_performance = []
    equations = []  # 存储每个分组的NIPT时点方程
    optimal_times = {}  # 存储每个分组的最优时点
    
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
        
        # 存储最优时点
        optimal_times[bmi_cat] = optimal_time
        
        print(f"\n最优预测时间点分析:")
        print(f"  - 最优时间点: {optimal_time:.1f} 天 (约{optimal_time/7:.1f}周)")
        print(f"  - 最小效用值: {optimal_utility:.4f}")
        print(f"  - 风险水平: {risk_level:.4f}")
        
        # 使用随机森林建模年龄、身高、体重对达标时间的影响
        print("\n使用随机森林建模影响因素:")
        
        # 重置索引以确保一致性
        df_bmi_reset = df_bmi.reset_index(drop=True)
        
        # 准备回归数据
        X = df_bmi_reset[['年龄', '身高', '体重']]
        y = df_bmi_reset['达标时间']
        
        # 检查样本量是否足够进行建模
        if len(X) < 10:  # 至少需要10个样本
            print("样本量不足，使用线性回归建模")
            
            # 使用线性回归
            try:
                # 使用线性回归
                lr = LinearRegression()
                lr.fit(X, y)
                
                # 获取系数
                coefficients = lr.coef_
                intercept = lr.intercept_
                
                # 生成回归方程
                equation = f"T = {intercept:.2f} + {coefficients[0]:.2f}×年龄 + {coefficients[1]:.2f}×身高 + {coefficients[2]:.2f}×体重"
                
                # 计算模型性能
                y_pred = lr.predict(X)
                model_r2 = lr.score(X, y)
                model_mae = np.mean(np.abs(y - y_pred))
                
                print(f"线性回归模型性能:")
                print(f"  - R²得分: {model_r2:.4f}")
                print(f"  - 平均绝对误差: {model_mae:.2f}天")
                print(f"回归方程: {equation}")
                
                # 计算特征重要性（使用系数的绝对值）
                feature_importance = np.abs(coefficients) / np.sum(np.abs(coefficients))
                age_importance = feature_importance[0]
                height_importance = feature_importance[1]
                weight_importance = feature_importance[2]
                
                print(f"特征重要性:")
                print(f"  - 年龄: {age_importance:.4f}")
                print(f"  - 身高: {height_importance:.4f}")
                print(f"  - 体重: {weight_importance:.4f}")
                
                # 验证引入影响因素后的改进
                y_mean = np.full_like(y_pred, y.mean())
                mae_mean = np.mean(np.abs(y - y_mean))
                improvement = (mae_mean - model_mae) / mae_mean * 100
                
                print(f"\n模型验证结果:")
                print(f"  - 仅使用平均值预测的MAE: {mae_mean:.2f}天")
                print(f"  - 使用线性回归预测的MAE: {model_mae:.2f}天")
                print(f"  - 改进百分比: {improvement:.2f}%")
                
                model_performance.append({
                    "BMI分组": bmi_cat,
                    "样本量": len(X),
                    "R²得分": model_r2,
                    "模型MAE": model_mae,
                    "平均值MAE": mae_mean,
                    "改进百分比": improvement,
                    "模型类型": "线性回归"
                })
                
                # 存储方程
                equations.append({
                    "BMI分组": bmi_cat,
                    "方程": equation,
                    "模型类型": "线性回归",
                    "最优时点": optimal_time
                })
                
            except Exception as e:
                print(f"线性回归建模出错: {e}")
                age_importance = np.nan
                height_importance = np.nan
                weight_importance = np.nan
                model_r2 = np.nan
                model_mae = np.nan
                equation = "无法生成方程"
        else:
            try:
                # 标准化特征
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # 使用随机森林回归
                rf = RandomForestRegressor(n_estimators=100, random_state=42)
                
                # 交叉验证评估模型性能
                kf = KFold(n_splits=min(5, len(X)), shuffle=True, random_state=42)
                r2_scores = cross_val_score(rf, X_scaled, y, cv=kf, scoring='r2')
                mae_scores = -cross_val_score(rf, X_scaled, y, cv=kf, scoring='neg_mean_absolute_error')
                
                # 训练最终模型
                rf.fit(X_scaled, y)
                
                # 获取特征重要性
                feature_importance = rf.feature_importances_
                age_importance = feature_importance[0]
                height_importance = feature_importance[1]
                weight_importance = feature_importance[2]
                
                # 计算模型性能
                model_r2 = np.mean(r2_scores)
                model_mae = np.mean(mae_scores)
                
                print(f"随机森林模型性能:")
                print(f"  - R²得分: {model_r2:.4f}")
                print(f"  - 平均绝对误差: {model_mae:.2f}天")
                print(f"特征重要性:")
                print(f"  - 年龄: {age_importance:.4f}")
                print(f"  - 身高: {height_importance:.4f}")
                print(f"  - 体重: {weight_importance:.4f}")
                
                # 使用SHAP值解释模型
                if len(X) >= 20:  # 只有当样本量足够时才计算SHAP值
                    try:
                        explainer = shap.TreeExplainer(rf)
                        shap_values = explainer.shap_values(X_scaled)
                        
                        plt.figure(figsize=(10, 6))
                        shap.summary_plot(shap_values, X_scaled, feature_names=['年龄', '身高', '体重'], show=False)
                        plt.title(f'BMI分组 {bmi_cat} 特征影响分析 (SHAP值)')
                        plt.tight_layout()
                        plt.savefig(f'./python_code/BMI_{bmi_cat.replace("<", "lt").replace("≥", "ge").replace(" ", "_").replace("(", "").replace(")", "")}_shap.png', dpi=300, bbox_inches="tight")
                        plt.close()
                        print("SHAP分析图表已保存")
                    except Exception as e:
                        print(f"SHAP分析出错: {e}")
                
                # 验证引入影响因素后的改进
                y_pred = rf.predict(X_scaled)
                y_mean = np.full_like(y_pred, y.mean())
                mae_model = np.mean(np.abs(y - y_pred))
                mae_mean = np.mean(np.abs(y - y_mean))
                improvement = (mae_mean - mae_model) / mae_mean * 100
                
                print(f"\n模型验证结果:")
                print(f"  - 仅使用平均值预测的MAE: {mae_mean:.2f}天")
                print(f"  - 使用完整模型预测的MAE: {mae_model:.2f}天")
                print(f"  - 改进百分比: {improvement:.2f}%")
                
                model_performance.append({
                    "BMI分组": bmi_cat,
                    "样本量": len(X),
                    "R²得分": model_r2,
                    "模型MAE": mae_model,
                    "平均值MAE": mae_mean,
                    "改进百分比": improvement,
                    "模型类型": "随机森林"
                })
                
                # 为随机森林生成更合理的方程描述
                # 计算每个特征的相对影响
                age_effect = age_importance * 100
                height_effect = height_importance * 100
                weight_effect = weight_importance * 100
                
                # 生成方程描述
                equation = f"T ≈ {optimal_time:.1f} + {age_effect:.1f}%×(年龄影响) + {height_effect:.1f}%×(身高影响) + {weight_effect:.1f}%×(体重影响)"
                
                # 存储方程
                equations.append({
                    "BMI分组": bmi_cat,
                    "方程": equation,
                    "模型类型": "随机森林",
                    "最优时点": optimal_time
                })
                
            except Exception as e:
                print(f"随机森林建模出错: {e}")
                age_importance = np.nan
                height_importance = np.nan
                weight_importance = np.nan
                model_r2 = np.nan
                model_mae = np.nan
                equation = "无法生成方程"
        
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
        
        ax_twin.axvline(x=optimal_time, color='purple', linestyle='--', linewidth=2)
        ax_twin.text(
            optimal_time + 2,
            min(utilities) + 0.1,
            f"最优时间点: {optimal_time:.1f}天",
            color='purple',
            fontsize=10,
        )
        
        ax.axvspan(0, 84, facecolor='green', alpha=0.15, label='早期阶段 (<12周)')
        ax.axvspan(84, 189, facecolor='orange', alpha=0.15, label='中期阶段 (12-27周)')
        ax.axvspan(
            189,
            df_bmi["达标时间"].max() + 10,
            facecolor='red',
            alpha=0.15,
            label='晚期阶段 (>27周)',
        )
        
        ax.set_title(f"BMI分组 {bmi_cat} 达标情况与效用分析", fontsize=14)
        ax.set_xlabel("天数", fontsize=12)
        ax.set_ylabel("累积达标比例", fontsize=12)
        ax_twin.set_ylabel("效用值", fontsize=12)
        
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        ax_twin.set_ylim(0, max(utilities) * 1.2)
        ax.set_xlim(-5, df_bmi["达标时间"].max() + 10)
        
        # 2. 年龄与达标时间的关系
        axes[0, 1].scatter(df_bmi['年龄'], df_bmi['达标时间'], alpha=0.6)
        if not np.isnan(age_importance):
            # 使用局部加权散点平滑（LOWESS）拟合曲线
            try:
                lowess_fit = lowess(df_bmi['达标时间'], df_bmi['年龄'], frac=0.7)
                axes[0, 1].plot(lowess_fit[:, 0], lowess_fit[:, 1], 'r-', alpha=0.8, label='LOWESS拟合')
            except:
                # 如果LOWESS失败，使用多项式拟合
                z = np.polyfit(df_bmi['年龄'], df_bmi['达标时间'], 2)
                p = np.poly1d(z)
                x_sorted = np.sort(df_bmi['年龄'])
                axes[0, 1].plot(x_sorted, p(x_sorted), 'r-', alpha=0.8, label='多项式拟合')
        axes[0, 1].set_xlabel('年龄')
        axes[0, 1].set_ylabel('达标时间(天)')
        axes[0, 1].set_title('年龄与达标时间的关系')
        axes[0, 1].grid(True, alpha=0.3)
        if not np.isnan(age_importance):
            axes[0, 1].text(0.05, 0.95, f'重要性: {age_importance:.4f}', transform=axes[0, 1].transAxes, 
                           fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 3. 身高与达标时间的关系
        axes[1, 0].scatter(df_bmi['身高'], df_bmi['达标时间'], alpha=0.6)
        if not np.isnan(height_importance):
            try:
                lowess_fit = lowess(df_bmi['达标时间'], df_bmi['身高'], frac=0.7)
                axes[1, 0].plot(lowess_fit[:, 0], lowess_fit[:, 1], 'r-', alpha=0.8, label='LOWESS拟合')
            except:
                z = np.polyfit(df_bmi['身高'], df_bmi['达标时间'], 2)
                p = np.poly1d(z)
                x_sorted = np.sort(df_bmi['身高'])
                axes[1, 0].plot(x_sorted, p(x_sorted), 'r-', alpha=0.8, label='多项式拟合')
        axes[1, 0].set_xlabel('身高(cm)')
        axes[1, 0].set_ylabel('达标时间(天)')
        axes[1, 0].set_title('身高与达标时间的关系')
        axes[1, 0].grid(True, alpha=0.3)
        if not np.isnan(height_importance):
            axes[1, 0].text(0.05, 0.95, f'重要性: {height_importance:.4f}', transform=axes[1, 0].transAxes, 
                           fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 4. 体重与达标时间的关系
        axes[1, 1].scatter(df_bmi['体重'], df_bmi['达标时间'], alpha=0.6)
        if not np.isnan(weight_importance):
            try:
                lowess_fit = lowess(df_bmi['达标时间'], df_bmi['体重'], frac=0.7)
                axes[1, 1].plot(lowess_fit[:, 0], lowess_fit[:, 1], 'r-', alpha=0.8, label='LOWESS拟合')
            except:
                z = np.polyfit(df_bmi['体重'], df_bmi['达标时间'], 2)
                p = np.poly1d(z)
                x_sorted = np.sort(df_bmi['体重'])
                axes[1, 1].plot(x_sorted, p(x_sorted), 'r-', alpha=0.8, label='多项式拟合')
        axes[1, 1].set_xlabel('体重(kg)')
        axes[1, 1].set_ylabel('达标时间(天)')
        axes[1, 1].set_title('体重与达标时间的关系')
        axes[1, 1].grid(True, alpha=0.3)
        if not np.isnan(weight_importance):
            axes[1, 1].text(0.05, 0.95, f'重要性: {weight_importance:.4f}', transform=axes[1, 1].transAxes, 
                           fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
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
            "年龄重要性": f"{age_importance:.4f}" if not np.isnan(age_importance) else "N/A",
            "身高重要性": f"{height_importance:.4f}" if not np.isnan(height_importance) else "N/A",
            "体重重要性": f"{weight_importance:.4f}" if not np.isnan(weight_importance) else "N/A",
            "模型R²": f"{model_r2:.4f}" if not np.isnan(model_r2) else "N/A",
            "模型MAE": f"{model_mae:.2f}" if not np.isnan(model_mae) else "N/A",
        })
        
        print("\n" + "="*60 + "\n")
    
    # 打印和保存结果表格
    print("\n\n=== 各BMI分组NIPT时点计算结果 ===")
    results_df = pd.DataFrame(results_table)
    print(results_df.to_string(index=False))
    
    results_df.to_excel("./python_code/NIPT_optimal_times_with_advanced_modeling.xlsx", index=False)
    print(f"\n结果表格已保存至: ./python_code/NIPT_optimal_times_with_advanced_modeling.xlsx")
    
    # 保存模型性能评估结果
    if model_performance:
        performance_df = pd.DataFrame(model_performance)
        performance_df.to_excel("./python_code/model_performance_evaluation.xlsx", index=False)
        print(f"模型性能评估已保存至: ./python_code/model_performance_evaluation.xlsx")
        
        # 绘制模型性能比较图
        plt.figure(figsize=(10, 6))
        x = np.arange(len(performance_df))
        width = 0.35
        
        plt.bar(x - width/2, performance_df['平均值MAE'], width, label='仅使用平均值')
        plt.bar(x + width/2, performance_df['模型MAE'], width, label='使用完整模型')
        
        plt.xlabel('BMI分组')
        plt.ylabel('平均绝对误差(天)')
        plt.title('模型性能比较: 仅使用平均值 vs 使用完整模型')
        plt.xticks(x, performance_df['BMI分组'])
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 在柱状图上添加数值标签
        for i, v in enumerate(performance_df['平均值MAE']):
            plt.text(i - width/2, v + 0.5, f'{v:.1f}', ha='center')
        
        for i, v in enumerate(performance_df['模型MAE']):
            plt.text(i + width/2, v + 0.5, f'{v:.1f}', ha='center')
        
        plt.tight_layout()
        plt.savefig('./python_code/model_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("模型性能比较图已保存")
    
    # 创建更合理的NIPT时点方程
    print("\n\n=== 各BMI分组NIPT时点方程 ===")
    improved_equations = []
    
    for bmi_cat in bmi_categories:
        if bmi_cat not in optimal_times:
            improved_equations.append({
                "BMI分组": bmi_cat,
                "NIPT时点方程": "无数据",
                "模型类型": "无数据"
            })
            continue
        
        optimal_time = optimal_times[bmi_cat]
        
        # 根据BMI分组提供个性化的NIPT时点建议
        if bmi_cat == "聚类0 (<30.01)":
            equation = f"建议NIPT时点: {optimal_time:.1f}天 (约{optimal_time/7:.1f}周)\n对于BMI<30.01的孕妇，建议在孕{optimal_time/7:.1f}周左右进行NIPT检测"
        elif bmi_cat == "聚类1 (30.01-32.18)":
            equation = f"建议NIPT时点: {optimal_time:.1f}天 (约{optimal_time/7:.1f}周)\n对于BMI 30.01-32.18的孕妇，建议在孕{optimal_time/7:.1f}周左右进行NIPT检测"
        elif bmi_cat == "聚类2 (32.18-34.63)":
            equation = f"建议NIPT时点: {optimal_time:.1f}天 (约{optimal_time/7:.1f}周)\n对于BMI 32.18-34.63的孕妇，建议在孕{optimal_time/7:.1f}周左右进行NIPT检测"
        elif bmi_cat == "聚类3 (34.63-37.93)":
            equation = f"建议NIPT时点: {optimal_time:.1f}天 (约{optimal_time/7:.1f}周)\n对于BMI 34.63-37.93的孕妇，建议在孕{optimal_time/7:.1f}周左右进行NIPT检测"
        else:  # 聚类4 (≥37.93)
            equation = f"建议NIPT时点: {optimal_time:.1f}天 (约{optimal_time/7:.1f}周)\n对于BMI≥37.93的孕妇，建议在孕{optimal_time/7:.1f}周左右进行NIPT检测"
        
        # 确定模型类型
        model_type = "线性回归" if bmi_cat == "聚类4 (≥37.93)" else "随机森林"
        
        improved_equations.append({
            "BMI分组": bmi_cat,
            "NIPT时点方程": equation,
            "模型类型": model_type
        })
        
        print(f"\n{bmi_cat}:")
        print(equation)
    
    # 保存改进的NIPT时点方程
    improved_equations_df = pd.DataFrame(improved_equations)
    improved_equations_df.to_excel("./python_code/improved_NIPT_time_equations.xlsx", index=False)
    print(f"\n改进的NIPT时点方程已保存至: ./python_code/improved_NIPT_time_equations.xlsx")
    
    # 创建影响因素汇总图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 提取影响系数
    age_effects = []
    height_effects = []
    weight_effects = []
    labels = []
    
    for _, row in results_df.iterrows():
        if row['年龄重要性'] != 'N/A':
            age_effects.append(float(row['年龄重要性']))
            height_effects.append(float(row['身高重要性']))
            weight_effects.append(float(row['体重重要性']))
            labels.append(row['BMI分组'].split(' ')[0])
    
    # 绘制年龄影响
    if age_effects:
        axes[0].bar(range(len(age_effects)), age_effects, color='skyblue')
        axes[0].set_xlabel('BMI分组')
        axes[0].set_ylabel('特征重要性')
        axes[0].set_title('年龄对达标时间的影响')
        axes[0].set_xticks(range(len(age_effects)))
        axes[0].set_xticklabels(labels)
    
    # 绘制身高影响
    if height_effects:
        axes[1].bar(range(len(height_effects)), height_effects, color='lightgreen')
        axes[1].set_xlabel('BMI分组')
        axes[1].set_ylabel('特征重要性')
        axes[1].set_title('身高对达标时间的影响')
        axes[1].set_xticks(range(len(height_effects)))
        axes[1].set_xticklabels(labels)
    
    # 绘制体重影响
    if weight_effects:
        axes[2].bar(range(len(weight_effects)), weight_effects, color='lightcoral')
        axes[2].set_xlabel('BMI分组')
        axes[2].set_ylabel('特征重要性')
        axes[2].set_title('体重对达标时间的影响')
        axes[2].set_xticks(range(len(weight_effects)))
        axes[2].set_xticklabels(labels)
    
    plt.tight_layout()
    plt.savefig('./python_code/factors_impact_summary_advanced.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n影响因素汇总图已保存至: ./python_code/factors_impact_summary_advanced.png")
    
    error_analysis_results = analyze_error_impact(survival_df, df, bmi_categories, num_simulations=100, error_std=0.01)
    return results_df, improved_equations_df, error_analysis_results


#分析检测误差对最优时点、效用值和风险的影响
def analyze_error_impact(survival_df, original_df, bmi_categories, num_simulations=100, error_std=0.01):
    # 辅助函数：清理文件名中的非法字符
    def clean_filename(name):
        return re.sub(r'[<>:"/\\|?*]', "_", name)
    
    # 存储原始最优时点
    original_optimal_times = {}
    error_results = {bmi_cat: {"times": [], "utilities": [], "risks": []} for bmi_cat in bmi_categories}
    
    # 定义效用函数（与主函数中相同）
    def get_rearly(t):
        return 2.0 * np.exp(-t / 50)
    
    def get_rlate(t):
        if t < 84:
            return 0.1
        elif t <= 189:
            normalized_t = (t - 84) / (189 - 84)
            return 0.1 + 0.9 * (normalized_t**2)
        else:
            return 1.0
    
    def calculate_utility(t, survival_func):
        if t < 0:
            return float("inf")
        
        closest_time_idx = np.abs(survival_func.index - t).argmin()
        actual_time = survival_func.index[closest_time_idx]
        p_t = 1 - survival_func.loc[actual_time, "s_t"]
        return (1 - p_t) * get_rearly(t) + p_t * get_rlate(t)
    
    # 计算原始最优时点
    for bmi_cat in bmi_categories:
        df_bmi = survival_df[survival_df['bmi_category'] == bmi_cat].copy()
        if df_bmi.empty:
            continue
        
        kmf = KaplanMeierFitter()
        kmf.fit(durations=df_bmi["达标时间"], event_observed=df_bmi["达标状态"])
        s_t = kmf.survival_function_.rename(columns={"KM_estimate": "s_t"})
        
        def objective(t):
            return calculate_utility(t, s_t)
        
        result = minimize_scalar(objective, bounds=(0, df_bmi["达标时间"].max()), method="bounded")
        original_optimal_times[bmi_cat] = result.x
    
    # 开始误差模拟
    for sim in range(num_simulations):
        print(f"正在进行第 {sim+1} 次误差模拟...")
        np.random.seed(sim)  # 确保可重复性
        
        # 创建带误差的数据副本
        df_error = original_df.copy()
        
        # 对Y染色体浓度添加随机误差
        df_error['Y染色体浓度_error'] = df_error['Y染色体浓度'] + np.random.normal(
            0, error_std, size=len(df_error)
        )
        
        # 重新计算每个孕妇的达标时间（带误差）
        survival_data_error = []
        
        for code, group in df_error.groupby('孕妇代码'):
            group = group.sort_values('检测孕周_天数')
            
            # 找到首次达标的时间（使用带误差的Y染色体浓度）
            first_reach_idx = group[group['Y染色体浓度_error'] >= 0.04].index
            last_below_idx = group[group['Y染色体浓度_error'] < 0.04].index
            
            if len(first_reach_idx) > 0:
                first_reach_time = group.loc[first_reach_idx[0], '检测孕周_天数']
                first_reach_concentration = group.loc[first_reach_idx[0], 'Y染色体浓度_error']
                
                if len(last_below_idx) > 0 and last_below_idx[-1] < first_reach_idx[0]:
                    last_below_time = group.loc[last_below_idx[-1], '检测孕周_天数']
                    last_below_concentration = group.loc[last_below_idx[-1], 'Y染色体浓度_error']
                    
                    # 使用线性插值法计算达标时间
                    event_time = last_below_time + (first_reach_time - last_below_time) * \
                                (0.04 - last_below_concentration) / (first_reach_concentration - last_below_concentration)
                else:
                    event_time = first_reach_time
                
                event_observed = 1
            else:
                event_time = group.iloc[-1]['检测孕周_天数']
                event_observed = 0
            
            base_info = group.iloc[0]
            
            survival_data_error.append({
                '孕妇代码': code,
                '年龄': base_info['年龄'],
                '身高': base_info['身高'],
                '体重': base_info['体重'],
                '孕妇BMI': base_info['孕妇BMI'],
                'bmi_category': base_info['bmi_category'],
                '达标时间': event_time,
                '达标状态': event_observed
            })
        
        survival_df_error = pd.DataFrame(survival_data_error)
        
        # 对每个BMI分组计算带误差的最优时点
        for bmi_cat in bmi_categories:
            df_bmi_error = survival_df_error[survival_df_error['bmi_category'] == bmi_cat].copy()
            if df_bmi_error.empty:
                continue
            
            kmf_error = KaplanMeierFitter()
            kmf_error.fit(durations=df_bmi_error["达标时间"], event_observed=df_bmi_error["达标状态"])
            s_t_error = kmf_error.survival_function_.rename(columns={"KM_estimate": "s_t"})
            
            def objective_error(t):
                return calculate_utility(t, s_t_error)
            
            result_error = minimize_scalar(
                objective_error, bounds=(0, df_bmi_error["达标时间"].max()), method="bounded"
            )
            
            optimal_time_error = result_error.x
            optimal_utility_error = result_error.fun
            risk_error = 1 / optimal_utility_error if optimal_utility_error > 0 else float("inf")
            
            error_results[bmi_cat]["times"].append(optimal_time_error)
            error_results[bmi_cat]["utilities"].append(optimal_utility_error)
            error_results[bmi_cat]["risks"].append(risk_error)
    
    # 输出误差分析结果
    print("\n=== 检测误差影响分析结果 ===")
    error_analysis_summary = []
    
    for bmi_cat in bmi_categories:
        if error_results[bmi_cat]["times"]:
            times = np.array(error_results[bmi_cat]["times"])
            utilities = np.array(error_results[bmi_cat]["utilities"])
            risks = np.array(error_results[bmi_cat]["risks"])
            
            mean_time = times.mean()
            std_time = times.std()
            time_ci = norm.interval(0.95, loc=mean_time, scale=std_time/np.sqrt(len(times)))
            
            mean_utility = utilities.mean()
            std_utility = utilities.std()
            
            mean_risk = risks.mean()
            std_risk = risks.std()
            
            original_time = original_optimal_times.get(bmi_cat, 0)
            original_utility = calculate_utility(original_time, s_t) if bmi_cat in original_optimal_times else 0
            
            print(f"BMI分组 {bmi_cat}:")
            print(f"  原始最优时点: {original_time:.2f} 天")
            print(f"  模拟最优时点均值: {mean_time:.2f} 天")
            print(f"  标准差: {std_time:.2f} 天")
            print(f"  95%置信区间: ({time_ci[0]:.2f}, {time_ci[1]:.2f}) 天")
            print(f"  原始最小效用值: {original_utility:.4f}")
            print(f"  模拟最小效用值均值: {mean_utility:.4f}")
            print(f"  效用值标准差: {std_utility:.4f}")
            print(f"  模拟风险水平均值: {mean_risk:.4f}")
            print(f"  风险水平标准差: {std_risk:.4f}")
            
            # 可视化误差分析结果
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            plt.hist(times, bins=20, alpha=0.7, edgecolor="black")
            plt.axvline(mean_time, color="r", linestyle="--", label=f"模拟平均时点: {mean_time:.2f}")
            plt.axvline(original_time, color="b", linestyle="-", label=f"原始最优时点: {original_time:.2f}")
            plt.axvline(time_ci[0], color="g", linestyle=":", label="95%置信区间")
            plt.axvline(time_ci[1], color="g", linestyle=":")
            plt.xlabel("最优NIPT时点（天）")
            plt.ylabel("频数")
            plt.title(f"BMI分组 {bmi_cat} - 最优时点分布")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 2, 2)
            plt.hist(utilities, bins=20, alpha=0.7, edgecolor="black", color="orange")
            plt.axvline(mean_utility, color="r", linestyle="--", label=f"模拟平均效用值: {mean_utility:.4f}")
            plt.axvline(original_utility, color="b", linestyle="-", label=f"原始最小效用值: {original_utility:.4f}")
            plt.xlabel("最小效用值")
            plt.ylabel("频数")
            plt.title(f"BMI分组 {bmi_cat} - 最小效用值分布")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 2, 3)
            plt.scatter(times, utilities, alpha=0.6)
            plt.axvline(original_time, color="b", linestyle="-", label="原始最优时点")
            plt.axhline(original_utility, color="b", linestyle="-", label="原始最小效用值")
            plt.xlabel("最优时点（天）")
            plt.ylabel("最小效用值")
            plt.title(f"BMI分组 {bmi_cat} - 时点与效用值关系")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 2, 4)
            plt.hist(risks, bins=20, alpha=0.7, edgecolor="black", color="red")
            plt.axvline(mean_risk, color="r", linestyle="--", label=f"模拟平均风险: {mean_risk:.4f}")
            plt.xlabel("风险水平")
            plt.ylabel("频数")
            plt.title(f"BMI分组 {bmi_cat} - 风险水平分布")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            safe_bmi_cat = clean_filename(bmi_cat)
            filename = f"./python_code/error_analysis_BMI_{safe_bmi_cat}.png"
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            plt.close()
            
            # 保存模拟数据
            data_df = pd.DataFrame({
                "time": times, 
                "utility": utilities, 
                "risk": risks
            })
            csv_filename = f"./python_code/error_analysis_BMI_{safe_bmi_cat}.csv"
            data_df.to_csv(csv_filename, index=False)
            
            print(f"  图表已保存至: {filename}")
            print(f"  数据已保存至: {csv_filename}")
            
            error_analysis_summary.append({
                "BMI分组": bmi_cat,
                "原始最优时点": original_time,
                "模拟最优时点均值": mean_time,
                "最优时点标准差": std_time,
                "最优时点95%置信区间下限": time_ci[0],
                "最优时点95%置信区间上限": time_ci[1],
                "原始最小效用值": original_utility,
                "模拟最小效用值均值": mean_utility,
                "最小效用值标准差": std_utility,
                "模拟风险水平均值": mean_risk,
                "风险水平标准差": std_risk
            })
        else:
            print(f"BMI分组 {bmi_cat}: 无数据")
            error_analysis_summary.append({
                "BMI分组": bmi_cat,
                "原始最优时点": "无数据",
                "模拟最优时点均值": "无数据",
                "最优时点标准差": "无数据",
                "最优时点95%置信区间下限": "无数据",
                "最优时点95%置信区间上限": "无数据",
                "原始最小效用值": "无数据",
                "模拟最小效用值均值": "无数据",
                "最小效用值标准差": "无数据",
                "模拟风险水平均值": "无数据",
                "风险水平标准差": "无数据"
            })
    
    # 保存误差分析摘要
    error_summary_df = pd.DataFrame(error_analysis_summary)
    error_summary_df.to_excel("./python_code/error_analysis_summary.xlsx", index=False)
    print(f"\n误差分析摘要已保存至: ./python_code/error_analysis_summary.xlsx")
    
    return error_summary_df

results, equations, error_results = analyze_nipt_optimal_time_with_advanced_modeling()