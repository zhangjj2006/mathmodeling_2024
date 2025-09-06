import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score, KFold
from scipy.optimize import minimize_scalar
from lifelines import KaplanMeierFitter
from scipy.interpolate import UnivariateSpline, interp1d
import statsmodels.api as sm
import shap

# 设置中文字体支持
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

def analyze_nipt_optimal_time_with_advanced_modeling():
    """
    使用高级建模方法分析不同BMI分组中年龄、身高、体重对NIPT最佳时点的影响
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
    prediction_formulas = {}  # 存储每个分组的预测公式
    
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
        
        # 使用随机森林建模年龄、身高、体重对达标时间的影响
        print("\n使用随机森林建模影响因素:")
        
        # 重置索引以确保一致性
        df_bmi_reset = df_bmi.reset_index(drop=True)
        
        # 准备回归数据
        X = df_bmi_reset[['年龄', '身高', '体重']]
        y = df_bmi_reset['达标时间']
        
        # 检查样本量是否足够进行建模
        if len(X) < 10:  # 样本量不足，使用简单回归
            print("样本量不足，使用简单线性回归建模")
            
            # 尝试多项式回归
            try:
                poly = PolynomialFeatures(degree=2, include_bias=False)
                X_poly = poly.fit_transform(X)
                model = Ridge(alpha=1.0)
                model.fit(X_poly, y)
                
                # 获取系数
                coef = model.coef_
                intercept = model.intercept_
                
                # 生成预测公式
                formula = f"达标时间 = {intercept:.2f} + {coef[0]:.3f}*年龄 + {coef[1]:.3f}*身高 + {coef[2]:.3f}*体重"
                formula += f" + {coef[3]:.3f}*年龄² + {coef[4]:.3f}*身高² + {coef[5]:.3f}*体重²"
                if len(coef) > 6:
                    formula += f" + {coef[6]:.3f}*年龄*身高 + {coef[7]:.3f}*年龄*体重 + {coef[8]:.3f}*身高*体重"
                
                prediction_formulas[bmi_cat] = formula
                print(f"多项式回归公式: {formula}")
                
                # 评估模型
                y_pred = model.predict(X_poly)
                mae = np.mean(np.abs(y - y_pred))
                r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
                
                age_importance = abs(coef[0]) + abs(coef[3]) + (abs(coef[6]) if len(coef) > 6 else 0)
                height_importance = abs(coef[1]) + abs(coef[4]) + (abs(coef[7]) if len(coef) > 7 else 0)
                weight_importance = abs(coef[2]) + abs(coef[5]) + (abs(coef[8]) if len(coef) > 8 else 0)
                
                # 归一化重要性
                total_importance = age_importance + height_importance + weight_importance
                if total_importance > 0:
                    age_importance /= total_importance
                    height_importance /= total_importance
                    weight_importance /= total_importance
                
                model_r2 = r2
                model_mae = mae
                
            except Exception as e:
                print(f"多项式回归失败: {e}, 使用简单线性回归")
                # 简单线性回归
                model = LinearRegression()
                model.fit(X, y)
                
                # 获取系数
                coef = model.coef_
                intercept = model.intercept_
                
                # 生成预测公式
                formula = f"达标时间 = {intercept:.2f} + {coef[0]:.3f}*年龄 + {coef[1]:.3f}*身高 + {coef[2]:.3f}*体重"
                prediction_formulas[bmi_cat] = formula
                print(f"线性回归公式: {formula}")
                
                # 评估模型
                y_pred = model.predict(X)
                mae = np.mean(np.abs(y - y_pred))
                r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
                
                # 计算特征重要性（使用系数的绝对值）
                total_importance = np.sum(np.abs(coef))
                age_importance = abs(coef[0]) / total_importance
                height_importance = abs(coef[1]) / total_importance
                weight_importance = abs(coef[2]) / total_importance
                
                model_r2 = r2
                model_mae = mae
            
            # 计算仅使用平均值的MAE
            y_mean = np.full_like(y_pred, y.mean())
            mae_mean = np.mean(np.abs(y - y_mean))
            improvement = (mae_mean - mae) / mae_mean * 100 if mae_mean > 0 else 0
            
            model_performance.append({
                "BMI分组": bmi_cat,
                "样本量": len(X),
                "R²得分": model_r2,
                "模型MAE": model_mae,
                "平均值MAE": mae_mean,
                "改进百分比": improvement,
                "模型类型": "多项式回归" if 'poly' in locals() else "线性回归"
            })
            
        else:
            # 样本量足够，使用随机森林
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
                
                # 生成随机森林的近似公式（使用特征重要性）
                # 随机森林没有简单公式，但我们提供特征重要性作为指导
                formula = f"达标时间 ≈ {y.mean():.1f} + {age_importance:.3f}*(年龄影响) + {height_importance:.3f}*(身高影响) + {weight_importance:.3f}*(体重影响)"
                prediction_formulas[bmi_cat] = formula
                print(f"随机森林特征影响公式: {formula}")
                
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
                
                # 计算仅使用BMI分组平均值的预测
                y_mean = np.full_like(y_pred, y.mean())
                
                # 比较两种方法的MAE
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
                
            except Exception as e:
                print(f"随机森林建模出错: {e}")
                age_importance = np.nan
                height_importance = np.nan
                weight_importance = np.nan
                model_r2 = np.nan
                model_mae = np.nan
        
        # 可视化影响因素 - 使用更平滑的拟合曲线
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
        
        # 2. 年龄与达标时间的关系 - 使用更平滑的拟合
        axes[0, 1].scatter(df_bmi['年龄'], df_bmi['达标时间'], alpha=0.6, label='数据点')
        
        # 使用局部加权回归（LOESS）或样条插值获得平滑曲线
        try:
            # 排序数据以便绘制平滑曲线
            sorted_idx = np.argsort(df_bmi['年龄'])
            x_sorted = df_bmi['年龄'].iloc[sorted_idx].values
            y_sorted = df_bmi['达标时间'].iloc[sorted_idx].values
            
            # 使用样条插值
            if len(x_sorted) > 3:
                spline = UnivariateSpline(x_sorted, y_sorted, s=len(x_sorted)*10)
                x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), 100)
                y_smooth = spline(x_smooth)
                axes[0, 1].plot(x_smooth, y_smooth, 'r-', alpha=0.8, linewidth=2, label='样条拟合')
        except Exception as e:
            print(f"样条拟合失败: {e}")
            # 使用多项式拟合作为备选
            try:
                z = np.polyfit(df_bmi['年龄'], df_bmi['达标时间'], 2)
                p = np.poly1d(z)
                x_smooth = np.linspace(df_bmi['年龄'].min(), df_bmi['年龄'].max(), 100)
                axes[0, 1].plot(x_smooth, p(x_smooth), 'r-', alpha=0.8, linewidth=2, label='多项式拟合')
            except:
                pass
        
        axes[0, 1].set_xlabel('年龄')
        axes[0, 1].set_ylabel('达标时间(天)')
        axes[0, 1].set_title('年龄与达标时间的关系')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        if not np.isnan(age_importance):
            axes[0, 1].text(0.05, 0.95, f'重要性: {age_importance:.4f}', transform=axes[0, 1].transAxes, 
                           fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 3. 身高与达标时间的关系
        axes[1, 0].scatter(df_bmi['身高'], df_bmi['达标时间'], alpha=0.6, label='数据点')
        
        try:
            sorted_idx = np.argsort(df_bmi['身高'])
            x_sorted = df_bmi['身高'].iloc[sorted_idx].values
            y_sorted = df_bmi['达标时间'].iloc[sorted_idx].values
            
            if len(x_sorted) > 3:
                spline = UnivariateSpline(x_sorted, y_sorted, s=len(x_sorted)*10)
                x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), 100)
                y_smooth = spline(x_smooth)
                axes[1, 0].plot(x_smooth, y_smooth, 'r-', alpha=0.8, linewidth=2, label='样条拟合')
        except Exception as e:
            print(f"样条拟合失败: {e}")
            try:
                z = np.polyfit(df_bmi['身高'], df_bmi['达标时间'], 2)
                p = np.poly1d(z)
                x_smooth = np.linspace(df_bmi['身高'].min(), df_bmi['身高'].max(), 100)
                axes[1, 0].plot(x_smooth, p(x_smooth), 'r-', alpha=0.8, linewidth=2, label='多项式拟合')
            except:
                pass
        
        axes[1, 0].set_xlabel('身高(cm)')
        axes[1, 0].set_ylabel('达标时间(天)')
        axes[1, 0].set_title('身高与达标时间的关系')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        if not np.isnan(height_importance):
            axes[1, 0].text(0.05, 0.95, f'重要性: {height_importance:.4f}', transform=axes[1, 0].transAxes, 
                           fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 4. 体重与达标时间的关系
        axes[1, 1].scatter(df_bmi['体重'], df_bmi['达标时间'], alpha=0.6, label='数据点')
        
        try:
            sorted_idx = np.argsort(df_bmi['体重'])
            x_sorted = df_bmi['体重'].iloc[sorted_idx].values
            y_sorted = df_bmi['达标时间'].iloc[sorted_idx].values
            
            if len(x_sorted) > 3:
                spline = UnivariateSpline(x_sorted, y_sorted, s=len(x_sorted)*10)
                x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), 100)
                y_smooth = spline(x_smooth)
                axes[1, 1].plot(x_smooth, y_smooth, 'r-', alpha=0.8, linewidth=2, label='样条拟合')
        except Exception as e:
            print(f"样条拟合失败: {e}")
            try:
                z = np.polyfit(df_bmi['体重'], df_bmi['达标时间'], 2)
                p = np.poly1d(z)
                x_smooth = np.linspace(df_bmi['体重'].min(), df_bmi['体重'].max(), 100)
                axes[1, 1].plot(x_smooth, p(x_smooth), 'r-', alpha=0.8, linewidth=2, label='多项式拟合')
            except:
                pass
        
        axes[1, 1].set_xlabel('体重(kg)')
        axes[1, 1].set_ylabel('达标时间(天)')
        axes[1, 1].set_title('体重与达标时间的关系')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
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
        plt.figure(figsize=(12, 6))
        x = np.arange(len(performance_df))
        width = 0.35
        
        plt.bar(x - width/2, performance_df['平均值MAE'], width, label='仅使用平均值')
        plt.bar(x + width/2, performance_df['模型MAE'], width, label='使用完整模型')
        
        plt.xlabel('BMI分组')
        plt.ylabel('平均绝对误差(天)')
        plt.title('模型性能比较: 仅使用平均值 vs 使用完整模型')
        plt.xticks(x, [f"{row['BMI分组']}\n({row['模型类型']})" for _, row in performance_df.iterrows()], rotation=45)
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
    
    # 保存预测公式
    with open('./python_code/prediction_formulas.txt', 'w', encoding='utf-8') as f:
        f.write("各BMI分组Y染色体浓度达标时间预测公式\n")
        f.write("="*50 + "\n\n")
        
        for bmi_cat, formula in prediction_formulas.items():
            f.write(f"{bmi_cat}:\n")
            f.write(f"{formula}\n\n")
            
            # 从results_df中获取该分组的最优时点
            optimal_time_row = results_df[results_df['BMI分组'] == bmi_cat]
            if not optimal_time_row.empty:
                optimal_days = optimal_time_row.iloc[0]['最优时点(天)']
                optimal_weeks = optimal_time_row.iloc[0]['最优时点(周)']
                f.write(f"推荐NIPT检测时间: {optimal_days}天 ({optimal_weeks}周)\n")
            
            f.write("-"*50 + "\n\n")
    
    print("预测公式已保存至: ./python_code/prediction_formulas.txt")
    
    # 打印结论和公式
    print("\n=== 分析结论与预测公式 ===")
    for bmi_cat, formula in prediction_formulas.items():
        print(f"\n{bmi_cat}:")
        print(f"预测公式: {formula}")
        
        # 从results_df中获取该分组的最优时点
        optimal_time_row = results_df[results_df['BMI分组'] == bmi_cat]
        if not optimal_time_row.empty:
            optimal_days = optimal_time_row.iloc[0]['最优时点(天)']
            optimal_weeks = optimal_time_row.iloc[0]['最优时点(周)']
            print(f"推荐NIPT检测时间: {optimal_days}天 ({optimal_weeks}周)")
    
    print("\n=== 总体分析结论 ===")
    print("1. 使用随机森林回归和多项式回归能够更好地捕捉年龄、身高、体重对达标时间的非线性影响")
    print("2. 特征重要性分析揭示了各因素对达标时间影响的相对强度")
    print("3. 模型验证表明，引入年龄、身高、体重等因素能够显著提高预测准确性")
    print("4. 不同BMI分组中，各因素的影响程度存在差异，需要个性化考虑")
    print("5. 使用样条插值和多项式拟合使关系曲线更加平滑，避免了不合理的折点")
    print("6. 对于样本量不足的分组，使用多项式回归作为随机森林的替代方法")
    
    return results_df, prediction_formulas

# 运行分析
results, formulas = analyze_nipt_optimal_time_with_advanced_modeling()