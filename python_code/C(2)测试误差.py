import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from scipy.optimize import minimize_scalar
from scipy.stats import norm

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def analyze_error_impact():
    # --- 定义效用函数（与主分析相同）---
    def get_rearly(t):
        return 2.0 * np.exp(-t/50)
    
    def get_rlate(t):
        if t < 84:
            return 0.1
        elif t <= 189:
            normalized_t = (t - 84) / (189 - 84)
            return 0.1 + 0.9 * (normalized_t ** 2)
        else:
            return 1.0

    def calculate_utility(t, survival_func):
        if t < 0:
            return float('inf')
        closest_time_idx = np.abs(survival_func.index - t).argmin()
        actual_time = survival_func.index[closest_time_idx]
        p_t = 1 - survival_func.loc[actual_time, 's_t']
        return (1 - p_t) * get_rearly(t) + p_t * get_rlate(t)

    # --- 加载数据 ---
    df_middle = pd.read_excel('./python_code/bmi_Y_middle_result.xlsx')
    df_cannot_test = pd.read_excel('./python_code/bmi_Y_cannot_test_result.xlsx')
    df_always_can_test = pd.read_excel('./python_code/bmi_Y_always_can_test_result.xlsx')
    
    df_all = pd.concat([
        df_middle.assign(category='middle'),
        df_cannot_test.assign(category='cannot'),
        df_always_can_test.assign(category='always_can')
    ], ignore_index=True)
    
    df_all['duration'] = 0.0
    df_all['event_observed'] = 0

    df_all.loc[df_all['category'] == 'cannot', 'duration'] = df_all['最晚不达标天数']
    df_all.loc[df_all['category'] == 'cannot', 'event_observed'] = 0
    
    df_all.loc[df_all['category'] == 'middle', 'duration'] = df_all['预测达标天数']
    df_all.loc[df_all['category'] == 'middle', 'event_observed'] = 1
    
    df_all.loc[df_all['category'] == 'always_can', 'duration'] = 0.1
    df_all.loc[df_all['category'] == 'always_can', 'event_observed'] = 1

    # BMI 分类函数
    def categorize_bmi(bmi):
        if bmi < 30.26:
            return '<30.26'
        elif 30.26 <= bmi < 32.30:
            return '30.26-32.30'
        elif 32.30 <= bmi < 34.92:
            return '32.30-34.92'
        elif 34.92 <= bmi < 39.49:
            return '34.92-39.49'
        else:
            return '>39.49'
    
    df_all['bmi_category'] = df_all['BMI'].apply(categorize_bmi)
    
    # 定义BMI区间
    bmi_categories = ['<30.26', '30.26-32.30', '32.30-34.92', '34.92-39.49', '>39.49']
    
    # 存储误差分析结果
    error_analysis_results = {}
    
    # 定义误差参数
    error_params = {
        'bmi_error_std': 0.5,  # BMI测量误差标准差
        'duration_error_std': 7,  # 达标天数测量误差标准差（天）
        'n_simulations': 200    # 蒙特卡洛模拟次数
    }
    
    for bmi_cat in bmi_categories:
        df_bmi = df_all[df_all['bmi_category'] == bmi_cat].copy()
        
        if df_bmi.empty:
            continue
            
        print(f"--- 分析 BMI 区间: {bmi_cat} 的误差影响 ---")
        
        # 原始分析（无误差）
        kmf = KaplanMeierFitter()
        kmf.fit(durations=df_bmi['duration'], event_observed=df_bmi['event_observed'])
        s_t = kmf.survival_function_.rename(columns={'KM_estimate': 's_t'})
        
        def objective(t):
            return calculate_utility(t, s_t)
        
        result = minimize_scalar(
            objective,
            bounds=(0, df_bmi['duration'].max()),
            method='bounded'
        )
        
        optimal_time_original = result.x
        optimal_utility_original = result.fun
        
        # 蒙特卡洛模拟（引入误差）
        optimal_times_simulated = []
        optimal_utilities_simulated = []
        
        for i in range(error_params['n_simulations']):
            # 复制数据并添加误差
            df_simulated = df_bmi.copy()
            
            # 添加BMI测量误差
            bmi_errors = np.random.normal(0, error_params['bmi_error_std'], len(df_simulated))
            df_simulated['BMI_simulated'] = df_simulated['BMI'] + bmi_errors
            
            # 添加达标天数测量误差
            duration_errors = np.random.normal(0, error_params['duration_error_std'], len(df_simulated))
            df_simulated['duration_simulated'] = df_simulated['duration'] + duration_errors
            df_simulated['duration_simulated'] = df_simulated['duration_simulated'].clip(lower=0)  # 确保非负
            
            # 重新进行生存分析
            kmf_simulated = KaplanMeierFitter()
            kmf_simulated.fit(
                durations=df_simulated['duration_simulated'], 
                event_observed=df_simulated['event_observed']
            )
            
            s_t_simulated = kmf_simulated.survival_function_.rename(columns={'KM_estimate': 's_t'})
            
            # 计算最优时点
            def objective_simulated(t):
                return calculate_utility(t, s_t_simulated)
            
            try:
                result_simulated = minimize_scalar(
                    objective_simulated,
                    bounds=(0, df_simulated['duration_simulated'].max()),
                    method='bounded'
                )
                
                optimal_times_simulated.append(result_simulated.x)
                optimal_utilities_simulated.append(result_simulated.fun)
            except:
                # 如果优化失败，跳过此次模拟
                continue
        
        # 分析模拟结果
        if optimal_times_simulated:
            optimal_times_simulated = np.array(optimal_times_simulated)
            optimal_utilities_simulated = np.array(optimal_utilities_simulated)
            
            # 计算统计量
            mean_optimal_time = np.mean(optimal_times_simulated)
            std_optimal_time = np.std(optimal_times_simulated)
            mean_optimal_utility = np.mean(optimal_utilities_simulated)
            
            # 计算偏差和相对偏差
            time_bias = mean_optimal_time - optimal_time_original
            time_relative_bias = time_bias / optimal_time_original * 100 if optimal_time_original > 0 else 0
            
            utility_bias = mean_optimal_utility - optimal_utility_original
            utility_relative_bias = utility_bias / optimal_utility_original * 100 if optimal_utility_original > 0 else 0
            
            # 存储结果
            error_analysis_results[bmi_cat] = {
                'original_time': optimal_time_original,
                'original_utility': optimal_utility_original,
                'mean_simulated_time': mean_optimal_time,
                'std_simulated_time': std_optimal_time,
                'mean_simulated_utility': mean_optimal_utility,
                'time_bias': time_bias,
                'time_relative_bias': time_relative_bias,
                'utility_bias': utility_bias,
                'utility_relative_bias': utility_relative_bias,
                'simulated_times': optimal_times_simulated,
                'simulated_utilities': optimal_utilities_simulated
            }
            
            # 打印结果
            print(f"原始最优时点: {optimal_time_original:.2f} 天")
            print(f"模拟平均最优时点: {mean_optimal_time:.2f} ± {std_optimal_time:.2f} 天")
            print(f"时点偏差: {time_bias:.2f} 天 ({time_relative_bias:.2f}%)")
            print(f"原始最小效用值: {optimal_utility_original:.4f}")
            print(f"模拟平均最小效用值: {mean_optimal_utility:.4f}")
            print(f"效用值偏差: {utility_bias:.4f} ({utility_relative_bias:.2f}%)")
            print()
            
            # 绘制误差分析图
            plt.figure(figsize=(12, 8))
            
            # 时点分布直方图
            plt.subplot(2, 2, 1)
            plt.hist(optimal_times_simulated, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            plt.axvline(optimal_time_original, color='red', linestyle='--', linewidth=2, label='原始最优时点')
            plt.axvline(mean_optimal_time, color='green', linestyle='--', linewidth=2, label='模拟平均时点')
            plt.xlabel('最优时点 (天)')
            plt.ylabel('频数')
            plt.title(f'BMI {bmi_cat} - 最优时点分布')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 效用值分布直方图
            plt.subplot(2, 2, 2)
            plt.hist(optimal_utilities_simulated, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
            plt.axvline(optimal_utility_original, color='red', linestyle='--', linewidth=2, label='原始最小效用值')
            plt.axvline(mean_optimal_utility, color='green', linestyle='--', linewidth=2, label='模拟平均效用值')
            plt.xlabel('最小效用值')
            plt.ylabel('频数')
            plt.title(f'BMI {bmi_cat} - 最小效用值分布')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 时点与效用值的关系
            plt.subplot(2, 2, 3)
            plt.scatter(optimal_times_simulated, optimal_utilities_simulated, alpha=0.6, color='purple')
            plt.axvline(optimal_time_original, color='red', linestyle='--', linewidth=1, alpha=0.7)
            plt.axhline(optimal_utility_original, color='red', linestyle='--', linewidth=1, alpha=0.7)
            plt.xlabel('最优时点 (天)')
            plt.ylabel('最小效用值')
            plt.title(f'BMI {bmi_cat} - 时点与效用值关系')
            plt.grid(True, alpha=0.3)
            
            # 风险变化分析
            plt.subplot(2, 2, 4)
            # 将时点转换为孕周
            gestational_weeks_original = optimal_time_original / 7
            gestational_weeks_simulated = optimal_times_simulated / 7
            
            # 计算风险（根据孕周）
            def calculate_risk(weeks):
                if weeks <= 12:
                    return 0.05  # 早期风险低
                elif weeks <= 27:
                    return 0.05 + 0.45 * (weeks - 12) / 15  # 中期风险增加
                else:
                    return 0.5 + 0.5 * min((weeks - 27) / 13, 1)  # 晚期风险极高
            
            risks_original = calculate_risk(gestational_weeks_original)
            risks_simulated = [calculate_risk(w) for w in gestational_weeks_simulated]
            
            plt.hist(risks_simulated, bins=20, alpha=0.7, color='orange', edgecolor='black')
            plt.axvline(risks_original, color='red', linestyle='--', linewidth=2, label='原始风险')
            plt.axvline(np.mean(risks_simulated), color='green', linestyle='--', linewidth=2, label='模拟平均风险')
            plt.xlabel('风险水平')
            plt.ylabel('频数')
            plt.title(f'BMI {bmi_cat} - 风险水平分布')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            filename = f'./python_code/error_analysis_BMI_{bmi_cat.replace("<", "lt").replace(">", "gt").replace("-", "_")}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"误差分析图表已保存至: {filename}")
        else:
            print(f"BMI {bmi_cat} 的模拟失败，无法进行误差分析")
        
        print("\n" + "="*50 + "\n")
    
    # 综合误差分析结果
    print("=== 综合误差分析结果 ===")
    for bmi_cat, results in error_analysis_results.items():
        print(f"BMI {bmi_cat}:")
        print(f"  时点偏差: {results['time_bias']:.2f} 天 ({results['time_relative_bias']:.2f}%)")
        print(f"  效用值偏差: {results['utility_bias']:.4f} ({results['utility_relative_bias']:.2f}%)")
        print(f"  时点变异系数: {results['std_simulated_time']/results['mean_simulated_time']*100:.2f}%")
        print()
    
    # 绘制综合比较图
    plt.figure(figsize=(14, 10))
    
    # 时点偏差比较
    plt.subplot(2, 2, 1)
    bmi_cats = list(error_analysis_results.keys())
    time_biases = [error_analysis_results[cat]['time_bias'] for cat in bmi_cats]
    time_rel_biases = [error_analysis_results[cat]['time_relative_bias'] for cat in bmi_cats]
    
    x = np.arange(len(bmi_cats))
    width = 0.35
    
    plt.bar(x - width/2, time_biases, width, label='绝对偏差 (天)')
    plt.bar(x + width/2, time_rel_biases, width, label='相对偏差 (%)', alpha=0.7)
    plt.xlabel('BMI 分组')
    plt.ylabel('偏差')
    plt.title('各BMI分组时点偏差比较')
    plt.xticks(x, bmi_cats)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 效用值偏差比较
    plt.subplot(2, 2, 2)
    utility_biases = [error_analysis_results[cat]['utility_bias'] for cat in bmi_cats]
    utility_rel_biases = [error_analysis_results[cat]['utility_relative_bias'] for cat in bmi_cats]
    
    plt.bar(x - width/2, utility_biases, width, label='绝对偏差')
    plt.bar(x + width/2, utility_rel_biases, width, label='相对偏差 (%)', alpha=0.7)
    plt.xlabel('BMI 分组')
    plt.ylabel('偏差')
    plt.title('各BMI分组效用值偏差比较')
    plt.xticks(x, bmi_cats)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 时点变异系数比较
    plt.subplot(2, 2, 3)
    time_cvs = [error_analysis_results[cat]['std_simulated_time']/error_analysis_results[cat]['mean_simulated_time']*100 
               for cat in bmi_cats]
    
    plt.bar(x, time_cvs, width, color='green', alpha=0.7)
    plt.xlabel('BMI 分组')
    plt.ylabel('变异系数 (%)')
    plt.title('各BMI分组时点变异系数比较')
    plt.xticks(x, bmi_cats)
    plt.grid(True, alpha=0.3)
    
    # 风险增加比较
    plt.subplot(2, 2, 4)
    risk_increases = []
    for cat in bmi_cats:
        # 将时点转换为孕周
        weeks_original = error_analysis_results[cat]['original_time'] / 7
        weeks_simulated = error_analysis_results[cat]['mean_simulated_time'] / 7
        
        # 计算风险函数
        def calculate_risk(weeks):
            if weeks <= 12:
                return 0.05
            elif weeks <= 27:
                return 0.05 + 0.45 * (weeks - 12) / 15
            else:
                return 0.5 + 0.5 * min((weeks - 27) / 13, 1)
        
        risk_original = calculate_risk(weeks_original)
        risk_simulated = calculate_risk(weeks_simulated)
        risk_increases.append((risk_simulated - risk_original) / risk_original * 100)
    
    plt.bar(x, risk_increases, width, color='red', alpha=0.7)
    plt.xlabel('BMI 分组')
    plt.ylabel('风险增加 (%)')
    plt.title('各BMI分组风险增加比较')
    plt.xticks(x, bmi_cats)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./python_code/error_analysis_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("综合误差分析图表已保存至: ./python_code/error_analysis_summary.png")
    
    return error_analysis_results

# 运行误差分析
error_results = analyze_error_impact()