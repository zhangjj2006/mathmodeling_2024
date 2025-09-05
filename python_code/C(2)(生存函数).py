import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 确保已安装 lifelines 库: pip install lifelines
from lifelines import KaplanMeierFitter
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
def analyze_with_dynamic_utility():
    # --- 0. 定义临床效用函数参数 ---
    # 这个函数现在考虑了“结果未成熟期”
    IMMATURE_PERIOD_END = 4 * 7  # 前4周，结果可能不准
    OPTIMAL_WINDOW_END = 12 * 7  # 12周，最佳窗口结束
    MID_RISK_END = 27 * 7        # 27周，中风险期结束
    
    UTILITY_IMMATURE_START = 0.3 # 未成熟期起始效用
    UTILITY_OPTIMAL = 1.0        # 最佳窗口期效用
    UTILITY_MID_RISK = 0.5       # 中风险期效用
    UTILITY_HIGH_RISK = 0.1      # 高风险期效用

    def get_dynamic_clinical_utility(day):
        if day <= IMMATURE_PERIOD_END:
            # 在未成熟期内，效用从 UTILITY_IMMATURE_START 线性增长到 UTILITY_OPTIMAL
            return UTILITY_IMMATURE_START + (UTILITY_OPTIMAL - UTILITY_IMMATURE_START) * (day / IMMATURE_PERIOD_END)
        elif day <= OPTIMAL_WINDOW_END:
            return UTILITY_OPTIMAL
        elif day <= MID_RISK_END:
            return UTILITY_MID_RISK
        else:
            return UTILITY_HIGH_RISK

    # --- 1. 加载和准备数据 ---
    df_middle = pd.read_excel('./python_code/bmi_Y_middle_result.xlsx')
    df_cannot_test = pd.read_excel('./python_code/bmi_Y_cannot_test_result.xlsx')
    df_always_can_test = pd.read_excel('./python_code/bmi_Y_always_can_test_result.xlsx')
    
    df_all = pd.concat([
        df_middle.assign(category='middle'),
        df_cannot_test.assign(category='cannot'),
        df_always_can_test.assign(category='always_can')
    ], ignore_index=True)
    
    # **修复**: 初始化为浮点数以避免 FutureWarning
    df_all['duration'] = 0.0
    df_all['event_observed'] = 0

    # 为三类数据赋值
    df_all.loc[df_all['category'] == 'cannot', 'duration'] = df_all['最晚不达标天数']
    df_all.loc[df_all['category'] == 'cannot', 'event_observed'] = 0
    
    df_all.loc[df_all['category'] == 'middle', 'duration'] = df_all['预测达标天数']
    df_all.loc[df_all['category'] == 'middle', 'event_observed'] = 1
    
    # **修复**: 将 always_can 的 duration 设为 0.1 而不是 0，以解决 "0.00天" 问题
    df_all.loc[df_all['category'] == 'always_can', 'duration'] = 0.1
    df_all.loc[df_all['category'] == 'always_can', 'event_observed'] = 1

    # BMI 分组
    def categorize_bmi(bmi):
        if bmi < 30.17: return '<30.17'
        elif 30.17 <= bmi < 32.25: return '30.17-32.25'
        elif 32.25 <= bmi < 34.70: return '32.25-34.70'
        elif 34.70 <= bmi < 37.11: return '34.70-37.11'
        else: return '>37.11'
    
    df_all['bmi_category'] = df_all['BMI'].apply(categorize_bmi)
    bmi_categories = sorted(df_all['bmi_category'].unique())
    
    # --- 2. 循环处理每个BMI分组 ---
    for bmi_cat in bmi_categories:
        df_bmi = df_all[df_all['bmi_category'] == bmi_cat].copy()
        
        if df_bmi.empty: continue
            
        print(f"--- 正在分析 BMI 区间: {bmi_cat} ---")
        
        kmf = KaplanMeierFitter()
        kmf.fit(durations=df_bmi['duration'], event_observed=df_bmi['event_observed'])
        
        # --- 3. 计算预期临床效用 ---
        s_t = kmf.survival_function_.rename(columns={'KM_estimate': 's_t'})
        s_t_minus_1 = s_t.shift(1).fillna(1.0)
        pdf_df = (s_t_minus_1 - s_t).rename(columns={'s_t': 'prob_event_at_t'})
        
        pdf_df['utility'] = [get_dynamic_clinical_utility(day) for day in pdf_df.index]
        expected_utility = (pdf_df['prob_event_at_t'] * pdf_df['utility']).sum()
        
        print(f"\n综合评价指标:")
        print(f"  - 预期临床效用分数: {expected_utility:.4f} (平衡了时效性、风险和准确性)")

        # --- 4. 计算特定达标率的天数及评估 ---
        probabilities = [0.85, 0.90, 0.95]
        print("\n特定累积达标率对应的天数及评估:")
        
        # --- 5. 可视化 ---
        plt.figure(figsize=(14, 8))
        ax = plt.gca()
        
        (1 - kmf.survival_function_).plot(ax=ax, label=f'累积达标函数 F(t) ({bmi_cat})', color='seagreen', linewidth=2.5)
        
        # 绘制背景色块代表不同效用区域
        ax.axvspan(0, IMMATURE_PERIOD_END, facecolor='gray', alpha=0.15, label=f'结果未成熟期 (<{IMMATURE_PERIOD_END/7:.0f}周)')
        ax.axvspan(IMMATURE_PERIOD_END, OPTIMAL_WINDOW_END, facecolor='green', alpha=0.15, label='最佳窗口期')
        ax.axvspan(OPTIMAL_WINDOW_END, MID_RISK_END, facecolor='orange', alpha=0.15, label='中风险期')
        ax.axvspan(MID_RISK_END, df_all['duration'].max() + 10, facecolor='red', alpha=0.15, label='高风险期')

        # 重新计算并标记百分位点
        for prob in probabilities:
            try:
                day_value = kmf.percentile(prob)
                utility = get_dynamic_clinical_utility(day_value)
                ax.axvline(x=day_value, color='purple', linestyle='--', linewidth=1)
                ax.axhline(y=prob, color='purple', linestyle='--', linewidth=1)
                ax.text(day_value + 2, prob - 0.05, f'{int(prob*100)}% -> {day_value:.1f}天\n效用: {utility:.2f}', color='purple', fontsize=10)
                print(f"  - {int(prob*100)}% 达标天数: {day_value:.2f} 天 (动态效用: {utility:.2f})")
            except Exception:
                print(f"  - {int(prob*100)}% 的累积达标率在观测期内无法达到。")

        ax.text(0.98, 0.1, f'预期临床效用: {expected_utility:.3f}',
                transform=ax.transAxes, fontsize=12, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.8))

        ax.set_title(f'BMI区间 {bmi_cat} 达标情况的动态临床效用分析', fontsize=16)
        ax.set_xlabel('天数', fontsize=12)
        ax.set_ylabel('累积达标比例', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(0, 1.05)
        ax.set_xlim(-5, df_all['duration'].max() + 10)
        
        filename = f'./python_code/BMI_{bmi_cat.replace("<", "lt").replace(">", "gt").replace("-", "_")}_dynamic_utility_analysis.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n图表已保存至: {filename}")
        print("\n" + "="*50 + "\n")

# --- 运行主函数 ---
analyze_with_dynamic_utility()