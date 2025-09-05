import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams
from scipy import stats
import re

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def categorize_bmi(bmi):
    """根据BMI值分类"""
    if bmi < 30:
        return '<30'
    elif 30 <= bmi < 32:
        return '30-32'
    elif 32 <= bmi < 34:
        return '32-34'
    elif 34 <= bmi < 36:
        return '34-36'
    else:
        return '>36'

def normality_test_and_plot_with_qq():
    # 读取之前生成的数据
    df_middle = pd.read_excel('./python_code/bmi_Y_middle_result.xlsx')
    df_cannot_test = pd.read_excel('./python_code/bmi_Y_cannot_test_result.xlsx')
    df_always_can_test = pd.read_excel('./python_code/bmi_Y_always_can_test_result.xlsx')
    
    # 根据BMI值分类
    df_middle['bmi_category'] = df_middle['BMI'].apply(categorize_bmi)
    df_cannot_test['bmi_category'] = df_cannot_test['BMI'].apply(categorize_bmi)
    df_always_can_test['bmi_category'] = df_always_can_test['BMI'].apply(categorize_bmi)
    
    # 定义BMI区间
    bmi_categories = ['<30', '30-32', '32-34', '34-36', '>36']
    
    # 定义分类及颜色
    colors = {'cannot': 'red', 'middle': 'yellow', 'always_can': 'green'}
    labels = {'cannot': '不能达标', 'middle': '中间达标', 'always_can': '始终达标'}
    
    # 为每个BMI区间进行正态分布检验和绘图
    for bmi_cat in bmi_categories:
        # 筛选当前BMI区间的數據
        df_cannot = df_cannot_test[df_cannot_test['bmi_category'] == bmi_cat]
        df_middle_cat = df_middle[df_middle['bmi_category'] == bmi_cat]
        df_always = df_always_can_test[df_always_can_test['bmi_category'] == bmi_cat]
        
        # 收集所有天数数据（不区分分类）
        days_cannot = df_cannot['最晚不达标天数'].tolist() if not df_cannot.empty else []
        days_middle = df_middle_cat['预测达标天数'].tolist() if not df_middle_cat.empty else []
        days_always = df_always['最早达标天数'].tolist() if not df_always.empty else []
        
        # 合并所有数据
        all_days = days_cannot + days_middle + days_always
        
        if not all_days:
            print(f"警告: BMI区间 {bmi_cat} 没有数据")
            continue
            
        # 转换为numpy数组
        data = np.array(all_days)
        
        print(f"\n=== BMI区间 {bmi_cat} 的正态分布检验 ===")
        print(f"样本数量: {len(data)}")
        
        # Shapiro-Wilk检验（适用于小样本，n<5000）
        if len(data) <= 5000:
            try:
                shapiro_stat, shapiro_p = stats.shapiro(data)
                print(f"Shapiro-Wilk检验: 统计量={shapiro_stat:.6f}, p值={shapiro_p:.6f}")
                if shapiro_p > 0.05:
                    print("  结论: 数据符合正态分布 (p > 0.05)")
                else:
                    print("  结论: 数据不符合正态分布 (p <= 0.05)")
            except Exception as e:
                print(f"Shapiro-Wilk检验失败: {e}")
        else:
            print("Shapiro-Wilk检验: 样本量过大(>5000)，跳过该检验")
        
        # Jarque-Bera检验
        try:
            jb_stat, jb_p = stats.jarque_bera(data)
            print(f"Jarque-Bera检验: 统计量={jb_stat:.6f}, p值={jb_p:.6f}")
            if jb_p > 0.05:
                print("  结论: 数据符合正态分布 (p > 0.05)")
            else:
                print("  结论: 数据不符合正态分布 (p <= 0.05)")
        except Exception as e:
            print(f"Jarque-Bera检验失败: {e}")
        
        # 计算数据的基本统计信息
        mean = np.mean(data)
        std = np.std(data)
        print(f"数据均值: {mean:.2f}, 标准差: {std:.2f}")
        
        # 创建图表：直方图 + 正态分布曲线
        plt.figure(figsize=(12, 8))
        
        # 绘制分类数据的堆叠直方图
        data_to_plot = []
        plot_labels = []
        plot_colors = []
        
        if days_cannot:
            data_to_plot.append(days_cannot)
            plot_labels.append(labels['cannot'])
            plot_colors.append(colors['cannot'])
            
        if days_middle:
            data_to_plot.append(days_middle)
            plot_labels.append(labels['middle'])
            plot_colors.append(colors['middle'])
            
        if days_always:
            data_to_plot.append(days_always)
            plot_labels.append(labels['always_can'])
            plot_colors.append(colors['always_can'])
        
        # 绘制堆叠直方图
        if data_to_plot:
            n, bins, patches = plt.hist(data_to_plot, bins=20, stacked=True, 
                                      color=plot_colors, label=plot_labels, 
                                      alpha=0.7, edgecolor='black', linewidth=0.3)
        
        # 如果数据符合正态分布，则绘制正态分布曲线
        is_normal = False
        if len(data) <= 5000 and 'shapiro_p' in locals() and shapiro_p > 0.05:
            is_normal = True
        elif len(data) > 5000 and 'jb_p' in locals() and jb_p > 0.05:
            is_normal = True
        elif len(data) <= 5000 and 'shapiro_p' not in locals() and 'jb_p' in locals() and jb_p > 0.05:
            is_normal = True
            
        if is_normal:
            # 生成正态分布曲线的x值
            x = np.linspace(min(data), max(data), 1000)
            # 计算正态分布的y值
            y = stats.norm.pdf(x, mean, std)
            # 调整y值使其与直方图比例一致
            bin_width = bins[1] - bins[0]
            y_scaled = y * len(data) * bin_width
            
            # 绘制正态分布曲线
            plt.plot(x, y_scaled, 'b-', linewidth=2, 
                    label=f'正态分布拟合 (μ={mean:.1f}, σ={std:.1f})')
            
            print("  已在图中绘制正态分布拟合曲线")
        else:
            print("  数据不符合正态分布，未绘制正态分布曲线")
        
        # 设置图表属性
        plt.xlabel('天数', fontsize=14, fontweight='bold')
        plt.ylabel('人数', fontsize=14, fontweight='bold')
        plt.title(f'BMI区间 {bmi_cat} 的孕妇Y染色体达标情况分布及正态分布检验', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 设置中文字体
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        
        # 保存图表
        plt.tight_layout()
        normal_suffix = "_normal" if is_normal else "_non_normal"
        plt.savefig(f'./python_code/BMI_{bmi_cat.replace("<", "lt").replace(">", "gt")}_stacked_with_normal_test{normal_suffix}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 生成Q-Q图
        plt.figure(figsize=(10, 8))
        
        # 创建Q-Q图
        stats.probplot(data, dist="norm", plot=plt)
        plt.title(f'BMI区间 {bmi_cat} 的Q-Q图', fontsize=16, fontweight='bold')
        plt.xlabel('理论分位数', fontsize=14)
        plt.ylabel('样本分位数', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # 设置中文字体
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        
        # 添加直线方程和R²值
        # 计算线性拟合
        theoretical_quantiles, sample_quantiles = stats.probplot(data, dist="norm")
        slope, intercept, r_value, p_value, std_err = stats.linregress(theoretical_quantiles[0], theoretical_quantiles[1])
        r_squared = r_value ** 2
        
        # 在图上添加文本
        plt.text(0.05, 0.95, f'R² = {r_squared:.4f}', transform=plt.gca().transAxes, 
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'./python_code/BMI_{bmi_cat.replace("<", "lt").replace(">", "gt")}_qq_plot.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  已生成Q-Q图 (R² = {r_squared:.4f})")
        
        # 打印详细统计信息
        print(f"BMI区间 {bmi_cat} 详细统计:")
        print(f"  不能达标: {len(df_cannot)} 人")
        print(f"  中间达标: {len(df_middle_cat)} 人")
        print(f"  始终达标: {len(df_always)} 人")
        print(f"  总计: {len(data)} 人")

if __name__ == "__main__":
    # 执行正态分布检验和绘图（包括Q-Q图）
    normality_test_and_plot_with_qq()
    
    print("\n正态分布检验、直方图和Q-Q图生成完成！")