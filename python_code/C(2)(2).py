import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams
from scipy import stats
import re


plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
def best_bmi_Y():# 将y染色体达标时候的孕妇bmi挑出来
    def weeks_to_days(weeks_str):
        match = re.match(r'(\d+)[wW](?:\+(\d+))?', str(weeks_str), re.IGNORECASE)
        if match:
            weeks = int(match.group(1))
            days = int(match.group(2)) if match.group(2) else 0
            return weeks * 7 + days
        else:
            print(f"无法解析孕周格式: {weeks_str}")
            return None
    df = pd.read_excel('./python_code/附件.xlsx',sheet_name=0)
    df = df[['孕妇代码', '检测孕周','孕妇BMI', 'Y染色体浓度']]
    bmi_avg = df.groupby('孕妇代码')['孕妇BMI'].mean()
    df['孕妇平均BMI'] = df['孕妇代码'].map(bmi_avg)
    df['检测孕周_天数'] = df['检测孕周'].apply(weeks_to_days)

    # print(df)
    code_col = '孕妇代码'
    day_col = '检测孕周_天数'
    bmi_col = '孕妇平均BMI'
    y_col = 'Y染色体浓度'
    # df_can_test_codes = df[df[y_col] >= 0.04][code_col].unique() # not left
    # df_cannot_test_codes = df[~df[code_col].isin(df_can_test_codes)][code_col].unique()  # left
    # df_codes_cannot_test_before_codes = df[df[y_col] < 0.04][code_col].unique() # not right
    # df_always_can_test_codes = df[~df[code_col].isin(df_codes_cannot_test_before_codes)][code_col].unique() # right
    # df_middle_codes = np.setdiff1d(df_can_test_codes, df_always_can_test_codes)
    df_codes = df[code_col].unique()

    result_middle = []
    result_cannot_test = []
    result_always_can_test = []
    for code in df_codes:
        # 获取该孕妇的所有数据并按检测时间排序
        data = df[df[code_col] == code].sort_values(by=day_col)
        mean_bmi = data[bmi_col].mean()
        # 找到第一次Y染色体浓度≥4%的检测时间
        first_ok_idx = None
        for idx, row in data.iterrows():
            if row[y_col] >= 0.04:
                first_ok_idx = idx
                break
        
        if first_ok_idx is None:
            result_cannot_test.append({code_col: code, 'BMI': mean_bmi, '最晚不达标天数': data[day_col].max()})
            continue
        else:
            later_data = data.loc[first_ok_idx:]
            if (later_data[y_col] < 0.04).any():
                continue  # 如果有任何后续记录<0.04，跳过该孕妇
            else:
                if first_ok_idx == data.index[0]:
                    result_always_can_test.append({code_col: code, 'BMI': mean_bmi, '最早达标天数': data[day_col].min()})
                else:
                    ok_row = data.loc[first_ok_idx]
                    prev_row = data.loc[data.index[data.index.get_loc(first_ok_idx) - 1]]    
                    # 计算权重
                    w1 = abs(ok_row[y_col] - 0.04)
                    w2 = abs(prev_row[y_col] - 0.04)
                    predicted_day = (w2 * ok_row[day_col] + w1 * prev_row[day_col]) / (w1 + w2)
                    result_middle.append({code_col: code, 'BMI': mean_bmi, '预测达标天数': predicted_day})

    df_middle = pd.DataFrame(result_middle)
    df_cannot_test = pd.DataFrame(result_cannot_test)
    df_always_can_test = pd.DataFrame(result_always_can_test)

    df_middle.to_excel('./python_code/bmi_Y_middle_result.xlsx', index=False)
    df_cannot_test.to_excel('./python_code/bmi_Y_cannot_test_result.xlsx', index=False)
    df_always_can_test.to_excel('./python_code/bmi_Y_always_can_test_result.xlsx', index=False)

    # for code in df_middle_codes:
    #     data = df[df[code_col] == code]

    #     ok_data = data[data[y_col] >= 0.04]
    #     min_day = ok_data[week_col].min()
    #     data_before_ok = data[data[week_col] < min_day]
    #     if not data_before_ok.empty:
    #         df_cannot_test_codes = np.append(df_cannot_test_codes, code)
    #     else:
    #         df_always_can_test_codes = np.append(df_always_can_test_codes, code)
   


    # df_first = df_filtered.groupby('孕妇代码').first().reset_index()
    # print(df_first)
    # df_result = df_first[df_first['检测孕周_天数'] <= 100].copy()

    # df_result.to_excel('./python_code/bmi_Y_result.xlsx', index=False)


def generate_bmi_category_charts():
    # 读取之前生成的数据
    df_middle = pd.read_excel('./python_code/bmi_Y_middle_result.xlsx')
    df_cannot_test = pd.read_excel('./python_code/bmi_Y_cannot_test_result.xlsx')
    df_always_can_test = pd.read_excel('./python_code/bmi_Y_always_can_test_result.xlsx')
    
    # 添加分类标签
    df_middle['category'] = 'middle'
    df_cannot_test['category'] = 'cannot'
    df_always_can_test['category'] = 'always_can'
    
    # 合并所有数据
    df_all = pd.concat([df_middle, df_cannot_test, df_always_can_test], ignore_index=True)
    
    # 根据BMI值分类
    def categorize_bmi(bmi):
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
    
    df_all['bmi_category'] = df_all['BMI'].apply(categorize_bmi)
    
    # 定义BMI区间
    bmi_categories = ['<30', '30-32', '32-34', '34-36', '>36']
    
    # 定义分类及颜色
    categories = ['cannot', 'middle', 'always_can']
    colors = {'cannot': 'red', 'middle': 'yellow', 'always_can': 'green'}
    labels = {'cannot': '不能达标', 'middle': '中间达标', 'always_can': '始终达标'}
    
    # 为每个BMI区间生成图表
    for bmi_cat in bmi_categories:
        # 筛选当前BMI区间的數據
        df_bmi = df_all[df_all['bmi_category'] == bmi_cat]
        
        if df_bmi.empty:
            print(f"警告: BMI区间 {bmi_cat} 没有数据")
            continue
            
        # 创建图表
        plt.figure(figsize=(12, 8))
        
        # 对于每个分类，绘制直方图
        for category in categories:
            df_category = df_bmi[df_bmi['category'] == category]
            
            if not df_category.empty:
                # 根据不同类别使用不同的天数列
                if category == 'cannot':
                    days = df_category['最晚不达标天数']
                elif category == 'middle':
                    days = df_category['预测达标天数']
                else:  # always_can
                    days = df_category['最早达标天数']
                
                # 绘制直方图
                plt.hist(days, bins=20, alpha=0.7, color=colors[category], 
                        label=labels[category], edgecolor='black', linewidth=0.5)
        
        # 设置图表属性
        plt.xlabel('天数', fontsize=12)
        plt.ylabel('人数', fontsize=12)
        plt.title(f'BMI区间 {bmi_cat} 的孕妇Y染色体达标情况分布', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # 保存图表
        plt.savefig(f'./python_code/BMI_{bmi_cat.replace("<", "lt").replace(">", "gt")}_distribution.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 打印统计信息
        print(f"BMI区间 {bmi_cat} 统计:")
        for category in categories:
            count = len(df_bmi[df_bmi['category'] == category])
            print(f"  {labels[category]}: {count} 人")
        print()

def generate_bmi_mixed_charts():

    # 读取之前生成的数据
    df_middle = pd.read_excel('./python_code/bmi_Y_middle_result.xlsx')
    df_cannot_test = pd.read_excel('./python_code/bmi_Y_cannot_test_result.xlsx')
    df_always_can_test = pd.read_excel('./python_code/bmi_Y_always_can_test_result.xlsx')
    
    # 添加分类标签
    df_middle['category'] = 'middle'
    df_cannot_test['category'] = 'cannot'
    df_always_can_test['category'] = 'always_can'
    
    # 合并所有数据
    df_all = pd.concat([df_middle, df_cannot_test, df_always_can_test], ignore_index=True)
    
    # 根据BMI值分类
    def categorize_bmi(bmi):
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
    
    df_all['bmi_category'] = df_all['BMI'].apply(categorize_bmi)
    
    # 定义BMI区间
    bmi_categories = ['<30', '30-32', '32-34', '34-36', '>36']
    
    # 定义分类及颜色
    categories = ['cannot', 'middle', 'always_can']
    colors = {'cannot': 'red', 'middle': 'yellow', 'always_can': 'green'}
    labels = {'cannot': '不能达标', 'middle': '中间达标', 'always_can': '始终达标'}
    
    # 为每个BMI区间生成图表
    for bmi_cat in bmi_categories:
        # 筛选当前BMI区间的數據
        df_bmi = df_all[df_all['bmi_category'] == bmi_cat]
        
        if df_bmi.empty:
            print(f"警告: BMI区间 {bmi_cat} 没有数据")
            continue
            
        # 创建图表和双坐标轴
        fig, ax1 = plt.subplots(figsize=(12, 8))
        
        # 收集所有天数数据（不区分类别）
        all_days = []
        
        # 对于每个分类，绘制直方图并收集数据
        for category in categories:
            df_category = df_bmi[df_bmi['category'] == category]
            
            if not df_category.empty:
                # 根据不同类别使用不同的天数列
                if category == 'cannot':
                    days = df_category['最晚不达标天数']
                elif category == 'middle':
                    days = df_category['预测达标天数']
                else:  # always_can
                    days = df_category['最早达标天数']
                
                # 绘制直方图
                ax1.hist(days, bins=20, alpha=0.7, color=colors[category], 
                        label=labels[category], edgecolor='black', linewidth=0.5)
                
                # 收集所有天数数据
                all_days.extend(days.tolist())
        
        # 设置左侧y轴标签
        ax1.set_xlabel('天数', fontsize=12)
        ax1.set_ylabel('人数', fontsize=12)
        ax1.set_title(f'BMI区间 {bmi_cat} 的孕妇Y染色体达标情况分布', fontsize=14)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 如果有足够的数据点，绘制总体概率密度曲线
        if len(all_days) > 1:
            # 创建第二个y轴，用于概率密度曲线
            ax2 = ax1.twinx()
            
            # 计算总体核密度估计
            try:
                kde = stats.gaussian_kde(all_days)
                x_range = np.linspace(min(all_days), max(all_days), 1000)
                kde_values = kde(x_range)
                
                # 绘制总体概率密度曲线
                ax2.plot(x_range, kde_values, color='blue', linewidth=2, linestyle='-', 
                        label='总体概率密度')
                
                # 设置右侧y轴标签
                ax2.set_ylabel('概率密度', fontsize=12)
                ax2.legend(loc='upper right', fontsize=10)
            except Exception as e:
                print(f"BMI区间 {bmi_cat} 计算概率密度时出错: {e}")
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        plt.savefig(f'./python_code/BMI_{bmi_cat.replace("<", "lt").replace(">", "gt")}_distribution.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 打印统计信息
        print(f"BMI区间 {bmi_cat} 统计:")
        total_count = len(df_bmi)
        for category in categories:
            count = len(df_bmi[df_bmi['category'] == category])
            percentage = (count / total_count) * 100 if total_count > 0 else 0
            print(f"  {labels[category]}: {count} 人 ({percentage:.1f}%)")
        print(f"  总计: {total_count} 人")
        print()

def generate_stacked_bmi_charts_alternative():
    # 读取之前生成的数据
    df_middle = pd.read_excel('./python_code/bmi_Y_middle_result.xlsx')
    df_cannot_test = pd.read_excel('./python_code/bmi_Y_cannot_test_result.xlsx')
    df_always_can_test = pd.read_excel('./python_code/bmi_Y_always_can_test_result.xlsx')
    
    # 根据BMI值分类
    def categorize_bmi(bmi):
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
    
    df_middle['bmi_category'] = df_middle['BMI'].apply(categorize_bmi)
    df_cannot_test['bmi_category'] = df_cannot_test['BMI'].apply(categorize_bmi)
    df_always_can_test['bmi_category'] = df_always_can_test['BMI'].apply(categorize_bmi)
    
    # 定义BMI区间
    bmi_categories = ['<30', '30-32', '32-34', '34-36', '>36']
    
    # 定义分类及颜色（堆叠顺序：底部到顶部为 red -> yellow -> green）
    colors = {'cannot': 'red', 'middle': 'yellow', 'always_can': 'green'}
    labels = {'cannot': '不能达标', 'middle': '中间达标', 'always_can': '始终达标'}
    category_order = ['cannot', 'middle', 'always_can']
    
    # 为每个BMI区间生成堆叠直方图
    for bmi_cat in bmi_categories:
        # 筛选当前BMI区间的數據
        df_cannot = df_cannot_test[df_cannot_test['bmi_category'] == bmi_cat]
        df_middle_cat = df_middle[df_middle['bmi_category'] == bmi_cat]
        df_always = df_always_can_test[df_always_can_test['bmi_category'] == bmi_cat]
        
        # 检查是否有数据
        if df_cannot.empty and df_middle_cat.empty and df_always.empty:
            print(f"警告: BMI区间 {bmi_cat} 没有数据")
            continue
            
        # 创建图表
        plt.figure(figsize=(12, 8))
        
        # 收集所有天数数据
        days_cannot = df_cannot['最晚不达标天数'].tolist() if not df_cannot.empty else []
        days_middle = df_middle_cat['预测达标天数'].tolist() if not df_middle_cat.empty else []
        days_always = df_always['最早达标天数'].tolist() if not df_always.empty else []
        
        # 确定bins
        all_days = days_cannot + days_middle + days_always
        if not all_days:
            print(f"警告: BMI区间 {bmi_cat} 没有有效数据")
            continue
            
        bins = np.linspace(min(all_days), max(all_days), 21)  # 20个bins
        
        # 按照堆叠顺序准备数据（底部到顶部：cannot -> middle -> always_can）
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
            plt.hist(data_to_plot, bins=bins, stacked=True, 
                    color=plot_colors, label=plot_labels, 
                    alpha=0.7, edgecolor='black', linewidth=0.3)
        
        # 设置图表属性
        plt.xlabel('天数', fontsize=14, fontweight='bold')
        plt.ylabel('人数', fontsize=14, fontweight='bold')
        plt.title(f'BMI区间 {bmi_cat} 的孕妇Y染色体达标情况分布', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 设置中文字体
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        
        # 保存图表
        plt.tight_layout()
        plt.savefig(f'./python_code/BMI_{bmi_cat.replace("<", "lt").replace(">", "gt")}_stacked_v2.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 打印统计信息
        print(f"BMI区间 {bmi_cat} 统计:")
        print(f"  不能达标: {len(df_cannot)} 人")
        print(f"  中间达标: {len(df_middle_cat)} 人")
        print(f"  始终达标: {len(df_always)} 人")
        print(f"  总计: {len(df_cannot) + len(df_middle_cat) + len(df_always)} 人")
        print()


# plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
# plt.rcParams['axes.unicode_minus'] = False

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

def normality_test_and_plot():
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
        
        # 打印详细统计信息
        print(f"BMI区间 {bmi_cat} 详细统计:")
        print(f"  不能达标: {len(df_cannot)} 人")
        print(f"  中间达标: {len(df_middle_cat)} 人")
        print(f"  始终达标: {len(df_always)} 人")
        print(f"  总计: {len(data)} 人")

if __name__ == "__main__":
    # 执行正态分布检验和绘图
    normality_test_and_plot()
    
    print("\n正态分布检验和图表生成完成！")
    # 生成直方图
    # generate_bmi_category_charts()
    # generate_stacked_bmi_charts_alternative()
    # generate_bmi_mixed_charts()
    # 生成散点图（更符合你的描述）
    # generate_bmi_category_scatter_charts()