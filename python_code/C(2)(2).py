import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import re


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

def generate_bmi_category_scatter_charts():
    # 读取之前生成的数据
    df_middle = pd.read_excel('./python_code/bmi_Y_middle_result.xlsx')
    df_cannot_test = pd.read_excel('./python_code/bmi_Y_cannot_test_result.xlsx')
    df_always_can_test = pd.read_excel('./python_code/bmi_Y_always_can_test_result.xlsx')
    
    # 添加分类标签
    df_middle['category'] = 'middle'
    df_cannot_test['category'] = 'cannot'
    df_always_can_test['category'] = 'always_can'
    
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
    
    # 定义分类及颜色
    colors = {'cannot': 'red', 'middle': 'yellow', 'always_can': 'green'}
    labels = {'cannot': '不能达标', 'middle': '中间达标', 'always_can': '始终达标'}
    
    # 为每个BMI区间生成散点图
    for bmi_cat in bmi_categories:
        plt.figure(figsize=(12, 8))
        
        # 处理不能达标的孕妇
        df_cannot = df_cannot_test[df_cannot_test['bmi_category'] == bmi_cat]
        if not df_cannot.empty:
            plt.scatter(df_cannot['最晚不达标天数'], [1]*len(df_cannot), 
                       c=colors['cannot'], label=labels['cannot'], alpha=0.7, s=30)
        
        # 处理中间达标的孕妇
        df_middle_cat = df_middle[df_middle['bmi_category'] == bmi_cat]
        if not df_middle_cat.empty:
            plt.scatter(df_middle_cat['预测达标天数'], [1]*len(df_middle_cat), 
                       c=colors['middle'], label=labels['middle'], alpha=0.7, s=30)
        
        # 处理始终达标的孕妇
        df_always = df_always_can_test[df_always_can_test['bmi_category'] == bmi_cat]
        if not df_always.empty:
            plt.scatter(df_always['最早达标天数'], [1]*len(df_always), 
                       c=colors['always_can'], label=labels['always_can'], alpha=0.7, s=30)
        
        # 设置图表属性
        plt.xlabel('天数', fontsize=12)
        plt.ylabel('人数(示意)', fontsize=12)
        plt.title(f'BMI区间 {bmi_cat} 的孕妇Y染色体达标情况分布', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # 调整y轴显示
        plt.ylim(0.5, 1.5)
        plt.yticks([])
        
        # 保存图表
        plt.savefig(f'./python_code/BMI_{bmi_cat.replace("<", "lt").replace(">", "gt")}_scatter.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 打印统计信息
        print(f"BMI区间 {bmi_cat} 统计:")
        print(f"  不能达标: {len(df_cannot)} 人")
        print(f"  中间达标: {len(df_middle_cat)} 人")
        print(f"  始终达标: {len(df_always)} 人")
        print()

if __name__ == "__main__":
    # 生成直方图
    generate_bmi_category_charts()
    
    # 生成散点图（更符合你的描述）
    generate_bmi_category_scatter_charts()
    
    print("图表已生成完成！")

