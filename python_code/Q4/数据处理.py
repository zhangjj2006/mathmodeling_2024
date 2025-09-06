import pandas as pd
import numpy as np

def process_female_nipt_data(input_file_path, output_file_path):
    """
    处理女胎NIPT数据，进行质量控制和数据清洗，然后保存到新的Excel文件
    
    参数:
    input_file_path: 输入Excel文件路径
    output_file_path: 输出Excel文件路径
    """
    
    # 1. 读取数据 - 指定读取第二个工作表
    print("正在读取数据...")
    try:
        # 读取第二个工作表
        df = pd.read_excel(input_file_path, sheet_name=1)
        print(f"成功读取数据: {df.shape[0]} 行, {df.shape[1]} 列")
    except Exception as e:
        print(f"读取数据时出错: {e}")
        return
    
    # 2. 添加三种疾病的one-hot编码列
    print("\n正在添加疾病分类列...")
    
    # 确保"染色体的非整倍体"列存在
    if '染色体的非整倍体' not in df.columns:
        print("错误: 未找到'染色体的非整倍体'列")
        return
    
    # 创建三种疾病的标志列
    df['T21'] = 0  # 唐氏综合征 (21号染色体三体)
    df['T18'] = 0  # 爱德华氏综合征 (18号染色体三体)
    df['T13'] = 0  # 帕陶氏综合征 (13号染色体三体)
    
    # 根据"染色体的非整倍体"列设置疾病标志
    for idx, value in df['染色体的非整倍体'].items():
        if pd.isna(value) or value == '':
            continue
        
        # 根据内容设置相应的疾病标志
        if 'T21' in str(value) or '21' in str(value):
            df.at[idx, 'T21'] = 1
        if 'T18' in str(value) or '18' in str(value):
            df.at[idx, 'T18'] = 1
        if 'T13' in str(value) or '13' in str(value):
            df.at[idx, 'T13'] = 1
    
    # 统计疾病分布
    t21_count = df['T21'].sum()
    t18_count = df['T18'].sum()
    t13_count = df['T13'].sum()
    normal_count = len(df) - (t21_count + t18_count + t13_count)
    
    print(f"正常样本数: {normal_count}")
    print(f"T21 (唐氏综合征) 样本数: {t21_count}")
    print(f"T18 (爱德华氏综合征) 样本数: {t18_count}")
    print(f"T13 (帕陶氏综合征) 样本数: {t13_count}")
    
    # 3. 数据质量检查
    print("\n正在进行数据质量检查...")
    
    # 检查缺失值
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print("发现缺失值:")
        print(missing_values[missing_values > 0])
    else:
        print("未发现缺失值")
    
    # 检查重复行
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"发现 {duplicates} 个重复行，已移除")
        df = df.drop_duplicates()
    
    # 4. 数据质量控制 - 专门针对女胎数据
    print("\n正在进行数据质量控制...")
    initial_count = len(df)
    
    # 应用质量控制阈值
    qc_conditions = (
        (df['原始读段数'] > 3500000) &
        (df['在参考基因组上比对的比例'] > 0.75) &
        (df['重复读段的比例'] < 0.35) &
        (df['唯一比对的读段数']/df['原始读段数'] > 0.7) &
        (df['被过滤掉读段数的比例'] < 0.5) &
        # 对于女胎，X染色体浓度可能为负值，但其他质量指标正常即可
        (df['GC含量'].between(0.39, 0.6))  # GC含量应在正常范围内
        & (df['X染色体的Z值'] <= 3)
        & (df['X染色体的Z值'] >= -3)
    )
    
    # 筛选高质量数据
    hq_df = df[qc_conditions].copy()
    filtered_count = initial_count - len(hq_df)
    
    print(f"初始样本数: {initial_count}")
    print(f"高质量样本数: {len(hq_df)}")
    print(f"被过滤样本数: {filtered_count}")
    print(f"过滤比例: {filtered_count/initial_count*100:.2f}%")
    
    # 5. 保存处理后的数据
    print("\n正在保存处理后的数据...")
    try:
        hq_df.to_excel(output_file_path, index=False)
        print(f"处理后的数据已保存到: {output_file_path}")
        print("处理完成!")
        
    except Exception as e:
        print(f"保存数据时出错: {e}")



# 使用示例
if __name__ == "__main__":
    # 请修改为您的实际文件路径
    input_file = "./python_code/附件.xlsx"
    output_file = "./python_code/Q4/处理后的女胎NIPT数据.xlsx"
    
    process_female_nipt_data(input_file, output_file)

# 初始样本数: 605
# 高质量样本数: 556
# 被过滤样本数: 49
# 过滤比例: 8.10%