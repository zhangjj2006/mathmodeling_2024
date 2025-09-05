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

if __name__ == "__main__":
    best_bmi_Y()
