import pandas as pd


def process_pregnancy_data():
    # ---------- 1. 读取Excel文件 ----------
    excel_path = r"C:\Users\pc\Desktop\1\C题\附件.xlsx"  # 替换为你的Excel文件路径（用r字符串避免转义）
    df = pd.read_excel(excel_path)  # 若有多个工作表，可加sheet_name="Sheet1"

    # ---------- 2. 按“孕妇代码”分组，计算平均值 ----------
    # groupby：按“孕妇代码”分组
    # agg：对“孕周BMI”和“Y染色体浓度”分别求均值
    # reset_index：将“孕妇代码”从“分组索引”转为普通列
    result = (
        df.groupby("孕妇代码")
        .agg(
            {
                "孕妇BMI": "mean",  # 对“孕周BMI”列求平均
                "Y染色体浓度": "mean",  # 对“Y染色体浓度”列求平均（列名需与Excel完全一致）
            }
        )
        .reset_index()
    )

    # ---------- 3. 保存结果到新Excel ----------
    output_path = r"C:\Users\pc\Desktop\孕妇数据_平均值结果.xlsx"
    result.to_excel(output_path, index=False)  # index=False：不保存行索引
    print(f"处理完成！结果已保存至：{output_path}")


if __name__ == "__main__":
    process_pregnancy_data()
