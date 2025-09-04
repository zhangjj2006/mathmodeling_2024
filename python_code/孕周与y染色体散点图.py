import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_y_week_scatter(df):
    # 设置图片清晰度
    plt.rcParams["figure.dpi"] = 300

    # 设置中文字体
    plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei"]

    # 根据检测孕周列对数据进行排序
    df = df.sort_values(by="检测孕周")

    # 提取绘图所需列数据
    week = df["检测孕周"]
    y_concentration = df["Y染色体浓度"]

    # 绘制 Y 浓度 vs 孕周的散点图，设置点的大小为 10
    plt.scatter(week, y_concentration, color="blue", alpha=0.5, s=5)
    plt.title("Y  vs week", fontsize=5)  # 设置标题字体大小为 10
    plt.xlabel("week", fontsize=2)  # 设置 x 轴标签字体大小为 8
    plt.xticks(rotation=45, fontsize=2)  # 旋转 x 轴标签 45 度，设置字体大小为 8
    plt.ylabel("Y", fontsize=4)  # 设置 y 轴标签字体大小为 8

    plt.savefig("graph//y_week_scatter.png")
    # 显示图形
    plt.show()


if __name__ == "__main__":
    # 读取 Excel 文件
    excel_file = pd.ExcelFile("python_code//附件.xlsx")

    # 获取指定工作表中的数据
    df = excel_file.parse("男胎检测数据")

    # 调用函数绘制散点图
    plot_y_week_scatter(df)
