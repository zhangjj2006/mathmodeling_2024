import pandas as pd
import matplotlib.pyplot as plt


def plot_y_bmi_scatter(df):
    # 设置图片清晰度
    plt.rcParams["figure.dpi"] = 300

    # 设置中文字体
    plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei"]

    # 根据检测孕周列对数据进行排序
    df = df.sort_values(by="检测孕周")

    # 提取绘图所需列数据
    bmi = df["孕妇BMI"]
    y_concentration = df["Y染色体浓度"]

    # 绘制 Y 浓度 vs BMI 的散点图，设置点的大小为 10
    plt.scatter(bmi, y_concentration, color="red", alpha=0.5, s=3)
    plt.title("Y 浓度 vs BMI", fontsize=10)  # 设置标题字体大小为 10
    plt.xlabel("BMI", fontsize=8)  # 设置 x 轴标签字体大小为 8
    plt.ylabel("Y 浓度", fontsize=8)  # 设置 y 轴标签字体大小为 8

    # 保存图片
    plt.savefig("graph//y_bmi_scatter.png")

    # 显示图形
    plt.show()


if __name__ == "__main__":
    # 读取 Excel 文件（直接使用文件名）
    excel_file = pd.ExcelFile("python_code//附件.xlsx")

    # 获取指定工作表中的数据
    df = excel_file.parse("男胎检测数据")

    # 调用函数绘制散点图
    plot_y_bmi_scatter(df)
