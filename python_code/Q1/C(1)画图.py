import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


def linear_regression_analysis(excel_path, sheet_name, x_col, y_col):
    """
    读取Excel中两列数据，进行线性回归分析并可视化

    参数说明：
    excel_path: str - Excel文件的完整路径（如"C:/data/your_data.xlsx"）
    sheet_name: str - 数据所在的工作表名称（如"Sheet1"）
    x_col: str - 自变量（X轴）对应的列名（如"检测孕周_天数"）
    y_col: str - 因变量（Y轴）对应的列名（如"Y染色体浓度"）
    """
    # --------------------------
    # 1. 读取并预处理Excel数据
    # --------------------------
    try:
        # 读取Excel文件
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        print(f"成功读取Excel文件，数据共{df.shape[0]}行、{df.shape[1]}列")

        # 提取目标列数据（删除含缺失值的行）
        data = df[[x_col, y_col]].dropna()
        x_data = data[x_col].values.reshape(-1, 1)  # 自变量需为2D数组（sklearn要求）
        y_data = data[y_col].values

        # 检查数据有效性（避免空数据或单一点）
        if len(x_data) < 2:
            raise ValueError(
                f"有效数据不足2个，请检查{excel_path}中{x_col}和{y_col}列的数据"
            )

        print(f"提取有效数据{len(x_data)}组（已删除缺失值）")
        print(f"自变量{x_col}范围：{x_data.min():.2f} ~ {x_data.max():.2f}")
        print(f"因变量{y_col}范围：{y_data.min():.6f} ~ {y_data.max():.6f}")

    except FileNotFoundError:
        print(f"错误：找不到文件 {excel_path}，请检查路径是否正确")
        return
    except KeyError as e:
        print(f"错误：Excel中不存在列 {e}，请检查列名是否与工作表一致")
        return
    except Exception as e:
        print(f"数据读取失败：{str(e)}")
        return

    # --------------------------
    # 2. 建立线性回归模型并拟合
    # --------------------------
    # 初始化线性回归模型
    lr_model = LinearRegression()
    # 拟合数据（计算斜率和截距）
    lr_model.fit(x_data, y_data)

    # 提取模型参数
    slope = lr_model.coef_[0]  # 斜率（回归系数）
    intercept = lr_model.intercept_  # 截距
    # 计算预测值（用于绘制拟合线）
    y_pred = lr_model.predict(x_data)
    # 计算R²（决定系数，评估拟合优度）
    r2 = r2_score(y_data, y_pred)

    # --------------------------
    # 3. 可视化：散点图 + 拟合线
    # --------------------------
    # 设置中文字体（避免中文乱码）
    plt.rcParams["font.sans-serif"] = [
        "WenQuanYi Zen Hei",
        "SimHei",
        "Arial Unicode MS",
    ]
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)  # dpi=300保证图片清晰度

    # 绘制散点图（原始数据）
    ax.scatter(x_data, y_data, color="#1f77b4", alpha=0.6, s=50, label="原始数据")

    # 绘制拟合线（按自变量排序，使线条平滑）
    sort_idx = np.argsort(x_data.flatten())  # 按x值排序的索引
    ax.plot(
        x_data[sort_idx],
        y_pred[sort_idx],
        color="#ff7f0e",
        linewidth=2.5,
        label=f"拟合线：y = {slope:.6f}x + {intercept:.6f}",
    )

    # 设置图表标签和标题
    ax.set_xlabel(x_col, fontsize=12, fontweight="bold")
    ax.set_ylabel(y_col, fontsize=12, fontweight="bold")
    ax.set_title(
        f"{x_col}与{y_col}线性回归分析\nR^2 = {r2:.6f}",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    # 添加网格和图例
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=10, loc="best")

    # 保存图表（避免plt.show()阻塞后续代码）
    plt.tight_layout()  # 自动调整布局，防止标签被截断
    plt.savefig(
        f"{x_col}_vs_{y_col}_linear_regression.png", dpi=300, bbox_inches="tight"
    )
    plt.close()  # 关闭图表，释放内存

    # --------------------------
    # 4. 输出结果
    # --------------------------
    print("\n" + "=" * 60)
    print("线性回归分析结果")
    print("=" * 60)
    print(f"线性回归方程：y = {slope:.6f} * x + {intercept:.6f}")
    print(f"其中：x = {x_col}, y = {y_col}")
    print(f"斜率（回归系数）：{slope:.6f}")
    print(f"截距：{intercept:.6f}")
    print(f"决定系数 R²：{r2:.6f}")
    print("=" * 60)
    print(f"拟合图已保存为：{x_col}_vs_{y_col}_linear_regression.png")


# --------------------------
# 调用函数（需根据你的Excel文件修改以下参数）
# --------------------------
if __name__ == "__main__":
    # 请替换为你的Excel文件路径、工作表名称、自变量/因变量列名
    EXCEL_PATH = "文件//Week_Y_BMI_Line_result(1).xlsx"
    SHEET_NAME = "Sheet1"
    X_COLUMN = "BMI增长速率"  # 自变量列名（如"检测孕周_天数"）
    Y_COLUMN = "Y染色体浓度增长速率"  # 因变量列名（如"Y染色体浓度"）

    # 执行线性回归分析
    linear_regression_analysis(
        excel_path=EXCEL_PATH, sheet_name=SHEET_NAME, x_col=X_COLUMN, y_col=Y_COLUMN
    )
