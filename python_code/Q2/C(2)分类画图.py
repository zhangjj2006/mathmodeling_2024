import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams
from scipy import stats
import re


plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

def generate_stacked_bmi_mixed_charts():
    df_middle = pd.read_excel("./python_code/bmi_Y_middle_result.xlsx")
    df_cannot_test = pd.read_excel("./python_code/bmi_Y_cannot_test_result.xlsx")
    df_always_can_test = pd.read_excel(
        "./python_code/bmi_Y_always_can_test_result.xlsx"
    )

    df_middle["category"] = "middle"
    df_cannot_test["category"] = "cannot"
    df_always_can_test["category"] = "always_can"

    df_all = pd.concat(
        [df_middle, df_cannot_test, df_always_can_test], ignore_index=True
    )

    def categorize_bmi(bmi):
        if bmi < 30.26:
            return "<30.26"
        elif 30.26 <= bmi < 32.30:
            return "30.26-32.30"
        elif 32.30 <= bmi < 34.92:
            return "32.30-34.92"
        elif 34.92 <= bmi < 39.49:
            return "34.92-39.49"
        else:
            return ">39.49"

    df_all["bmi_category"] = df_all["BMI"].apply(categorize_bmi)

    bmi_categories = ["<30.26", "30.26-32.30", "32.30-34.92", "34.92-39.49", ">39.49"]

    categories = ["cannot", "middle", "always_can"]
    colors = {"cannot": "lightcoral", "middle": "gold", "always_can": "seagreen"}
    labels = {"cannot": "不能达标", "middle": "中间达标", "always_can": "始终达标"}

    for bmi_cat in bmi_categories:
        df_bmi = df_all[df_all["bmi_category"] == bmi_cat]

        if df_bmi.empty:
            print(f"警告: BMI区间 {bmi_cat} 没有数据")
            continue

        fig, ax1 = plt.subplots(figsize=(12, 8))

        all_days = []

        days_data = {}
        for category in categories:
            df_category = df_bmi[df_bmi["category"] == category]

            if not df_category.empty:
                if category == "cannot":
                    days = df_category["最晚不达标天数"].tolist()
                elif category == "middle":
                    days = df_category["预测达标天数"].tolist()
                else:
                    days = df_category["最早达标天数"].tolist()

                days_data[category] = days
                all_days.extend(days)

        stacked_data = []
        stacked_labels = []
        stacked_colors = []

        if "cannot" in days_data:
            stacked_data.append(days_data["cannot"])
            stacked_labels.append(labels["cannot"])
            stacked_colors.append(colors["cannot"])

        if "middle" in days_data:
            stacked_data.append(days_data["middle"])
            stacked_labels.append(labels["middle"])
            stacked_colors.append(colors["middle"])

        if "always_can" in days_data:
            stacked_data.append(days_data["always_can"])
            stacked_labels.append(labels["always_can"])
            stacked_colors.append(colors["always_can"])

        if stacked_data:
            if len(all_days) > 0:
                data_range = max(all_days) - min(all_days)
                bins_count = min(100, max(50, int(data_range / 3)))
            else:
                bins_count = 50

            n, bins, patches = ax1.hist(
                stacked_data,
                bins=bins_count,
                stacked=True,
                color=stacked_colors,
                label=stacked_labels,
                alpha=0.95,
                edgecolor="black",
                linewidth=0.1,
            )

        ax1.set_xlabel("天数", fontsize=12)
        ax1.set_ylabel("人数", fontsize=12)
        ax1.set_title(
            f"BMI区间 {bmi_cat} 的孕妇Y染色体达标情况分布（堆叠图）", fontsize=14
        )
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        if len(all_days) > 1:
            ax2 = ax1.twinx()

            try:
                kde = stats.gaussian_kde(all_days)
                x_range = np.linspace(min(all_days), max(all_days), 1000)
                kde_values = kde(x_range)

                ax2.plot(
                    x_range,
                    kde_values,
                    color="dodgerblue",
                    linewidth=2,
                    linestyle="-",
                    label="总体概率密度",
                )

                ax2.set_ylabel("概率密度", fontsize=12)
                ax2.legend(bbox_to_anchor=(1.0, 0.9), fontsize=10)
            except Exception as e:
                print(f"BMI区间 {bmi_cat} 计算概率密度时出错: {e}")

        plt.tight_layout()

        filename = f'./python_code/BMI_{bmi_cat.replace("<", "lt").replace(">", "gt").replace("-", "_")}_stacked_distribution.png'
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"BMI区间 {bmi_cat} 统计:")
        total_count = len(df_bmi)
        for category in categories:
            count = len(df_bmi[df_bmi["category"] == category])
            percentage = (count / total_count) * 100 if total_count > 0 else 0
            print(f"  {labels[category]}: {count} 人 ({percentage:.1f}%)")
        print(f"  总计: {total_count} 人")
        print()


def generate_overall_stacked_bmi_chart():
    df_middle = pd.read_excel("./python_code/bmi_Y_middle_result.xlsx")
    df_cannot_test = pd.read_excel("./python_code/bmi_Y_cannot_test_result.xlsx")
    df_always_can_test = pd.read_excel(
        "./python_code/bmi_Y_always_can_test_result.xlsx"
    )

    df_middle["category"] = "middle"
    df_cannot_test["category"] = "cannot"
    df_always_can_test["category"] = "always_can"

    df_all = pd.concat(
        [df_middle, df_cannot_test, df_always_can_test], ignore_index=True
    )

    def categorize_bmi(bmi):
        if bmi < 30.26:
            return "<30.26"
        elif 30.26 <= bmi < 32.30:
            return "30.26-32.30"
        elif 32.30 <= bmi < 34.92:
            return "32.30-34.92"
        elif 34.92 <= bmi < 39.49:
            return "34.92-39.49"
        else:
            return ">39.49"

    df_all["bmi_category"] = df_all["BMI"].apply(categorize_bmi)

    bmi_categories = ["<30.26", "30.26-32.30", "32.30-34.92", "34.92-39.49", ">39.49"]

    categories = ["cannot", "middle", "always_can"]
    colors = {"cannot": "red", "middle": "yellow", "always_can": "green"}
    labels = {"cannot": "不能达标", "middle": "中间达标", "always_can": "始终达标"}

    fig, ax1 = plt.subplots(figsize=(15, 8))

    all_data_by_category = {cat: [] for cat in categories}

    for bmi_cat in bmi_categories:
        df_bmi = df_all[df_all["bmi_category"] == bmi_cat]

        for category in categories:
            df_category = df_bmi[df_bmi["category"] == category]

            if not df_category.empty:
                if category == "cannot":
                    days = df_category["最晚不达标天数"].tolist()
                elif category == "middle":
                    days = df_category["预测达标天数"].tolist()
                else:
                    days = df_category["最早达标天数"].tolist()

                labeled_days = [(day, bmi_cat) for day in days]
                all_data_by_category[category].extend(labeled_days)

    x_positions = np.arange(len(bmi_categories))
    bar_width = 0.6

    bottom_values = np.zeros(len(bmi_categories))

    for category in categories:
        heights = []
        for bmi_cat in bmi_categories:
            count = len([x for x in all_data_by_category[category] if x[1] == bmi_cat])
            heights.append(count)

        ax1.bar(
            x_positions,
            heights,
            bar_width,
            bottom=bottom_values,
            label=labels[category],
            color=colors[category],
            alpha=0.7,
            edgecolor="black",
            linewidth=0.5,
        )

        bottom_values = np.add(bottom_values, heights)

    ax1.set_xlabel("BMI区间", fontsize=12)
    ax1.set_ylabel("人数", fontsize=12)
    ax1.set_title("各BMI区间孕妇Y染色体达标情况分布（堆叠柱状图）", fontsize=14)
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(bmi_categories)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    plt.savefig(
        "./python_code/BMI_all_intervals_stacked_distribution_new.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print("总体堆叠柱状图已生成")


if __name__ == "__main__":
    generate_overall_stacked_bmi_chart()
    generate_stacked_bmi_mixed_charts()
    print("\n图表生成完成！")
