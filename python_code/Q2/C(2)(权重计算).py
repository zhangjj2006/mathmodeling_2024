import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def calculate_ahp_weights(matrix):

    n = matrix.shape[0]

    row_geom_mean = np.prod(matrix, axis=1) ** (1 / n)

    weights = row_geom_mean / np.sum(row_geom_mean)

    aw = np.dot(matrix, weights)
    lambda_max = np.mean(aw / weights)

    ci = (lambda_max - n) / (n - 1)

    ri_table = {
        1: 0,
        2: 0,
        3: 0.58,
        4: 0.90,
        5: 1.12,
        6: 1.24,
        7: 1.32,
        8: 1.41,
        9: 1.45,
        10: 1.49,
    }
    ri = ri_table.get(n, 1.51)

    cr = ci / ri if ri != 0 else 0

    return weights, cr


def generate_stacked_bmi_mixed_charts_with_ahp_weights():
    judgment_matrix = np.array([[1, 4, 6], [1 / 4, 1, 2], [1 / 6, 1 / 2, 1]])

    category_order = ["always_can", "middle", "cannot"]

    ahp_weights, cr = calculate_ahp_weights(judgment_matrix)

    print("--- AHP 权重计算结果 ---")
    print(f"判断矩阵:\n{judgment_matrix}\n")
    print("计算出的权重向量:")
    for i, category in enumerate(category_order):
        print(f"  - {category}: {ahp_weights[i]:.4f}")

    print(f"\n一致性指标 CI: {cr*0.58:.4f}")
    print(f"一致性比率 CR: {cr:.4f}")
    if cr < 0.1:
        print(" -> 判断矩阵具有满意的一致性 (CR < 0.1)。")
    else:
        print(" -> 警告: 判断矩阵的一致性较差 (CR >= 0.1)，建议调整比较值。")
    print("-" * 28 + "\n")

    objective_weights_dict = {
        category: weight for category, weight in zip(category_order, ahp_weights)
    }

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

    df_all["days_raw"] = 0
    df_all.loc[df_all["category"] == "cannot", "days_raw"] = df_all["最晚不达标天数"]
    df_all.loc[df_all["category"] == "middle", "days_raw"] = df_all["预测达标天数"]
    df_all.loc[df_all["category"] == "always_can", "days_raw"] = df_all["最早达标天数"]

    def categorize_bmi(bmi):
        if bmi < 30.17:
            return "<30.17"
        elif 30.17 <= bmi < 32.25:
            return "30.17-32.25"
        elif 32.25 <= bmi < 34.70:
            return "32.25-34.70"
        elif 34.70 <= bmi < 37.11:
            return "34.70-37.11"
        else:
            return ">37.11"

    df_all["bmi_category"] = df_all["BMI"].apply(categorize_bmi)

    df_all["weight"] = df_all["category"].map(objective_weights_dict)

    bmi_categories = ["<30.17", "30.17-32.25", "32.25-34.70", "34.70-37.11", ">37.11"]

    stacking_order = ["cannot", "middle", "always_can"]
    colors = {"cannot": "lightcoral", "middle": "goldenrod", "always_can": "seagreen"}
    labels = {"cannot": "不能达标", "middle": "中间达标", "always_can": "始终达标"}

    for bmi_cat in bmi_categories:
        df_bmi = df_all[df_all["bmi_category"] == bmi_cat]

        if df_bmi.empty:
            continue

        fig, ax1 = plt.subplots(figsize=(12, 8))

        all_days_raw = df_bmi["days_raw"].tolist()

        stacked_data, stacked_labels, stacked_colors = [], [], []
        for category in stacking_order:
            days = df_bmi[df_bmi["category"] == category]["days_raw"].tolist()
            if days:
                stacked_data.append(days)
                stacked_labels.append(labels[category])
                stacked_colors.append(colors[category])

        if stacked_data:
            data_range = max(all_days_raw) - min(all_days_raw) if all_days_raw else 0
            bins_count = (
                min(100, max(50, int(data_range / 3))) if data_range > 0 else 50
            )
            ax1.hist(
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
        ax1.set_title(f"BMI区间 {bmi_cat} 分布（AHP业务逻辑权重）", fontsize=14)
        ax1.legend(loc="upper left", fontsize=10)
        ax1.grid(True, alpha=0.3)

        if len(all_days_raw) > 1:
            ax2 = ax1.twinx()
            days_for_kde = df_bmi["days_raw"].to_numpy()
            weights_for_kde = df_bmi["weight"].to_numpy()

            try:
                weighted_kde = stats.gaussian_kde(days_for_kde, weights=weights_for_kde)
                x_range = np.linspace(min(all_days_raw), max(all_days_raw), 1000)
                kde_values = weighted_kde(x_range)

                ax2.plot(
                    x_range,
                    kde_values,
                    color="dodgerblue",
                    linewidth=2,
                    linestyle="-",
                    label="AHP加权概率密度",
                )
                ax2.set_ylabel("加权概率密度", fontsize=12)
                ax2.legend(loc="upper right", fontsize=10)

                cumulative_prob = np.cumsum(kde_values) * (x_range[1] - x_range[0])
                prob_levels = [0.85, 0.90, 0.95]
                for prob in prob_levels:
                    try:
                        index = np.where(cumulative_prob >= prob)[0][0]
                        day_value = x_range[index]
                        ax2.axvline(
                            x=day_value, color="purple", linestyle="--", linewidth=1
                        )
                        ax2.text(
                            day_value + 0.5,
                            max(kde_values) * (1 - prob),
                            f"{int(prob*100)}% -> {day_value:.1f}天",
                            color="purple",
                            rotation=90,
                        )
                    except IndexError:
                        pass

            except Exception as e:
                print(f"BMI区间 {bmi_cat} 计算加权概率密度时出错: {e}")

        plt.tight_layout()
        filename = f'./python_code/BMI_{bmi_cat.replace("<", "lt").replace(">", "gt").replace("-", "_")}_AHP_weighted_dist.png'
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()


generate_stacked_bmi_mixed_charts_with_ahp_weights()
