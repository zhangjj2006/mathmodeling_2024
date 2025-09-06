import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from scipy.optimize import minimize_scalar

plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def analyze_with_dynamic_utility():
    def get_rearly(t):
        return 2.0 * np.exp(-t / 50)

    def get_rlate(t):
        if t < 84:
            return 0.1
        elif t <= 189:
            normalized_t = (t - 84) / (189 - 84)
            return 0.1 + 0.9 * (normalized_t**2)
        else:
            return 1.0

    def calculate_utility(t, survival_func):
        if t < 0:
            return float("inf")
        closest_time_idx = np.abs(survival_func.index - t).argmin()
        actual_time = survival_func.index[closest_time_idx]
        p_t = 1 - survival_func.loc[actual_time, "s_t"]
        return (1 - p_t) * get_rearly(t) + p_t * get_rlate(t)

    df_middle = pd.read_excel("./python_code/bmi_Y_middle_result.xlsx")
    df_cannot_test = pd.read_excel("./python_code/bmi_Y_cannot_test_result.xlsx")
    df_always_can_test = pd.read_excel(
        "./python_code/bmi_Y_always_can_test_result.xlsx"
    )

    df_all = pd.concat(
        [
            df_middle.assign(category="middle"),
            df_cannot_test.assign(category="cannot"),
            df_always_can_test.assign(category="always_can"),
        ],
        ignore_index=True,
    )

    df_all["duration"] = 0.0
    df_all["event_observed"] = 0

    df_all.loc[df_all["category"] == "cannot", "duration"] = df_all["最晚不达标天数"]
    df_all.loc[df_all["category"] == "cannot", "event_observed"] = 0

    df_all.loc[df_all["category"] == "middle", "duration"] = df_all["预测达标天数"]
    df_all.loc[df_all["category"] == "middle", "event_observed"] = 1

    df_all.loc[df_all["category"] == "always_can", "duration"] = 0.1
    df_all.loc[df_all["category"] == "always_can", "event_observed"] = 1

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

    results_table = []

    for bmi_cat in bmi_categories:
        df_bmi = df_all[df_all["bmi_category"] == bmi_cat].copy()

        if df_bmi.empty:
            results_table.append(
                {
                    "BMI分组": bmi_cat,
                    "最优时点(天)": "无数据",
                    "最小效用值": "无数据",
                    "风险水平": "无数据",
                    "样本量": 0,
                }
            )
            continue

        print(f"--- 正在分析 BMI 区间: {bmi_cat} ---")

        kmf = KaplanMeierFitter()
        kmf.fit(durations=df_bmi["duration"], event_observed=df_bmi["event_observed"])

        s_t = kmf.survival_function_.rename(columns={"KM_estimate": "s_t"})

        def objective(t):
            return calculate_utility(t, s_t)

        result = minimize_scalar(
            objective, bounds=(0, df_bmi["duration"].max()), method="bounded"
        )

        optimal_time = result.x
        optimal_utility = result.fun
        risk_level = 1 / optimal_utility if optimal_utility > 0 else float("inf")
        sample_size = len(df_bmi)

        print(f"\n最优预测时间点分析:")
        print(f"  - 最优时间点: {optimal_time:.1f} 天")
        print(f"  - 最小效用值: {optimal_utility:.4f}")
        print(f"  - 风险水平: {risk_level:.4f}")
        print(f"  - 样本量: {sample_size}")

        plt.figure(figsize=(14, 8))
        ax = plt.gca()

        (1 - kmf.survival_function_).plot(
            ax=ax,
            label=f"累积达标函数 F(t) ({bmi_cat})",
            color="seagreen",
            linewidth=2.5,
        )

        time_points = np.linspace(0, df_bmi["duration"].max(), 200)
        utilities = [calculate_utility(t, s_t) for t in time_points]
        ax_twin = ax.twinx()
        ax_twin.plot(
            time_points, utilities, "--", color="red", label="效用函数 E(t)", alpha=0.6
        )

        ax_twin.axvline(x=optimal_time, color="purple", linestyle="--", linewidth=1)
        ax_twin.text(
            optimal_time + 2,
            min(utilities) + 0.1,
            f"最优时间点: {optimal_time:.1f}天\n效用值: {optimal_utility:.3f}",
            color="purple",
            fontsize=10,
        )

        ax.axvspan(0, 84, facecolor="green", alpha=0.15, label="早期阶段 (<84天)")
        ax.axvspan(84, 189, facecolor="orange", alpha=0.15, label="过渡阶段 (84-189天)")
        ax.axvspan(
            189,
            df_all["duration"].max() + 10,
            facecolor="red",
            alpha=0.15,
            label="晚期阶段 (>189天)",
        )

        ax.set_title(f"BMI区间 {bmi_cat} 达标情况与效用分析", fontsize=16)
        ax.set_xlabel("天数", fontsize=12)
        ax.set_ylabel("累积达标比例", fontsize=12)
        ax_twin.set_ylabel("效用值", fontsize=12)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        ax_twin.set_ylim(0, max(utilities) * 1.2)
        ax.set_xlim(-5, df_all["duration"].max() + 10)

        filename = f'./python_code/BMI_{bmi_cat.replace("<", "lt").replace(">", "gt").replace("-", "_")}_utility_analysis.png'
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()

        results_table.append(
            {
                "BMI分组": bmi_cat,
                "最优时点(天)": f"{optimal_time:.1f}",
                "最小效用值": f"{optimal_utility:.4f}",
                "风险水平": f"{risk_level:.4f}",
                "样本量": sample_size,
            }
        )

        print(f"\n图表已保存至: {filename}")
        print("\n" + "=" * 50 + "\n")

    print("\n\n=== 各BMI分组NIPT时点计算结果 ===")
    print("BMI分组\t\t最优时点(天)\t最小效用值\t风险水平\t样本量")
    print("-" * 70)

    for result in results_table:
        print(
            f"{result['BMI分组']}\t{result['最优时点(天)']}\t\t{result['最小效用值']}\t\t{result['风险水平']}\t\t{result['样本量']}"
        )

    results_df = pd.DataFrame(results_table)
    results_df.to_excel("./python_code/NIPT_optimal_times_results.xlsx", index=False)
    print(f"\n结果表格已保存至: ./python_code/NIPT_optimal_times_results.xlsx")

    return results_df


results = analyze_with_dynamic_utility()
