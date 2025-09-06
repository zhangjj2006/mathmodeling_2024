import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from scipy.optimize import minimize_scalar
from scipy.stats import norm
import re

plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def analyze_with_error_simulation(num_simulations=100, error_std=0.01):
    # 定义效用函数（与之前相同）
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

    # 辅助函数：清理文件名中的非法字符
    def clean_filename(name):
        # 替换文件名字符串中的非法字符
        return re.sub(r'[<>:"/\\|?*]', "_", name)

    # 加载数据
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

    results = {
        bmi_cat: {"times": [], "utilities": [], "risks": []}
        for bmi_cat in bmi_categories
    }

    original_results = {}
    for bmi_cat in bmi_categories:
        df_bmi = df_all[df_all["bmi_category"] == bmi_cat].copy()
        if df_bmi.empty:
            continue

        kmf = KaplanMeierFitter()
        kmf.fit(durations=df_bmi["duration"], event_observed=df_bmi["event_observed"])
        s_t = kmf.survival_function_.rename(columns={"KM_estimate": "s_t"})

        def objective(t):
            return calculate_utility(t, s_t)

        result = minimize_scalar(
            objective, bounds=(0, df_bmi["duration"].max()), method="bounded"
        )

        original_time = result.x
        original_utility = result.fun
        original_risk = 1 / original_utility if original_utility > 0 else float("inf")

        original_results[bmi_cat] = {
            "time": original_time,
            "utility": original_utility,
            "risk": original_risk,
        }

        print(f"BMI区间 {bmi_cat} 原始结果:")
        print(f"  最优时点: {original_time:.2f} 天")
        print(f"  最小效用值: {original_utility:.4f}")
        print(f"  风险水平: {original_risk:.4f}")

    # 开始模拟
    for simulation in range(num_simulations):
        print(f"正在进行第 {simulation+1} 次模拟...")

        df_simulated = df_all.copy()
        np.random.seed(simulation)

        mask_middle = df_simulated["category"] == "middle"
        mask_always = df_simulated["category"] == "always_can"

        if sum(mask_middle) > 0:
            error_middle = np.random.normal(
                0,
                error_std * df_simulated.loc[mask_middle, "预测达标天数"].mean(),
                size=sum(mask_middle),
            )
            df_simulated.loc[mask_middle, "duration"] += error_middle

        if sum(mask_always) > 0:
            error_always = np.random.normal(
                0,
                error_std * df_simulated.loc[mask_always, "最早达标天数"].mean(),
                size=sum(mask_always),
            )
            df_simulated.loc[mask_always, "duration"] += error_always

        mask_cannot = df_simulated["category"] == "cannot"
        if sum(mask_cannot) > 0:
            error_cannot = np.random.normal(
                0,
                error_std * df_simulated.loc[mask_cannot, "最晚不达标天数"].mean(),
                size=sum(mask_cannot),
            )
            df_simulated.loc[mask_cannot, "duration"] += error_cannot

        df_simulated["duration"] = df_simulated["duration"].clip(lower=0.1)

        for bmi_cat in bmi_categories:
            df_bmi = df_simulated[df_simulated["bmi_category"] == bmi_cat].copy()
            if df_bmi.empty:
                continue

            kmf = KaplanMeierFitter()
            kmf.fit(
                durations=df_bmi["duration"], event_observed=df_bmi["event_observed"]
            )
            s_t = kmf.survival_function_.rename(columns={"KM_estimate": "s_t"})

            def objective(t):
                return calculate_utility(t, s_t)

            result = minimize_scalar(
                objective, bounds=(0, df_bmi["duration"].max()), method="bounded"
            )

            optimal_time = result.x
            optimal_utility = result.fun
            risk = 1 / optimal_utility if optimal_utility > 0 else float("inf")

            results[bmi_cat]["times"].append(optimal_time)
            results[bmi_cat]["utilities"].append(optimal_utility)
            results[bmi_cat]["risks"].append(risk)

    print("\n=== 检测误差影响分析 ===")
    for bmi_cat in bmi_categories:
        if results[bmi_cat]["times"]:
            times = np.array(results[bmi_cat]["times"])
            utilities = np.array(results[bmi_cat]["utilities"])
            risks = np.array(results[bmi_cat]["risks"])

            mean_time = times.mean()
            std_time = times.std()
            time_confidence_interval = norm.interval(
                0.95, loc=mean_time, scale=std_time / np.sqrt(len(times))
            )

            mean_utility = utilities.mean()
            std_utility = utilities.std()

            mean_risk = risks.mean()
            std_risk = risks.std()

            original_time = original_results.get(bmi_cat, {}).get("time", 0)
            original_utility = original_results.get(bmi_cat, {}).get("utility", 0)
            original_risk = original_results.get(bmi_cat, {}).get("risk", 0)

            print(f"BMI区间 {bmi_cat}:")
            print(f"  原始最优时点: {original_time:.2f} 天")
            print(f"  模拟最优时点均值: {mean_time:.2f} 天")
            print(f"  标准差: {std_time:.2f} 天")
            print(
                f"  95%置信区间: ({time_confidence_interval[0]:.2f}, {time_confidence_interval[1]:.2f}) 天"
            )
            print(f"  原始最小效用值: {original_utility:.4f}")
            print(f"  模拟最小效用值均值: {mean_utility:.4f}")
            print(f"  效用值标准差: {std_utility:.4f}")
            print(f"  原始风险水平: {original_risk:.4f}")
            print(f"  模拟风险水平均值: {mean_risk:.4f}")
            print(f"  风险水平标准差: {std_risk:.4f}")

            plt.figure(figsize=(12, 8))

            plt.subplot(2, 2, 1)
            plt.hist(times, bins=20, alpha=0.7, edgecolor="black")
            plt.axvline(
                mean_time,
                color="r",
                linestyle="--",
                label=f"模拟平均时点: {mean_time:.2f}",
            )
            plt.axvline(
                original_time,
                color="b",
                linestyle="-",
                label=f"原始最优时点: {original_time:.2f}",
            )
            plt.axvline(
                time_confidence_interval[0],
                color="g",
                linestyle=":",
                label="95%置信区间",
            )
            plt.axvline(time_confidence_interval[1], color="g", linestyle=":")
            plt.xlabel("最优NIPT时点（天）")
            plt.ylabel("频数")
            plt.title(f"BMI区间 {bmi_cat} - 最优时点分布")
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.subplot(2, 2, 2)
            plt.hist(utilities, bins=20, alpha=0.7, edgecolor="black", color="orange")
            plt.axvline(
                mean_utility,
                color="r",
                linestyle="--",
                label=f"模拟平均效用值: {mean_utility:.4f}",
            )
            plt.axvline(
                original_utility,
                color="b",
                linestyle="-",
                label=f"原始最小效用值: {original_utility:.4f}",
            )
            plt.xlabel("最小效用值")
            plt.ylabel("频数")
            plt.title(f"BMI区间 {bmi_cat} - 最小效用值分布")
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.subplot(2, 2, 3)
            plt.scatter(times, utilities, alpha=0.6)
            plt.axvline(original_time, color="b", linestyle="-", label="原始最优时点")
            plt.axhline(
                original_utility, color="b", linestyle="-", label="原始最小效用值"
            )
            plt.xlabel("最优时点（天）")
            plt.ylabel("最小效用值")
            plt.title(f"BMI区间 {bmi_cat} - 时点与效用值关系")
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.subplot(2, 2, 4)
            plt.hist(risks, bins=20, alpha=0.7, edgecolor="black", color="red")
            plt.axvline(
                mean_risk,
                color="r",
                linestyle="--",
                label=f"模拟平均风险: {mean_risk:.4f}",
            )
            plt.axvline(
                original_risk,
                color="b",
                linestyle="-",
                label=f"原始风险: {original_risk:.4f}",
            )
            plt.xlabel("风险水平")
            plt.ylabel("频数")
            plt.title(f"BMI区间 {bmi_cat} - 风险水平分布")
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.tight_layout()

            safe_bmi_cat = clean_filename(bmi_cat)
            filename = f"./python_code/error_analysis_BMI_{safe_bmi_cat}.png"
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            plt.close()

            data_df = pd.DataFrame({"time": times, "utility": utilities, "risk": risks})
            csv_filename = f"./python_code/error_analysis_BMI_{safe_bmi_cat}.csv"
            data_df.to_csv(csv_filename, index=False)

            print(f"  图表已保存至: {filename}")
            print(f"  数据已保存至: {csv_filename}")
        else:
            print(f"BMI区间 {bmi_cat}: 无数据")


analyze_with_error_simulation(num_simulations=100, error_std=0.05)
