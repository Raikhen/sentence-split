"""
Generate 5 analysis plots from eval results.

Usage:
    python -m analysis.plot_results [--log-dir ./logs] [--output-dir ./results]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from analysis.parse_logs import parse_all_logs, results_to_dataframe


LOG_DIR = Path(__file__).parent.parent / "logs"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score confidence interval for a proportion."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p_hat = successes / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
    return p_hat, max(0, center - margin), min(1, center + margin)


def plot_1_grouped_bar(df, output_dir: Path):
    """Plot 1: Grouped bar chart - refusal rate per model, single vs split, with 95% Wilson CIs."""
    models = df["model"].unique()
    model_names = [m.split("/")[-1] for m in models]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(models))
    width = 0.35

    for i, condition in enumerate(["single", "split"]):
        rates, ci_lows, ci_highs = [], [], []
        for model in models:
            subset = df[(df["model"] == model) & (df["condition"] == condition)]
            n = len(subset)
            s = subset["refused"].sum()
            rate, ci_low, ci_high = wilson_ci(int(s), n)
            rates.append(rate)
            ci_lows.append(rate - ci_low)
            ci_highs.append(ci_high - rate)

        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, rates, width, label=condition.capitalize(),
                      yerr=[ci_lows, ci_highs], capsize=4, alpha=0.85)

    ax.set_ylabel("Refusal Rate")
    ax.set_title("Refusal Rate: Single Message vs Split Message")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "1_grouped_bar.png", dpi=150)
    plt.close()
    print("  Saved: 1_grouped_bar.png")


def plot_2_forest(df, output_dir: Path):
    """Plot 2: Forest plot - difference in refusal rate (single - split) per model with CIs."""
    models = df["model"].unique()
    model_names = [m.split("/")[-1] for m in models]

    fig, ax = plt.subplots(figsize=(10, 6))

    y_pos = np.arange(len(models))
    diffs, ci_lows, ci_highs = [], [], []

    for model in models:
        single = df[(df["model"] == model) & (df["condition"] == "single")]
        split = df[(df["model"] == model) & (df["condition"] == "split")]

        n_s = len(single)
        n_p = len(split)
        r_single = single["refused"].mean() if n_s > 0 else 0
        r_split = split["refused"].mean() if n_p > 0 else 0

        diff = r_single - r_split  # Positive = split reduces refusal

        # SE for difference of proportions
        se = np.sqrt(
            (r_single * (1 - r_single) / max(n_s, 1)) +
            (r_split * (1 - r_split) / max(n_p, 1))
        )
        ci = 1.96 * se

        diffs.append(diff)
        ci_lows.append(ci)
        ci_highs.append(ci)

    ax.errorbar(diffs, y_pos, xerr=[ci_lows, ci_highs], fmt="o", capsize=5,
                color="steelblue", markersize=8, linewidth=2)
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(model_names)
    ax.set_xlabel("Difference in Refusal Rate (Single - Split)")
    ax.set_title("Effect of Prompt Splitting on Refusal Rate")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "2_forest_plot.png", dpi=150)
    plt.close()
    print("  Saved: 2_forest_plot.png")


def plot_3_by_domain(df, output_dir: Path):
    """Plot 3: Faceted by risk domain - refusal rates broken down by 3 risk domains."""
    domains = df["risk_domain"].unique()
    models = df["model"].unique()
    model_names = [m.split("/")[-1] for m in models]

    fig, axes = plt.subplots(1, len(domains), figsize=(6 * len(domains), 6), sharey=True)
    if len(domains) == 1:
        axes = [axes]

    for ax, domain in zip(axes, domains):
        domain_df = df[df["risk_domain"] == domain]
        x = np.arange(len(models))
        width = 0.35

        for i, condition in enumerate(["single", "split"]):
            rates = []
            ci_lows, ci_highs = [], []
            for model in models:
                subset = domain_df[(domain_df["model"] == model) & (domain_df["condition"] == condition)]
                n = len(subset)
                s = subset["refused"].sum()
                rate, ci_low, ci_high = wilson_ci(int(s), n)
                rates.append(rate)
                ci_lows.append(rate - ci_low)
                ci_highs.append(ci_high - rate)

            offset = (i - 0.5) * width
            ax.bar(x + offset, rates, width, label=condition.capitalize(),
                   yerr=[ci_lows, ci_highs], capsize=3, alpha=0.85)

        ax.set_title(domain, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
        if ax == axes[0]:
            ax.set_ylabel("Refusal Rate")
            ax.legend(fontsize=8)

    plt.suptitle("Refusal Rate by Risk Domain", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "3_by_domain.png", dpi=150)
    plt.close()
    print("  Saved: 3_by_domain.png")


def plot_4_heatmap(df, output_dir: Path):
    """Plot 4: Heatmap - risk subdomain x model, colored by refusal rate difference."""
    models = df["model"].unique()
    model_names = [m.split("/")[-1] for m in models]
    subdomains = sorted(df["risk_subdomain"].unique())

    # Truncate long subdomain names
    subdomain_labels = [s[:50] + "..." if len(s) > 50 else s for s in subdomains]

    diff_matrix = np.zeros((len(subdomains), len(models)))

    for i, sd in enumerate(subdomains):
        for j, model in enumerate(models):
            single = df[(df["model"] == model) & (df["condition"] == "single") & (df["risk_subdomain"] == sd)]
            split = df[(df["model"] == model) & (df["condition"] == "split") & (df["risk_subdomain"] == sd)]
            r_single = single["refused"].mean() if len(single) > 0 else 0
            r_split = split["refused"].mean() if len(split) > 0 else 0
            diff_matrix[i, j] = r_single - r_split

    fig, ax = plt.subplots(figsize=(12, max(8, len(subdomains) * 0.5)))
    im = ax.imshow(diff_matrix, cmap="RdYlGn", aspect="auto", vmin=-0.5, vmax=0.5)

    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels(model_names, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(subdomains)))
    ax.set_yticklabels(subdomain_labels, fontsize=7)

    # Add text annotations
    for i in range(len(subdomains)):
        for j in range(len(models)):
            val = diff_matrix[i, j]
            color = "white" if abs(val) > 0.3 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color=color)

    plt.colorbar(im, ax=ax, label="Refusal Rate Diff (Single - Split)")
    ax.set_title("Refusal Rate Difference by Risk Subdomain x Model")
    plt.tight_layout()
    plt.savefig(output_dir / "4_heatmap.png", dpi=150)
    plt.close()
    print("  Saved: 4_heatmap.png")


def plot_5_biggest_diffs(df, output_dir: Path):
    """Plot 5: Top 20 samples with largest refusal rate difference between conditions."""
    # For each sample, compute average refusal rate across models for each condition
    sample_ids = df["sample_id"].unique()

    sample_diffs = []
    for sid in sample_ids:
        sample_df = df[df["sample_id"] == sid]
        single = sample_df[sample_df["condition"] == "single"]
        split = sample_df[sample_df["condition"] == "split"]

        if len(single) == 0 or len(split) == 0:
            continue

        r_single = single["refused"].mean()
        r_split = split["refused"].mean()
        diff = r_single - r_split

        # Which models flipped
        models_that_flipped = []
        for model in df["model"].unique():
            m_single = single[single["model"] == model]["refused"]
            m_split = split[split["model"] == model]["refused"]
            if len(m_single) > 0 and len(m_split) > 0:
                if m_single.iloc[0] and not m_split.iloc[0]:
                    models_that_flipped.append(model.split("/")[-1])

        sample_diffs.append({
            "sample_id": sid,
            "diff": diff,
            "abs_diff": abs(diff),
            "single_rate": r_single,
            "split_rate": r_split,
            "domain": sample_df["risk_domain"].iloc[0],
            "models_flipped": ", ".join(models_that_flipped),
        })

    # Sort by absolute difference, take top 20
    sample_diffs.sort(key=lambda x: x["abs_diff"], reverse=True)
    top_20 = sample_diffs[:20]

    if not top_20:
        print("  No sample differences to plot")
        return

    fig, ax = plt.subplots(figsize=(12, 8))
    y_pos = np.arange(len(top_20))
    colors = ["#d32f2f" if d["diff"] > 0 else "#1976d2" for d in top_20]

    ax.barh(y_pos, [d["diff"] for d in top_20], color=colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"ID {d['sample_id']} ({d['domain'][:20]})" for d in top_20], fontsize=8)
    ax.set_xlabel("Refusal Rate Difference (Single - Split)")
    ax.set_title("Top 20 Samples: Largest Refusal Rate Difference")
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(axis="x", alpha=0.3)

    # Add text labels
    for i, d in enumerate(top_20):
        ax.text(d["diff"] + 0.01 * np.sign(d["diff"]), i,
                f"S:{d['single_rate']:.0%} → P:{d['split_rate']:.0%}",
                va="center", fontsize=7)

    plt.tight_layout()
    plt.savefig(output_dir / "5_biggest_diffs.png", dpi=150)
    plt.close()
    print("  Saved: 5_biggest_diffs.png")


def main():
    parser = argparse.ArgumentParser(description="Generate analysis plots")
    parser.add_argument("--log-dir", type=str, default=str(LOG_DIR))
    parser.add_argument("--output-dir", type=str, default=str(RESULTS_DIR))
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Parsing eval logs...")
    results = parse_all_logs(log_dir)

    if not results:
        print("No results found. Run evals first.")
        return

    df = results_to_dataframe(results)
    print(f"Loaded {len(df)} results: {df['model'].nunique()} models, "
          f"{df['condition'].nunique()} conditions\n")

    print("Generating plots...")
    plot_1_grouped_bar(df, output_dir)
    plot_2_forest(df, output_dir)
    plot_3_by_domain(df, output_dir)
    plot_4_heatmap(df, output_dir)
    plot_5_biggest_diffs(df, output_dir)

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()
