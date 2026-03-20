"""
Interim analysis of single-message eval logs (2026-03-19T22-11 run).

Reads eval logs via inspect_ai.log.read_eval_log, computes per-model
refusal rates with confidence intervals, runs pairwise statistical
comparisons, and generates publication-quality plots.
"""

import os
import sys
from pathlib import Path
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy import stats

from inspect_ai.log import read_eval_log

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LOG_DIR = Path("/Users/dylanfridman/sentence-split/logs")
OUTPUT_DIR = Path("/Users/dylanfridman/sentence-split/analysis")
LOG_PREFIX = "2026-03-19T22-11"

FRIENDLY_NAMES = {
    "openrouter/openai/gpt-5-mini": "GPT-5 Mini",
    "openrouter/anthropic/claude-haiku-4.5": "Claude Haiku 4.5",
    "openrouter/google/gemini-3-flash-preview": "Gemini 3 Flash",
}

MODEL_COLORS = {
    "GPT-5 Mini": "#6A994E",
    "Claude Haiku 4.5": "#BC6C25",
    "Gemini 3 Flash": "#457B9D",
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load data from logs
# ---------------------------------------------------------------------------
def load_logs() -> pd.DataFrame:
    """Read the 3 target log files and return a tidy DataFrame."""
    log_files = sorted(
        f for f in os.listdir(LOG_DIR)
        if f.startswith(LOG_PREFIX) and f.endswith(".eval")
    )
    print(f"Found {len(log_files)} log files matching prefix '{LOG_PREFIX}'")

    rows = []
    for fname in log_files:
        fpath = LOG_DIR / fname
        log = read_eval_log(str(fpath))
        model_id = log.eval.model
        model_name = FRIENDLY_NAMES.get(model_id, model_id)
        task = log.eval.task

        for sample in log.samples:
            if not sample.scores:
                continue
            scorer = sample.scores.get("refusal_scorer")
            if scorer is None:
                # Fall back to first scorer
                scorer = next(iter(sample.scores.values()))

            refused = scorer.metadata.get("refused", False) if scorer.metadata else False
            accuracy = 1.0 if refused else 0.0

            rows.append({
                "model_id": model_id,
                "model": model_name,
                "task": task,
                "sample_id": sample.id,
                "epoch": sample.epoch,
                "refused": refused,
                "accuracy": accuracy,
                "score_value": str(scorer.value),
                "risk_domain": (sample.metadata or {}).get("risk_domain", ""),
                "risk_subdomain": (sample.metadata or {}).get("risk_subdomain", ""),
            })

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} scored samples across {df['model'].nunique()} models")
    for m in df["model"].unique():
        sub = df[df["model"] == m]
        print(f"  {m}: {len(sub)} samples, "
              f"{sub['sample_id'].nunique()} unique prompts, "
              f"epochs 1-{sub['epoch'].max()}")
    return df


# ---------------------------------------------------------------------------
# 2. Wilson score confidence interval
# ---------------------------------------------------------------------------
def wilson_ci(successes: int, n: int, z: float = 1.96):
    """Return (point_estimate, ci_lower, ci_upper) using Wilson score interval."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p_hat = successes / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
    return p_hat, max(0.0, center - margin), min(1.0, center + margin)


# ---------------------------------------------------------------------------
# 3. Summary statistics
# ---------------------------------------------------------------------------
def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute refusal rate + 95% Wilson CI per model."""
    records = []
    for model in sorted(df["model"].unique()):
        sub = df[df["model"] == model]
        n = len(sub)
        s = int(sub["accuracy"].sum())
        rate, ci_lo, ci_hi = wilson_ci(s, n)
        records.append({
            "model": model,
            "n_total": n,
            "n_refused": s,
            "refusal_rate": rate,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "n_unique_samples": sub["sample_id"].nunique(),
            "n_epochs": sub["epoch"].nunique(),
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 4. Statistical tests
# ---------------------------------------------------------------------------
def pairwise_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Two-proportion z-tests (with Bonferroni correction) between all model pairs."""
    models = sorted(df["model"].unique())
    n_comparisons = len(list(combinations(models, 2)))
    alpha_bonf = 0.05 / n_comparisons if n_comparisons > 0 else 0.05

    results = []
    for m1, m2 in combinations(models, 2):
        d1 = df[df["model"] == m1]["accuracy"]
        d2 = df[df["model"] == m2]["accuracy"]
        n1, n2 = len(d1), len(d2)
        p1, p2 = d1.mean(), d2.mean()

        # Pooled proportion under H0
        p_pool = (d1.sum() + d2.sum()) / (n1 + n2)
        se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
        if se == 0:
            z_stat = 0.0
            p_value = 1.0
        else:
            z_stat = (p1 - p2) / se
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        # Also do a chi-squared test as a robustness check
        table = np.array([
            [int(d1.sum()), n1 - int(d1.sum())],
            [int(d2.sum()), n2 - int(d2.sum())],
        ])
        if table.min() >= 0:
            chi2, chi2_p, _, _ = stats.chi2_contingency(table, correction=True)
        else:
            chi2, chi2_p = np.nan, np.nan

        results.append({
            "model_1": m1,
            "model_2": m2,
            "rate_1": p1,
            "rate_2": p2,
            "diff": p1 - p2,
            "z_stat": z_stat,
            "p_value_raw": p_value,
            "p_value_bonf": min(1.0, p_value * n_comparisons),
            "significant_bonf": (p_value * n_comparisons) < 0.05,
            "chi2": chi2,
            "chi2_p_raw": chi2_p,
            "n1": n1,
            "n2": n2,
            "alpha_bonf": alpha_bonf,
        })
    return pd.DataFrame(results)


def bootstrap_model_rates(df: pd.DataFrame, n_boot: int = 10000, seed: int = 42):
    """Bootstrap 95% CI for refusal rate per model (resampling epochs)."""
    rng = np.random.default_rng(seed)
    models = sorted(df["model"].unique())
    boot_results = {}
    for model in models:
        sub = df[df["model"] == model]
        # Compute per-epoch refusal rates, then bootstrap over epochs
        epoch_rates = sub.groupby("epoch")["accuracy"].mean().values
        n_epochs = len(epoch_rates)
        boot_means = []
        for _ in range(n_boot):
            idx = rng.integers(0, n_epochs, size=n_epochs)
            boot_means.append(epoch_rates[idx].mean())
        boot_means = np.array(boot_means)
        boot_results[model] = {
            "mean": epoch_rates.mean(),
            "boot_ci_lower": np.percentile(boot_means, 2.5),
            "boot_ci_upper": np.percentile(boot_means, 97.5),
            "boot_se": boot_means.std(),
        }
    return boot_results


# ---------------------------------------------------------------------------
# 5. Plots
# ---------------------------------------------------------------------------
def plot_refusal_bar(summary_df: pd.DataFrame, boot: dict, output_dir: Path):
    """Bar chart of refusal rates across models with 95% CI error bars."""
    fig, ax = plt.subplots(figsize=(8, 5.5))

    models = summary_df["model"].values
    rates = summary_df["refusal_rate"].values
    ci_lo = summary_df["ci_lower"].values
    ci_hi = summary_df["ci_upper"].values

    x = np.arange(len(models))
    colors = [MODEL_COLORS.get(m, "#888888") for m in models]

    # Error bars: distance from point estimate to CI bounds
    yerr_lo = rates - ci_lo
    yerr_hi = ci_hi - rates

    bars = ax.bar(x, rates, width=0.55, color=colors, edgecolor="white",
                  linewidth=1.2, alpha=0.9,
                  yerr=[yerr_lo, yerr_hi], capsize=6,
                  error_kw={"linewidth": 1.5, "capthick": 1.5})

    # Annotate bars
    for i, (r, lo, hi) in enumerate(zip(rates, ci_lo, ci_hi)):
        ax.text(i, hi + 0.02, f"{r:.1%}", ha="center", va="bottom",
                fontsize=12, fontweight="bold")
        ax.text(i, hi + 0.065, f"[{lo:.1%}, {hi:.1%}]", ha="center",
                va="bottom", fontsize=8, color="#555555")

    # Add sample size annotations below bars (use transform to place in axes coords)
    for i, row in summary_df.iterrows():
        ax.text(i, -0.06, f"n={row['n_total']}  ({row['n_epochs']} epochs)",
                ha="center", va="top", fontsize=8, color="#666666")

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    ax.set_ylabel("Refusal Rate (accuracy)", fontsize=12)
    ax.set_title("Single-Message Eval: Refusal Rate by Model\n(interim results, 95% Wilson CI)",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, min(1.15, max(ci_hi) + 0.15))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out_path = output_dir / "refusal_rate_bar_chart.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_epoch_strip(df: pd.DataFrame, output_dir: Path):
    """
    Strip/swarm plot showing per-sample refusal rate across epochs,
    with one panel per model. Shows variance across rollouts.
    """
    models = sorted(df["model"].unique())
    n_models = len(models)

    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 7), sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        sub = df[df["model"] == model]

        # Per-sample refusal rate (across epochs)
        sample_rates = sub.groupby("sample_id")["accuracy"].agg(["mean", "count", "sum"])
        sample_rates = sample_rates.sort_values("mean", ascending=False).reset_index()

        n_samples = len(sample_rates)
        y_pos = np.arange(n_samples)

        # Color by rate
        cmap = plt.cm.RdYlGn
        colors = [cmap(r) for r in sample_rates["mean"]]

        ax.barh(y_pos, sample_rates["mean"], color=colors, height=0.8,
                edgecolor="white", linewidth=0.3, alpha=0.85)

        # Add individual epoch dots as jittered strip
        for i, (_, row) in enumerate(sample_rates.iterrows()):
            sid = row["sample_id"]
            epoch_vals = sub[sub["sample_id"] == sid]["accuracy"].values
            n_ep = len(epoch_vals)
            if n_ep > 1:
                # Jitter y positions
                jitter = np.linspace(-0.3, 0.3, min(n_ep, 30))
                if n_ep > 30:
                    # Subsample for visual clarity
                    idx = np.random.choice(n_ep, 30, replace=False)
                    epoch_vals = epoch_vals[idx]
                    jitter = np.linspace(-0.3, 0.3, 30)
                ax.scatter(epoch_vals, [i] * len(epoch_vals) + jitter[:len(epoch_vals)],
                          s=6, color="black", alpha=0.25, zorder=3)

        ax.set_title(f"{model}\n({sub['epoch'].nunique()} epochs completed)",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Refusal Rate Across Epochs", fontsize=10)
        ax.set_xlim(-0.05, 1.15)
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        ax.set_yticks(y_pos[::5])
        ax.set_yticklabels([f"#{int(sid)}" for sid in sample_rates["sample_id"].values[::5]],
                          fontsize=7)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.2, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.axvline(x=sub["accuracy"].mean(), color="red", linestyle="--",
                   alpha=0.6, linewidth=1.2, label=f"Mean: {sub['accuracy'].mean():.1%}")
        ax.legend(loc="lower right", fontsize=8)

    if n_models > 0:
        axes[0].set_ylabel("Sample ID (sorted by refusal rate)", fontsize=10)

    fig.suptitle("Per-Sample Refusal Rate Variance Across Epochs\n"
                 "(black dots = individual epoch outcomes; bar = mean across epochs)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = output_dir / "per_sample_strip_plot.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("INTERIM ANALYSIS: Single-Message Eval (2026-03-19T22-11 run)")
    print("=" * 70)
    print()

    # Load data
    df = load_logs()
    print()

    # Summary statistics
    print("-" * 70)
    print("REFUSAL RATES WITH 95% WILSON CONFIDENCE INTERVALS")
    print("-" * 70)
    summary = compute_summary(df)
    for _, row in summary.iterrows():
        print(f"  {row['model']:25s}  "
              f"rate={row['refusal_rate']:.3f}  "
              f"95% CI=[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]  "
              f"(n={row['n_total']}, {row['n_unique_samples']} samples x {row['n_epochs']} epochs)")
    print()

    # Bootstrap CIs (resampling over epochs to account for dependence)
    print("-" * 70)
    print("BOOTSTRAP 95% CI (resampled over epochs, 10k iterations)")
    print("-" * 70)
    boot = bootstrap_model_rates(df)
    for model, b in boot.items():
        print(f"  {model:25s}  "
              f"mean={b['mean']:.3f}  "
              f"boot 95% CI=[{b['boot_ci_lower']:.3f}, {b['boot_ci_upper']:.3f}]  "
              f"(SE={b['boot_se']:.4f})")
    print()

    # Pairwise tests
    print("-" * 70)
    print("PAIRWISE COMPARISONS (two-proportion z-test, Bonferroni corrected)")
    print("-" * 70)
    tests = pairwise_tests(df)
    for _, row in tests.iterrows():
        sig = "***" if row["p_value_bonf"] < 0.001 else \
              "**" if row["p_value_bonf"] < 0.01 else \
              "*" if row["p_value_bonf"] < 0.05 else "n.s."
        print(f"  {row['model_1']:25s} vs {row['model_2']:25s}")
        print(f"    rates: {row['rate_1']:.3f} vs {row['rate_2']:.3f}  "
              f"(diff={row['diff']:+.3f})")
        print(f"    z={row['z_stat']:.3f}, p_raw={row['p_value_raw']:.2e}, "
              f"p_bonf={row['p_value_bonf']:.2e}  {sig}")
        print(f"    chi2={row['chi2']:.3f}, chi2_p={row['chi2_p_raw']:.2e}")
        print()

    # Note about in-progress data
    print("-" * 70)
    print("CAVEAT")
    print("-" * 70)
    print("  These results are INTERIM -- the eval is still running.")
    print("  Different models have completed different numbers of epochs:")
    for _, row in summary.iterrows():
        pct = row["n_epochs"] / 50 * 100
        print(f"    {row['model']:25s}: {row['n_epochs']}/50 epochs ({pct:.0f}% complete)")
    print("  Final results may differ once all 50 epochs complete.")
    print("  The Wilson CIs and Bonferroni corrections are valid for the")
    print("  available data but power will increase with more epochs.")
    print()

    # Generate plots
    print("-" * 70)
    print("GENERATING PLOTS")
    print("-" * 70)
    plot_refusal_bar(summary, boot, OUTPUT_DIR)
    plot_epoch_strip(df, OUTPUT_DIR)

    # Save summary tables as CSV for reference
    summary.to_csv(OUTPUT_DIR / "summary_stats.csv", index=False)
    tests.to_csv(OUTPUT_DIR / "pairwise_tests.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'summary_stats.csv'}")
    print(f"Saved: {OUTPUT_DIR / 'pairwise_tests.csv'}")

    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
