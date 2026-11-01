#!/usr/bin/env python3
"""
plot_recall_ci.py
Visualizes discrete retrieval benchmarks with 95% Wilson Confidence Intervals.
Separates discrete recall from continuous logit metrics.
Data source: docs/paper/data/raw_m5_acceptance_t1_t7.csv
"""
import os
import sys
import tempfile
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

import matplotlib.pyplot as plt

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(script_dir, "figure3_recall_wilson_ci.png")

    scenarios = [
        "T6: In-Memory Dense Attention Baseline (n=10)",
        "T5: 128-d Token Hash Ranker Top-1 (n=20)",
        "T6: Scratch-Sequence KV Restore (n=10)",
        "T6: Sliding Window-Only Baseline (n=10)",
    ]

    p = [20.0, 10.0, 0.0, 0.0]
    ci_lower = [5.7, 2.8, 0.0, 0.0]
    ci_upper = [51.2, 30.1, 28.3, 28.3]

    err_low = [pt - lo for pt, lo in zip(p, ci_lower)]
    err_high = [up - pt for pt, up in zip(p, ci_upper)]
    asymmetric_err = [err_low, err_high]

    colors = ["#6c757d", "#ffc107", "#dc3545", "#dc3545"]

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)

    y_pos = range(len(scenarios))
    ax.barh(y_pos, p, xerr=asymmetric_err, align="center", color=colors, alpha=0.85, edgecolor="black", capsize=5)

    # Random chance line for T5 (1/20 = 5.0%)
    ax.axvline(5.0, color="#d9534f", linestyle=":", linewidth=1.5, label="Random Chance Baseline for T5 (5.0%)")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(scenarios, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel("Retrieval Accuracy (%) with 95% Wilson Confidence Interval", fontsize=10)
    ax.set_xlim(0, 70)
    ax.set_title("Discrete Needle Retrieval Accuracy & Confidence Intervals (T5 & T6)\n(Negative & Inconclusive Empirical Results Highlighted)", fontsize=11, fontweight="bold", pad=12)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=8.5)

    # Point annotations
    for i, (pt, lo, up) in enumerate(zip(p, ci_lower, ci_upper)):
        ax.text(up + 2.0, i, f"{pt:.1f}% [{lo:.1f}%, {up:.1f}%]", va="center", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Successfully generated figure: {out_path}")

if __name__ == "__main__":
    main()
