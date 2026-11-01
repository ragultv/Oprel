#!/usr/bin/env python3
"""
plot_ttft_vs_length.py
Visualizes Time-To-First-Token (TTFT) scaling: Cold Prefill Baseline vs oprel SSD Warm Prefix Reuse.
Differentiates between Measured and Projected values.
Data source: docs/paper/data/raw_m6_prefix_reuse.csv and raw_m1_m2_prefill.csv
"""
import os
import sys
import tempfile
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

import matplotlib.pyplot as plt
import pandas as pd

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "raw_m6_prefix_reuse.csv")
    out_path = os.path.join(script_dir, "figure1_ttft_vs_context.png")

    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        sys.exit(1)

    df = pd.read_csv(data_path)

    # Split into measured and projected
    df_measured = df[df["metric_tag"] == "Measured"]
    df_projected = df[df["metric_tag"] == "Projected"]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    # Measured points
    ax.scatter(
        df_measured["total_history_tokens"],
        df_measured["ttft_cold_reprefill_sec"],
        color="#d9534f",
        s=60,
        zorder=4,
        label="Cold Prefill (Measured Points)",
    )
    ax.plot(
        df_measured["total_history_tokens"],
        df_measured["ttft_cold_reprefill_sec"],
        color="#d9534f",
        linestyle="-",
        linewidth=1.8,
    )

    # Projected cold line
    last_meas = df_measured.iloc[-1]
    proj_tokens = [last_meas["total_history_tokens"]] + df_projected["total_history_tokens"].tolist()
    proj_ttft = [last_meas["ttft_cold_reprefill_sec"]] + df_projected["ttft_cold_reprefill_sec"].tolist()
    ax.plot(
        proj_tokens,
        proj_ttft,
        color="#d9534f",
        linestyle="--",
        linewidth=1.8,
        label="Cold Prefill (Projected tokens/170.0)",
    )
    ax.scatter(
        df_projected["total_history_tokens"],
        df_projected["ttft_cold_reprefill_sec"],
        facecolors="none",
        edgecolors="#d9534f",
        s=60,
        linewidths=1.5,
        zorder=4,
    )

    # Warm prefix reuse (all measured points)
    ax.scatter(
        df["total_history_tokens"],
        df["ttft_oprel_ssd_warm_sec"],
        color="#0275d8",
        marker="s",
        s=50,
        zorder=5,
        label="oprel Warm Window Reuse (Measured)",
    )
    ax.plot(
        df["total_history_tokens"],
        df["ttft_oprel_ssd_warm_sec"],
        color="#0275d8",
        linestyle="-",
        linewidth=1.8,
    )

    ax.set_title("Time-To-First-Token (TTFT) Scaling vs Context Length\n(Gemma 3 4B on AMD Ryzen 5 5600H + NVIDIA GTX 1650)", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Total Prompt / History Tokens", fontsize=10)
    ax.set_ylabel("TTFT Latency (seconds, log scale)", fontsize=10)
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", fontsize=8.5)

    max_row = df.iloc[-1]
    ax.annotate(
        f"Warm Active Window Reuse: {max_row['ttft_oprel_ssd_warm_sec']}s\nvs Projected Cold Prefill: {max_row['ttft_cold_reprefill_sec']:.1f}s",
        xy=(max_row["total_history_tokens"], max_row["ttft_oprel_ssd_warm_sec"]),
        xytext=(max_row["total_history_tokens"] - 16000, max_row["ttft_oprel_ssd_warm_sec"] * 8),
        arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=5),
        fontsize=8.5,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="#f7f7f7", ec="#cccccc"),
    )

    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Successfully generated figure: {out_path}")

if __name__ == "__main__":
    main()
