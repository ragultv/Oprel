#!/usr/bin/env python3
"""
plot_vram_stack.py
Visualizes GPU VRAM breakdown and headroom on GTX 1650 (3286 MiB initial free VRAM).
Data source: docs/paper/data/raw_m4_memory.csv
"""
import os
import sys
import tempfile
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

import matplotlib.pyplot as plt

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(script_dir, "figure2_vram_stack.png")

    categories = ["Initial Free VRAM", "Gemma 3 4B Base Alloc", "oprel With Active Window"]
    
    # Allocations
    weights = [0.0, 2692.37, 2692.37]
    compute = [0.0, 517.13, 517.13]
    active_kv = [0.0, 0.0, 95.63]
    headroom = [3286.00, 3286.00 - (2692.37 + 517.13), 0.0]

    fig, ax = plt.subplots(figsize=(7.5, 6), dpi=300)

    bar_width = 0.45
    indices = [0, 1, 2]

    ax.bar(indices, weights, bar_width, label="Model Weights (CUDA0 35 layers: 2,692 MiB)", color="#2b5c8f")
    ax.bar(indices, compute, bar_width, bottom=weights, label="CUDA0 Compute Scratch Buffer (517 MiB)", color="#d9822b")
    ax.bar(indices, active_kv, bar_width, bottom=[w + c for w, c in zip(weights, compute)], label="Active Q4_0 KV Cache (2.5k: 95.6 MiB)", color="#3b9a57")
    ax.bar(indices, headroom, bar_width, bottom=[w + c + k for w, c, k in zip(weights, compute, active_kv)], label="Free Margin before KV (76.5 MiB)", color="#e0e0e0")

    ax.axhline(3286.00, color="red", linestyle="--", linewidth=1.5, label="Initial Free VRAM at Load (3,286 MiB)")
    ax.axhline(4095.00, color="gray", linestyle=":", linewidth=1.2, label="Hardware VRAM Ceiling (4,095 MiB)")

    ax.set_title("VRAM Allocation on NVIDIA GTX 1650 (Commodity 4GB Edge)\nBaseline Free VRAM: 3,286 MiB", fontsize=11, fontweight="bold", pad=12)
    ax.set_ylabel("VRAM Allocation (MiB)", fontsize=10)
    ax.set_xticks(indices)
    ax.set_xticklabels(categories, fontsize=9.5, fontweight="bold")
    ax.set_ylim(0, 4400)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=8)

    # Text annotations
    ax.text(1, 2692.37 / 2, "2,692 MiB\nWeights", ha="center", va="center", color="white", fontweight="bold", fontsize=9)
    ax.text(1, 2692.37 + 517.13 / 2, "517 MiB", ha="center", va="center", color="white", fontweight="bold", fontsize=8.5)
    ax.text(1, 3209.50 + 38, "76.5 MiB", ha="center", va="center", color="#333333", fontweight="bold", fontsize=8)
    ax.text(2, 2692.37 + 517.13 + 95.63 / 2, "95.6 MiB (KV)", ha="center", va="center", color="white", fontweight="bold", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Successfully generated figure: {out_path}")

if __name__ == "__main__":
    main()
