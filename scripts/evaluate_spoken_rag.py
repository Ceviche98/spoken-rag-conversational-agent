"""
Benchmark Runner & Visualization Generator for Spoken RAG Conversational Agent.
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from spoken_rag.spoken_asr import load_asr_evaluation_results, summarize_asr_models


def plot_spoken_rag_benchmarks(data_path: str, out_dir: str = "docs/assets") -> None:
    os.makedirs(out_dir, exist_ok=True)
    df = load_asr_evaluation_results(data_path)
    summary = summarize_asr_models(df)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), dpi=200)

    models = list(summary.keys())
    wers = [summary[m]["mean_wer"] * 100 for m in models]
    lats = [summary[m]["mean_lat"] for m in models]
    rtfs = [summary[m]["mean_rtf"] for m in models]
    sims = [summary[m]["mean_sim"] * 100 for m in models]

    # Panel 1: WER and Latency Comparison
    x = np.arange(len(models))
    w = 0.35
    ax1 = axes[0]
    ax2 = ax1.twinx()

    b1 = ax1.bar(x - w / 2, wers, w, label="Word Error Rate (WER %)", color="#d95f02")
    b2 = ax2.bar(x + w / 2, lats, w, label="Latency (seconds)", color="#7570b3")

    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontweight="bold")
    ax1.set_ylabel("WER (%)", color="#d95f02", fontweight="bold")
    ax2.set_ylabel("Latency (s)", color="#7570b3", fontweight="bold")
    ax1.set_title("Whisper ASR: Accuracy vs Inference Latency", fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    # Combine legends
    lines = [b1, b2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right")

    # Panel 2: WER vs Downstream Semantic Similarity
    axes[1].plot(wers, sims, "o-", color="#1b9e77", linewidth=2.5, markersize=8)
    for i, m in enumerate(models):
        axes[1].annotate(
            f"{m}\n({wers[i]:.1f}%, {sims[i]:.1f}%)",
            (wers[i], sims[i]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontweight="bold",
            fontsize=8.5,
        )

    axes[1].set_xlabel("Word Error Rate (WER %)", fontweight="bold")
    axes[1].set_ylabel("Semantic Retrieval Alignment (%)", fontweight="bold")
    axes[1].set_title("Acoustic Invariance: Downstream Semantic Alignment vs WER", fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    out_file = os.path.join(out_dir, "spoken_rag_benchmarks.png")
    plt.savefig(out_file, bbox_inches="tight")
    print(f"Generated benchmark plot at: {out_file}")


if __name__ == "__main__":
    csv_file = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "asr_evaluation_results.csv")
    )
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs", "assets"))
    plot_spoken_rag_benchmarks(csv_file, out_dir)
