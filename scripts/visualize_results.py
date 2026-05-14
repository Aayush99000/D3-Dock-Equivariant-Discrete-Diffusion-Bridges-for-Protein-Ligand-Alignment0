#!/usr/bin/env python3
"""
D3-Dock Results Visualization Script.

Generates plots for:
1. Training loss curves (all 5 losses over 100 epochs)
2. RMSD distribution — val vs test
3. Surface-Ligand Overlap metrics — val vs test
4. Per-system SLO breakdown (violation fraction, clean surface rate)
5. Clash-free rate summary bar chart

Usage:
    python scripts/visualize_results.py \
        --train-log  /scratch/katoch.aa/d3dock/logs/06_train_6613708.out \
        --val-csv    /scratch/katoch.aa/d3dock/outputs/eval_val/metrics.csv \
        --test-csv   /scratch/katoch.aa/d3dock/outputs/eval_test/metrics.csv \
        --val-json   /scratch/katoch.aa/d3dock/outputs/eval_val/summary.json \
        --test-json  /scratch/katoch.aa/d3dock/outputs/eval_test/summary.json \
        --output-dir ./outputs/figures
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

PALETTE = {
    "val":  "#4C72B0",
    "test": "#DD8452",
    "cont": "#55A868",
    "atom": "#C44E52",
    "bond": "#8172B2",
    "phys": "#937860",
    "loss": "#2d2d2d",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_train_log(log_path: str) -> pd.DataFrame:
    rows = []
    pattern = re.compile(
        r"\[Epoch\s+(\d+)\]\s+"
        r"loss=([\d.]+)\s+"
        r"cont=([\d.]+)\s+"
        r"atom=([\d.]+)\s+"
        r"bond=([\d.]+)\s+"
        r"phys=([\d.]+)"
    )
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                rows.append({
                    "epoch": int(m.group(1)),
                    "loss":  float(m.group(2)),
                    "cont":  float(m.group(3)),
                    "atom":  float(m.group(4)),
                    "bond":  float(m.group(5)),
                    "phys":  float(m.group(6)),
                })
    return pd.DataFrame(rows)


def savefig(fig: plt.Figure, path: Path, name: str) -> None:
    out = path / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── plot functions ────────────────────────────────────────────────────────────

def plot_training_curves(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("D3-Dock Training Loss Curves (100 Epochs)", fontsize=14, fontweight="bold")

    # Top: total loss
    ax = axes[0]
    ax.plot(df["epoch"], df["loss"], color=PALETTE["loss"], lw=2, label="Total loss")
    ax.set_ylabel("Total Loss", fontsize=11)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # Bottom: component losses
    ax = axes[1]
    for col, label in [("cont", "Coord (cont)"), ("atom", "Atom type"), ("bond", "Bond type"), ("phys", "Physics penalty")]:
        ax.plot(df["epoch"], df[col], color=PALETTE[col], lw=1.8, label=label)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Component Loss", fontsize=11)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    savefig(fig, out_dir, "01_training_curves.png")


def plot_training_curves_log(df: pd.DataFrame, out_dir: Path) -> None:
    """Same as above but y-axis in log scale — better for showing atom/bond convergence."""
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle("D3-Dock Training Loss (Log Scale)", fontsize=14, fontweight="bold")

    for col, label in [
        ("loss", "Total"), ("cont", "Coord"), ("atom", "Atom"), ("bond", "Bond"), ("phys", "Physics")
    ]:
        ax.semilogy(df["epoch"], df[col], color=PALETTE[col], lw=1.8, label=label)

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss (log scale)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    savefig(fig, out_dir, "02_training_curves_log.png")


def plot_rmsd_distribution(val_df: pd.DataFrame, test_df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("RMSD Distribution — Val vs Test", fontsize=14, fontweight="bold")

    for ax, df, split, color in [
        (axes[0], val_df,  "Val (39)",   PALETTE["val"]),
        (axes[1], test_df, "Test (111)", PALETTE["test"]),
    ]:
        rmsd = df["rmsd"].dropna()
        ax.hist(rmsd, bins=20, color=color, edgecolor="white", alpha=0.85)
        ax.axvline(rmsd.mean(),   color="black", lw=1.5, linestyle="--", label=f"Mean: {rmsd.mean():.1f}Å")
        ax.axvline(rmsd.median(), color="gray",  lw=1.5, linestyle=":",  label=f"Median: {rmsd.median():.1f}Å")
        ax.set_title(split, fontsize=12)
        ax.set_xlabel("RMSD (Å)", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    savefig(fig, out_dir, "03_rmsd_distribution.png")


def plot_slo_overview(val_json: dict, test_json: dict, out_dir: Path) -> None:
    metrics = {
        "Violation\nFraction":   ("slo_mean_violation_fraction",   True,  "Lower is better"),
        "Mean Overlap\nDepth (Å)": ("slo_mean_overlap_depth_A",    True,  "Lower is better"),
        "Max Overlap\nDepth (Å)": ("slo_mean_max_overlap_depth_A", True,  "Lower is better"),
        "Clean\nSurface Rate":   ("slo_mean_clean_surface_rate",   False, "Higher is better"),
        "Clash-Free\nRate":      ("slo_clash_free_rate",           False, "Higher is better"),
    }

    labels = list(metrics.keys())
    val_vals  = [val_json.get(v[0], 0) for v in metrics.values()]
    test_vals = [test_json.get(v[0], 0) for v in metrics.values()]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.suptitle("Surface-Ligand Overlap Metrics — Val vs Test", fontsize=14, fontweight="bold")

    bars_val  = ax.bar(x - width/2, val_vals,  width, label="Val",  color=PALETTE["val"],  alpha=0.85, edgecolor="white")
    bars_test = ax.bar(x + width/2, test_vals, width, label="Test", color=PALETTE["test"], alpha=0.85, edgecolor="white")

    for bar in list(bars_val) + list(bars_test):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, axis="y")

    # Annotate direction
    for i, (label, (key, lower_better, hint)) in enumerate(metrics.items()):
        ax.text(i, -0.12, hint, ha="center", fontsize=7, color="gray",
                transform=ax.get_xaxis_transform())

    fig.tight_layout()
    savefig(fig, out_dir, "04_slo_overview.png")


def plot_per_system_slo(val_df: pd.DataFrame, test_df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Per-System Surface-Ligand Overlap — Val & Test", fontsize=14, fontweight="bold")

    for row, (df, split, color) in enumerate([
        (val_df,  "Val",  PALETTE["val"]),
        (test_df, "Test", PALETTE["test"]),
    ]):
        df_sorted = df.sort_values("slo_violation_fraction")
        ids = range(len(df_sorted))

        # Violation fraction
        ax = axes[row][0]
        ax.bar(ids, df_sorted["slo_violation_fraction"], color=color, alpha=0.8, edgecolor="none")
        ax.axhline(df_sorted["slo_violation_fraction"].mean(), color="black", lw=1.5, linestyle="--",
                   label=f"Mean: {df_sorted['slo_violation_fraction'].mean():.3f}")
        ax.set_title(f"{split} — Violation Fraction per System", fontsize=11)
        ax.set_xlabel("System (sorted)", fontsize=10)
        ax.set_ylabel("Fraction of atoms inside protein", fontsize=10)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, axis="y")

        # Max overlap depth
        ax = axes[row][1]
        ax.bar(ids, df_sorted["slo_max_overlap_depth_A"], color=color, alpha=0.8, edgecolor="none")
        ax.axhline(df_sorted["slo_max_overlap_depth_A"].mean(), color="black", lw=1.5, linestyle="--",
                   label=f"Mean: {df_sorted['slo_max_overlap_depth_A'].mean():.3f}Å")
        ax.set_title(f"{split} — Max Overlap Depth per System", fontsize=11)
        ax.set_xlabel("System (sorted by violation)", fontsize=10)
        ax.set_ylabel("Max penetration depth (Å)", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    savefig(fig, out_dir, "05_per_system_slo.png")


def plot_slo_scatter(test_df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle("Test Set: Atoms vs Violations", fontsize=14, fontweight="bold")

    scatter = ax.scatter(
        test_df["slo_n_atoms"],
        test_df["slo_n_violations"],
        c=test_df["slo_max_overlap_depth_A"],
        cmap="RdYlGn_r",
        alpha=0.75,
        s=60,
        edgecolors="white",
        linewidths=0.5,
    )
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Max Overlap Depth (Å)", fontsize=10)

    ax.set_xlabel("Number of Heavy Atoms", fontsize=11)
    ax.set_ylabel("Number of Violations (atoms inside protein)", fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    savefig(fig, out_dir, "06_slo_scatter.png")


def plot_summary_card(val_json: dict, test_json: dict, train_df: pd.DataFrame, out_dir: Path) -> None:
    fig = plt.figure(figsize=(12, 5))
    fig.patch.set_facecolor("#f8f9fa")
    ax = fig.add_subplot(111)
    ax.axis("off")

    rows = [
        ["Metric", "Val (39 systems)", "Test (111 systems)"],
        ["─" * 30, "─" * 20, "─" * 20],
        ["Training epochs", "100", "100"],
        ["Final total loss", f"{train_df['loss'].iloc[-1]:.4f}", "—"],
        ["Final coord loss", f"{train_df['cont'].iloc[-1]:.4f}", "—"],
        ["Final atom loss",  f"{train_df['atom'].iloc[-1]:.4f}", "—"],
        ["Final bond loss",  f"{train_df['bond'].iloc[-1]:.4f}", "—"],
        ["─" * 30, "─" * 20, "─" * 20],
        ["Mean RMSD (Å)",    f"{val_json['mean_rmsd']:.1f}",   f"{test_json['mean_rmsd']:.1f}"],
        ["Median RMSD (Å)",  f"{val_json['median_rmsd']:.1f}", f"{test_json['median_rmsd']:.1f}"],
        ["Success rate (RMSD<2Å)", f"{val_json['success_rate_rmsd_lt_2A']:.1%}", f"{test_json['success_rate_rmsd_lt_2A']:.1%}"],
        ["─" * 30, "─" * 20, "─" * 20],
        ["SLO violation fraction",  f"{val_json['slo_mean_violation_fraction']:.3f}", f"{test_json['slo_mean_violation_fraction']:.3f}"],
        ["SLO mean overlap depth",  f"{val_json['slo_mean_overlap_depth_A']:.4f}Å",  f"{test_json['slo_mean_overlap_depth_A']:.4f}Å"],
        ["SLO max overlap depth",   f"{val_json['slo_mean_max_overlap_depth_A']:.4f}Å", f"{test_json['slo_mean_max_overlap_depth_A']:.4f}Å"],
        ["SLO clean surface rate",  f"{val_json['slo_mean_clean_surface_rate']:.1%}", f"{test_json['slo_mean_clean_surface_rate']:.1%}"],
        ["Clash-free pose rate",    f"{val_json['slo_clash_free_rate']:.1%}",         f"{test_json['slo_clash_free_rate']:.1%}"],
    ]

    table = ax.table(
        cellText=rows,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.6)

    # Header styling
    for j in range(3):
        table[0, j].set_facecolor("#2d2d2d")
        table[0, j].set_text_props(color="white", fontweight="bold")

    fig.suptitle("D3-Dock Evaluation Summary", fontsize=15, fontweight="bold", y=0.98)
    savefig(fig, out_dir, "00_summary_card.png")


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize D3-Dock training and evaluation results.")
    p.add_argument("--train-log",  required=True, help="Path to training .out log file.")
    p.add_argument("--val-csv",    required=True, help="Val metrics.csv from validate_test_d3dock.py.")
    p.add_argument("--test-csv",   required=True, help="Test metrics.csv from validate_test_d3dock.py.")
    p.add_argument("--val-json",   required=True, help="Val summary.json.")
    p.add_argument("--test-json",  required=True, help="Test summary.json.")
    p.add_argument("--output-dir", default="outputs/figures", help="Directory to save figures.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving figures to: {out_dir}")

    print("Loading data...")
    train_df = parse_train_log(args.train_log)
    val_df   = pd.read_csv(args.val_csv)
    test_df  = pd.read_csv(args.test_csv)
    with open(args.val_json)  as f: val_json  = json.load(f)
    with open(args.test_json) as f: test_json = json.load(f)

    print(f"  Training epochs: {len(train_df)}")
    print(f"  Val systems: {len(val_df)} | Test systems: {len(test_df)}")

    print("Generating plots...")
    plot_summary_card(val_json, test_json, train_df, out_dir)
    plot_training_curves(train_df, out_dir)
    plot_training_curves_log(train_df, out_dir)
    plot_rmsd_distribution(val_df, test_df, out_dir)
    plot_slo_overview(val_json, test_json, out_dir)
    plot_per_system_slo(val_df, test_df, out_dir)
    plot_slo_scatter(test_df, out_dir)

    print(f"\nDone. {len(list(out_dir.glob('*.png')))} figures saved to {out_dir}")


if __name__ == "__main__":
    main()
