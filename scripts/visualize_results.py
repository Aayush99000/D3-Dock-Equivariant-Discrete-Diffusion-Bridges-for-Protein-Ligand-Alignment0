#!/usr/bin/env python3
"""
D3-Dock Results Visualization Script.

Generates plots for:
1. Training loss curves (all 5 losses over N epochs)
2. RMSD distribution — val vs test
3. Surface-Ligand Overlap metrics — val vs test (skipped if SLO data absent)
4. Per-system SLO breakdown (violation fraction, clean surface rate)
5. Clash-free rate summary bar chart
6. v3 vs v4 comparison (if --v3-val-json and --v3-test-json provided)

Usage:
    python scripts/visualize_results.py \
        --train-logs  /scratch/.../logs/10_train_v4_8168604.out \
                      /scratch/.../logs/10_train_v4_8176693.out \
                      /scratch/.../logs/10_train_v4_8184216.out \
                      /scratch/.../logs/10_train_v4_8215107.out \
                      /scratch/.../logs/10_train_v4_8243672.out \
                      /scratch/.../logs/10_train_v4_8259118.out \
        --val-csv     /scratch/.../eval_val_v4_T200/metrics.csv \
        --test-csv    /scratch/.../eval_test_v4_T200/metrics.csv \
        --val-json    /scratch/.../eval_val_v4_T200/summary.json \
        --test-json   /scratch/.../eval_test_v4_T200/summary.json \
        --output-dir  ./outputs/figures_v4
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    "v3":   "#999999",
    "v4":   "#2ca02c",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_train_logs(log_paths: list[str]) -> pd.DataFrame:
    """Parse one or more SLURM training logs, dedup by epoch, sort."""
    rows = []
    pattern = re.compile(
        r"\[Epoch\s+(\d+)\]\s+"
        r"loss=([\d.]+)\s+"
        r"cont=([\d.]+)\s+"
        r"atom=([\d.]+)\s+"
        r"bond=([\d.]+)\s+"
        r"phys=([\d.]+)"
    )
    for log_path in log_paths:
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
    df = pd.DataFrame(rows).drop_duplicates("epoch").sort_values("epoch").reset_index(drop=True)
    return df


def has_slo(df: pd.DataFrame) -> bool:
    return "slo_violation_fraction" in df.columns and df["slo_violation_fraction"].notna().any()


def savefig(fig: plt.Figure, path: Path, name: str) -> None:
    out = path / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── plot functions ────────────────────────────────────────────────────────────

def plot_training_curves(df: pd.DataFrame, out_dir: Path) -> None:
    n_epochs = df["epoch"].max()
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"D3-Dock Training Loss Curves ({n_epochs} Epochs)", fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.plot(df["epoch"], df["loss"], color=PALETTE["loss"], lw=2, label="Total loss")
    ax.set_ylabel("Total Loss", fontsize=11)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

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
        (axes[0], val_df,  f"Val ({len(val_df)})",   PALETTE["val"]),
        (axes[1], test_df, f"Test ({len(test_df)})", PALETTE["test"]),
    ]:
        rmsd = df["rmsd"].dropna()
        ax.hist(rmsd, bins=20, color=color, edgecolor="white", alpha=0.85)
        ax.axvline(rmsd.mean(),   color="black", lw=1.5, linestyle="--", label=f"Mean: {rmsd.mean():.2f}Å")
        ax.axvline(rmsd.median(), color="gray",  lw=1.5, linestyle=":",  label=f"Median: {rmsd.median():.2f}Å")
        ax.axvline(2.0, color="red", lw=1.5, linestyle="-", alpha=0.6, label="2Å threshold")
        n_success = (rmsd < 2.0).sum()
        ax.set_title(f"{split}  |  <2Å: {n_success}/{len(rmsd)} ({n_success/len(rmsd):.1%})", fontsize=12)
        ax.set_xlabel("RMSD (Å)", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    savefig(fig, out_dir, "03_rmsd_distribution.png")


def plot_slo_overview(val_json: dict, test_json: dict, out_dir: Path) -> None:
    slo_keys = ["slo_mean_violation_fraction", "slo_mean_overlap_depth_A",
                "slo_mean_max_overlap_depth_A", "slo_mean_clean_surface_rate", "slo_clash_free_rate"]
    if not any(k in val_json for k in slo_keys):
        print("  Skipping SLO overview — no SLO data in summary JSON.")
        return

    metrics = {
        "Violation\nFraction":     ("slo_mean_violation_fraction",   "Lower is better"),
        "Mean Overlap\nDepth (Å)": ("slo_mean_overlap_depth_A",      "Lower is better"),
        "Max Overlap\nDepth (Å)":  ("slo_mean_max_overlap_depth_A",  "Lower is better"),
        "Clean\nSurface Rate":     ("slo_mean_clean_surface_rate",   "Higher is better"),
        "Clash-Free\nRate":        ("slo_clash_free_rate",           "Higher is better"),
    }

    labels   = list(metrics.keys())
    val_vals  = [val_json.get(v[0], 0)  for v in metrics.values()]
    test_vals = [test_json.get(v[0], 0) for v in metrics.values()]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.suptitle("Surface-Ligand Overlap Metrics — Val vs Test", fontsize=14, fontweight="bold")
    bars_val  = ax.bar(x - width/2, val_vals,  width, label="Val",  color=PALETTE["val"],  alpha=0.85, edgecolor="white")
    bars_test = ax.bar(x + width/2, test_vals, width, label="Test", color=PALETTE["test"], alpha=0.85, edgecolor="white")
    for bar in list(bars_val) + list(bars_test):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005, f"{h:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, axis="y")
    for i, (_, (_, hint)) in enumerate(metrics.items()):
        ax.text(i, -0.12, hint, ha="center", fontsize=7, color="gray", transform=ax.get_xaxis_transform())
    fig.tight_layout()
    savefig(fig, out_dir, "04_slo_overview.png")


def plot_per_system_slo(val_df: pd.DataFrame, test_df: pd.DataFrame, out_dir: Path) -> None:
    if not has_slo(val_df) and not has_slo(test_df):
        print("  Skipping per-system SLO — no SLO columns in metrics CSV.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Per-System Surface-Ligand Overlap — Val & Test", fontsize=14, fontweight="bold")
    for row, (df, split, color) in enumerate([(val_df, "Val", PALETTE["val"]), (test_df, "Test", PALETTE["test"])]):
        if not has_slo(df):
            continue
        df_sorted = df.sort_values("slo_violation_fraction")
        ids = range(len(df_sorted))
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


def plot_v3_v4_comparison(val_json: dict, test_json: dict, out_dir: Path,
                           v3_val_json: dict, v3_test_json: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("v3 vs v4 — RMSD and Success Rate", fontsize=14, fontweight="bold")

    splits = ["Val", "Test"]
    v3_rmsd = [v3_val_json["mean_rmsd"], v3_test_json["mean_rmsd"]]
    v4_rmsd = [val_json["mean_rmsd"],    test_json["mean_rmsd"]]
    v3_sr   = [v3_val_json["success_rate_rmsd_lt_2A"] * 100, v3_test_json["success_rate_rmsd_lt_2A"] * 100]
    v4_sr   = [val_json["success_rate_rmsd_lt_2A"] * 100,    test_json["success_rate_rmsd_lt_2A"] * 100]

    x = np.arange(2)
    w = 0.35

    ax = axes[0]
    ax.bar(x - w/2, v3_rmsd, w, label="v3 (64-dim, bond-only)", color=PALETTE["v3"], alpha=0.85)
    ax.bar(x + w/2, v4_rmsd, w, label="v4 (128-dim, radius graph)", color=PALETTE["v4"], alpha=0.85)
    for i, (v3, v4) in enumerate(zip(v3_rmsd, v4_rmsd)):
        ax.text(i - w/2, v3 + 0.05, f"{v3:.2f}", ha="center", fontsize=9)
        ax.text(i + w/2, v4 + 0.05, f"{v4:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(splits, fontsize=11)
    ax.set_ylabel("Mean RMSD (Å)", fontsize=11)
    ax.set_ylim(0, 6)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    ax.set_title("Mean RMSD", fontsize=12)

    ax = axes[1]
    ax.bar(x - w/2, v3_sr, w, label="v3", color=PALETTE["v3"], alpha=0.85)
    ax.bar(x + w/2, v4_sr, w, label="v4", color=PALETTE["v4"], alpha=0.85)
    for i, (v3, v4) in enumerate(zip(v3_sr, v4_sr)):
        ax.text(i - w/2, v3 + 0.2, f"{v3:.1f}%", ha="center", fontsize=9)
        ax.text(i + w/2, v4 + 0.2, f"{v4:.1f}%", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(splits, fontsize=11)
    ax.set_ylabel("Success Rate (<2Å)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    ax.set_title("Success Rate (<2Å)", fontsize=12)

    fig.tight_layout()
    savefig(fig, out_dir, "06_v3_v4_comparison.png")


def plot_summary_card(val_json: dict, test_json: dict, train_df: pd.DataFrame, out_dir: Path) -> None:
    n_epochs = int(train_df["epoch"].max())
    fig = plt.figure(figsize=(12, 5))
    fig.patch.set_facecolor("#f8f9fa")
    ax = fig.add_subplot(111)
    ax.axis("off")

    has_slo_data = "slo_mean_violation_fraction" in val_json

    rows = [
        ["Metric", f"Val ({val_json['total_samples']} systems)", f"Test ({test_json['total_samples']} systems)"],
        ["─" * 30, "─" * 20, "─" * 20],
        ["Training epochs", str(n_epochs), "—"],
        ["Final coord loss", f"{train_df['cont'].iloc[-1]:.4f}", "—"],
        ["─" * 30, "─" * 20, "─" * 20],
        ["Mean RMSD (Å)",    f"{val_json['mean_rmsd']:.2f}",   f"{test_json['mean_rmsd']:.2f}"],
        ["Median RMSD (Å)",  f"{val_json['median_rmsd']:.2f}", f"{test_json['median_rmsd']:.2f}"],
        ["Success rate (<2Å)", f"{val_json['success_rate_rmsd_lt_2A']:.1%}", f"{test_json['success_rate_rmsd_lt_2A']:.1%}"],
    ]
    if has_slo_data:
        rows += [
            ["─" * 30, "─" * 20, "─" * 20],
            ["SLO violation fraction",  f"{val_json['slo_mean_violation_fraction']:.3f}", f"{test_json.get('slo_mean_violation_fraction', 'n/a')}"],
            ["SLO clean surface rate",  f"{val_json['slo_mean_clean_surface_rate']:.1%}",  f"{test_json.get('slo_mean_clean_surface_rate', 'n/a')}"],
            ["Clash-free pose rate",    f"{val_json['slo_clash_free_rate']:.1%}",           f"{test_json.get('slo_clash_free_rate', 'n/a')}"],
        ]

    table = ax.table(cellText=rows, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.6)
    for j in range(3):
        table[0, j].set_facecolor("#2d2d2d")
        table[0, j].set_text_props(color="white", fontweight="bold")

    fig.suptitle("D3-Dock v4 Evaluation Summary", fontsize=15, fontweight="bold", y=0.98)
    savefig(fig, out_dir, "00_summary_card.png")


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize D3-Dock training and evaluation results.")
    p.add_argument("--train-logs", nargs="+", required=True, help="One or more SLURM training .out log files.")
    p.add_argument("--val-csv",    required=True)
    p.add_argument("--test-csv",   required=True)
    p.add_argument("--val-json",   required=True)
    p.add_argument("--test-json",  required=True)
    p.add_argument("--v3-val-json",  default=None, help="Optional v3 val summary.json for comparison plot.")
    p.add_argument("--v3-test-json", default=None, help="Optional v3 test summary.json for comparison plot.")
    p.add_argument("--output-dir", default="outputs/figures_v4")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving figures to: {out_dir}")

    print("Loading data...")
    train_df = parse_train_logs(args.train_logs)
    val_df   = pd.read_csv(args.val_csv)
    test_df  = pd.read_csv(args.test_csv)
    with open(args.val_json)  as f: val_json  = json.load(f)
    with open(args.test_json) as f: test_json = json.load(f)

    print(f"  Training epochs parsed: {len(train_df)} (ep {train_df['epoch'].min()}–{train_df['epoch'].max()})")
    print(f"  Val systems: {len(val_df)} | Test systems: {len(test_df)}")

    print("Generating plots...")
    plot_summary_card(val_json, test_json, train_df, out_dir)
    plot_training_curves(train_df, out_dir)
    plot_training_curves_log(train_df, out_dir)
    plot_rmsd_distribution(val_df, test_df, out_dir)
    plot_slo_overview(val_json, test_json, out_dir)
    plot_per_system_slo(val_df, test_df, out_dir)

    if args.v3_val_json and args.v3_test_json:
        with open(args.v3_val_json)  as f: v3_val_json  = json.load(f)
        with open(args.v3_test_json) as f: v3_test_json = json.load(f)
        plot_v3_v4_comparison(val_json, test_json, out_dir, v3_val_json, v3_test_json)

    print(f"\nDone. {len(list(out_dir.glob('*.png')))} figures saved to {out_dir}")


if __name__ == "__main__":
    main()
