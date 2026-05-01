#!/usr/bin/env python3
"""
Partition processed PyG tensors by PLINDER PL50 split and verify leakage.

What it does:
1) Reads PLINDER split parquet (PL50 split labels).
2) Maps local processed .pt files to plinder_id.
3) Writes:
   - train.txt
   - val.txt
   - test.txt
   with absolute .pt paths.
4) Verifies no test protein has >40% sequence identity to any train protein.
   It uses the best available method in this order:
   A) precomputed parquet column (max identity to train),
   B) 40%-cluster non-overlap,
   C) explicit sequence alignment from sequence columns.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from Bio import pairwise2

LOG = logging.getLogger("d3dock_partition_pl50")


ID_COL_CANDIDATES = ["plinder_id", "system_id", "id"]
SPLIT_COL_CANDIDATES = ["pl50_split", "split", "partition", "subset", "fold"]
PROTEIN_ID_COL_CANDIDATES = [
    "protein_id",
    "receptor_id",
    "uniprot_id",
    "protein_chain_id",
    "pdb_id",
]
SEQ_COL_CANDIDATES = [
    "protein_sequence",
    "sequence",
    "receptor_sequence",
    "aa_sequence",
]
MAX_IDENTITY_TO_TRAIN_COL_CANDIDATES = [
    "max_seq_identity_to_train",
    "max_identity_to_train",
    "test_to_train_max_identity",
]
CLUSTER_40_COL_CANDIDATES = [
    "cluster_40",
    "seq_cluster_40",
    "pl40_cluster",
    "protein_cluster_40",
]


@dataclass
class VerificationResult:
    method: str
    passed: bool
    violations: int
    details: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Partition local PyG .pt tensors by PLINDER PL50 split with leakage checks."
    )
    p.add_argument(
        "--processed-dir",
        required=True,
        help="Directory containing generated PyG .pt files (recursive search).",
    )
    p.add_argument(
        "--split-parquet",
        required=True,
        help="PLINDER split parquet file path.",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write train/val/test txt lists.",
    )
    p.add_argument(
        "--pt-glob",
        default="**/*.pt",
        help="Glob pattern under processed-dir for tensor files (default: **/*.pt).",
    )
    p.add_argument(
        "--id-regex",
        default=r"^([A-Za-z0-9_\-]+)",
        help=(
            "Regex to extract plinder_id from .pt basename stem. "
            "Group 1 should capture the ID."
        ),
    )
    p.add_argument(
        "--identity-threshold",
        type=float,
        default=0.40,
        help="Maximum allowed sequence identity between test and train proteins.",
    )
    p.add_argument(
        "--max-align-test-proteins",
        type=int,
        default=3000,
        help=(
            "Safety cap for explicit sequence-alignment verification path. "
            "If exceeded, script aborts unless precomputed/cluster checks are available."
        ),
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return p.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def pick_column(columns: Iterable[str], candidates: list[str], purpose: str) -> str:
    colset = set(columns)
    for c in candidates:
        if c in colset:
            return c
    raise KeyError(
        f"Could not find column for {purpose}. Tried {candidates}. "
        f"Available columns: {sorted(colset)}"
    )


def normalize_split_label(v: object) -> str:
    s = str(v).strip().lower()
    if s in {"train", "tr", "training"}:
        return "train"
    if s in {"val", "valid", "validation", "dev"}:
        return "val"
    if s in {"test", "te", "testing"}:
        return "test"
    return "unknown"


def collect_pt_files(processed_dir: str, pattern: str, id_regex: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(processed_dir, pattern), recursive=True))
    rex = re.compile(id_regex)
    rows = []
    for p in files:
        base = Path(p).stem
        m = rex.match(base)
        if not m:
            continue
        pid = m.group(1)
        rows.append({"plinder_id": pid, "tensor_path": str(Path(p).resolve())})
    if not rows:
        raise ValueError(
            f"No .pt files matched under {processed_dir} with pattern {pattern} and id_regex {id_regex}"
        )
    out = pd.DataFrame(rows).drop_duplicates(subset=["plinder_id", "tensor_path"])
    return out


def write_split_list(out_path: str, paths: list[str]) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for p in paths:
            f.write(f"{p}\n")


def verify_with_precomputed_col(
    merged: pd.DataFrame, threshold: float
) -> Optional[VerificationResult]:
    for c in MAX_IDENTITY_TO_TRAIN_COL_CANDIDATES:
        if c in merged.columns:
            test = merged[merged["_split_norm"] == "test"].copy()
            vals = pd.to_numeric(test[c], errors="coerce")
            violations = int((vals > threshold).sum())
            return VerificationResult(
                method=f"precomputed:{c}",
                passed=violations == 0,
                violations=violations,
                details=f"Checked {len(test)} test rows using '{c}'",
            )
    return None


def verify_with_cluster_40(merged: pd.DataFrame) -> Optional[VerificationResult]:
    cluster_col = None
    for c in CLUSTER_40_COL_CANDIDATES:
        if c in merged.columns:
            cluster_col = c
            break
    if cluster_col is None:
        return None

    tr = merged[merged["_split_norm"] == "train"][cluster_col].dropna().astype(str)
    te = merged[merged["_split_norm"] == "test"][cluster_col].dropna().astype(str)
    train_clusters = set(tr.tolist())
    test_clusters = set(te.tolist())
    overlap = train_clusters.intersection(test_clusters)
    return VerificationResult(
        method=f"cluster40:{cluster_col}",
        passed=len(overlap) == 0,
        violations=len(overlap),
        details=f"Overlapping 40%-clusters: {len(overlap)}",
    )


def seq_identity(a: str, b: str) -> float:
    """
    Global sequence identity based on pairwise alignment.
    Returns matches / max(len(a), len(b)).
    """
    if len(a) == 0 or len(b) == 0:
        return 0.0
    aln = pairwise2.align.globalxx(a, b, one_alignment_only=True, score_only=False)
    if not aln:
        return 0.0
    matches = aln[0].score
    return float(matches) / float(max(len(a), len(b)))


def verify_with_sequences(
    merged: pd.DataFrame, threshold: float, max_align_test_proteins: int
) -> Optional[VerificationResult]:
    seq_col = None
    for c in SEQ_COL_CANDIDATES:
        if c in merged.columns:
            seq_col = c
            break
    if seq_col is None:
        return None

    prot_col = None
    for c in PROTEIN_ID_COL_CANDIDATES:
        if c in merged.columns:
            prot_col = c
            break

    # If no explicit protein id is present, fall back to plinder_id uniqueness.
    key_col = prot_col if prot_col is not None else "plinder_id"

    tr = merged[merged["_split_norm"] == "train"][[key_col, seq_col]].dropna().drop_duplicates(key_col)
    te = merged[merged["_split_norm"] == "test"][[key_col, seq_col]].dropna().drop_duplicates(key_col)

    if len(te) > max_align_test_proteins:
        raise RuntimeError(
            f"Explicit alignment verification would process {len(te)} test proteins "
            f"(> --max-align-test-proteins={max_align_test_proteins}). "
            "Use precomputed identity columns or cluster columns in split parquet."
        )

    train_sequences = [str(x) for x in tr[seq_col].tolist()]
    violations = 0

    for _, row in te.iterrows():
        q = str(row[seq_col])
        max_id = 0.0
        for ref in train_sequences:
            ident = seq_identity(q, ref)
            if ident > max_id:
                max_id = ident
            if max_id > threshold:
                violations += 1
                break

    return VerificationResult(
        method=f"alignment:{seq_col}",
        passed=violations == 0,
        violations=violations,
        details=f"Aligned {len(te)} test vs {len(train_sequences)} train sequences",
    )


def run_verification(
    merged: pd.DataFrame, threshold: float, max_align_test_proteins: int
) -> VerificationResult:
    for fn in (
        lambda: verify_with_precomputed_col(merged, threshold),
        lambda: verify_with_cluster_40(merged),
        lambda: verify_with_sequences(merged, threshold, max_align_test_proteins),
    ):
        result = fn()
        if result is not None:
            return result

    raise RuntimeError(
        "Could not run leakage verification: no suitable columns found "
        "(precomputed identity, 40%-cluster, or sequence columns missing)."
    )


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    LOG.info("Reading split parquet: %s", args.split_parquet)
    split_df = pd.read_parquet(args.split_parquet)
    id_col = pick_column(split_df.columns, ID_COL_CANDIDATES, "plinder id")
    split_col = pick_column(split_df.columns, SPLIT_COL_CANDIDATES, "split label")

    split_df = split_df.copy()
    split_df["plinder_id"] = split_df[id_col].astype(str)
    split_df["_split_norm"] = split_df[split_col].map(normalize_split_label)
    split_df = split_df[split_df["_split_norm"].isin({"train", "val", "test"})]
    split_df = split_df.drop_duplicates(subset=["plinder_id"])
    LOG.info("Loaded split rows (train/val/test): %d", len(split_df))

    local_df = collect_pt_files(args.processed_dir, args.pt_glob, args.id_regex)
    LOG.info("Discovered local .pt tensors: %d", len(local_df))

    merged = local_df.merge(split_df, on="plinder_id", how="left")
    missing_split = merged["_split_norm"].isna().sum()
    if missing_split > 0:
        LOG.warning(
            "Found %d local tensors without split assignment; they will be excluded.",
            int(missing_split),
        )
    merged = merged[merged["_split_norm"].isin({"train", "val", "test"})].copy()

    train_paths = merged.loc[merged["_split_norm"] == "train", "tensor_path"].tolist()
    val_paths = merged.loc[merged["_split_norm"] == "val", "tensor_path"].tolist()
    test_paths = merged.loc[merged["_split_norm"] == "test", "tensor_path"].tolist()

    train_txt = os.path.join(args.output_dir, "train.txt")
    val_txt = os.path.join(args.output_dir, "val.txt")
    test_txt = os.path.join(args.output_dir, "test.txt")
    write_split_list(train_txt, train_paths)
    write_split_list(val_txt, val_paths)
    write_split_list(test_txt, test_paths)

    LOG.info("Wrote %s (%d paths)", train_txt, len(train_paths))
    LOG.info("Wrote %s (%d paths)", val_txt, len(val_paths))
    LOG.info("Wrote %s (%d paths)", test_txt, len(test_paths))

    result = run_verification(
        merged=merged,
        threshold=args.identity_threshold,
        max_align_test_proteins=args.max_align_test_proteins,
    )
    LOG.info(
        "Leakage verification method=%s | passed=%s | violations=%d | %s",
        result.method,
        result.passed,
        result.violations,
        result.details,
    )
    if not result.passed:
        LOG.error(
            "FAILED: Test-to-train sequence identity exceeds %.2f for %d test proteins/clusters.",
            args.identity_threshold,
            result.violations,
        )
        return 1

    LOG.info("SUCCESS: PL50 partition and leakage verification completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
