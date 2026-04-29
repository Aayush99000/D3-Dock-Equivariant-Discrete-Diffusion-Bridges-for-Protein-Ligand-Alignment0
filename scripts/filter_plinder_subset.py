#!/usr/bin/env python3
"""
Filter PLINDER systems to a high-quality D3-Dock subset.

Filters applied:
1) Resolution < 2.5 A
2) Must have associated Apo structure
3) Ligand molecular weight in [100, 800] Da

Outputs a CSV with:
- plinder_id
- file_path
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Iterable

import pandas as pd


LOG = logging.getLogger("d3dock_plinder_filter")


def _pick_column(columns: Iterable[str], candidates: list[str], purpose: str) -> str:
    """Select the first matching column name from candidates."""
    column_set = set(columns)
    for candidate in candidates:
        if candidate in column_set:
            return candidate
    raise KeyError(
        f"Could not find a column for {purpose}. "
        f"Tried candidates: {candidates}. "
        f"Available columns: {sorted(column_set)}"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Filter PLINDER plindex for high-quality D3-Dock systems."
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="filtered_plinder_systems.csv",
        help="Path for filtered CSV output.",
    )
    parser.add_argument(
        "--max-resolution",
        type=float,
        default=2.5,
        help="Maximum allowed crystallographic resolution (Angstrom).",
    )
    parser.add_argument(
        "--min-ligand-mw",
        type=float,
        default=100.0,
        help="Minimum ligand molecular weight (Da).",
    )
    parser.add_argument(
        "--max-ligand-mw",
        type=float,
        default=800.0,
        help="Maximum ligand molecular weight (Da).",
    )
    parser.add_argument(
        "--plinder-data-root",
        type=str,
        default=os.environ.get("PLINDER_DATA_ROOT", ""),
        help=(
            "Optional PLINDER data root used to build fallback file_path values "
            "as <root>/<plinder_id> when no path column exists."
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )
    return parser


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _load_plindex() -> pd.DataFrame:
    """
    Load the PLINDER index with plinder.core API.

    This uses query_index() and converts the result to pandas for efficient filtering.
    """
    try:
        from plinder.core import query_index  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Could not import 'query_index' from plinder.core. "
            "Install/activate the PLINDER environment first."
        ) from exc

    index_obj = query_index()

    if isinstance(index_obj, pd.DataFrame):
        return index_obj

    if hasattr(index_obj, "to_pandas"):
        return index_obj.to_pandas()

    if hasattr(index_obj, "to_df"):
        return index_obj.to_df()

    raise TypeError(
        f"Unsupported query_index() return type: {type(index_obj)}. "
        "Expected pandas DataFrame or an object with to_pandas()/to_df()."
    )


def _log_discard(step_name: str, before: int, after: int) -> None:
    removed = before - after
    pct = (removed / before * 100.0) if before else 0.0
    LOG.info(
        "%s | kept=%d removed=%d (%.2f%% removed)",
        step_name,
        after,
        removed,
        pct,
    )


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    _configure_logging(args.log_level)

    LOG.info("Loading PLINDER index via plinder.core.query_index()")
    df = _load_plindex()
    LOG.info("Loaded index rows=%d cols=%d", len(df), len(df.columns))

    # Resolve key schema columns with robust aliases.
    plinder_id_col = _pick_column(
        df.columns,
        candidates=["plinder_id", "system_id", "id"],
        purpose="PLINDER system id",
    )
    resolution_col = _pick_column(
        df.columns,
        candidates=["resolution", "resolution_angstrom", "pdb_resolution"],
        purpose="resolution",
    )
    apo_col = _pick_column(
        df.columns,
        candidates=["has_apo", "apo_available", "apo_present", "apo"],
        purpose="apo-availability flag",
    )
    ligand_mw_col = _pick_column(
        df.columns,
        candidates=["ligand_mw", "ligand_molecular_weight", "mol_wt", "mw"],
        purpose="ligand molecular weight",
    )

    # Convert to numeric where needed, preserving NaN for non-parsable values.
    df[resolution_col] = pd.to_numeric(df[resolution_col], errors="coerce")
    df[ligand_mw_col] = pd.to_numeric(df[ligand_mw_col], errors="coerce")

    total = len(df)
    LOG.info("Initial systems: %d", total)

    # 1) Resolution filter.
    before = len(df)
    df = df[df[resolution_col] < args.max_resolution].copy()
    _log_discard(
        f"Resolution < {args.max_resolution}A",
        before=before,
        after=len(df),
    )

    # 2) Apo availability filter.
    before = len(df)
    if pd.api.types.is_bool_dtype(df[apo_col]) or set(df[apo_col].dropna().unique()) <= {
        True,
        False,
    }:
        df = df[df[apo_col] == True].copy()  # noqa: E712
    else:
        normalized = (
            df[apo_col]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin({"1", "true", "t", "yes", "y", "apo"})
        )
        df = df[normalized].copy()
    _log_discard("Has Apo structure", before=before, after=len(df))

    # 3) Ligand molecular weight filter.
    before = len(df)
    mw_mask = (df[ligand_mw_col] >= args.min_ligand_mw) & (
        df[ligand_mw_col] <= args.max_ligand_mw
    )
    df = df[mw_mask].copy()
    _log_discard(
        f"Ligand MW in [{args.min_ligand_mw}, {args.max_ligand_mw}] Da",
        before=before,
        after=len(df),
    )

    # Resolve path column if present; otherwise fallback from root + plinder_id.
    path_col = None
    for candidate in [
        "file_path",
        "system_path",
        "path",
        "archive_path",
        "protein_ligand_path",
    ]:
        if candidate in df.columns:
            path_col = candidate
            break

    if path_col is not None:
        output_df = df[[plinder_id_col, path_col]].rename(
            columns={plinder_id_col: "plinder_id", path_col: "file_path"}
        )
    else:
        if not args.plinder_data_root:
            raise ValueError(
                "No path column found in plindex and --plinder-data-root is empty. "
                "Provide --plinder-data-root to synthesize file paths."
            )
        output_df = df[[plinder_id_col]].rename(columns={plinder_id_col: "plinder_id"})
        output_df["file_path"] = output_df["plinder_id"].map(
            lambda pid: os.path.join(args.plinder_data_root, str(pid))
        )
        LOG.warning(
            "No explicit path column found. Synthesized file_path as "
            "<plinder_data_root>/<plinder_id>."
        )

    output_df = output_df.drop_duplicates(subset=["plinder_id"]).reset_index(drop=True)
    output_df.to_csv(args.output_csv, index=False)

    LOG.info("Final kept systems: %d / %d", len(output_df), total)
    LOG.info("Wrote filtered CSV: %s", args.output_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
