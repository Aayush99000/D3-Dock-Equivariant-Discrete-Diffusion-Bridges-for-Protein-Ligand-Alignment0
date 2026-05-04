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
        "--annotation-parquet",
        type=str,
        default="",
        help=(
            "Path to local annotation_table.parquet. When provided, the parquet is "
            "read directly (column-selective, memory-safe) instead of using the "
            "plinder.core API."
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


def _load_plindex_from_parquet(parquet_path: str) -> pd.DataFrame:
    """
    Read annotation_table.parquet directly, loading only the columns we need.
    Memory-safe: reads ~3 columns out of 743 via pyarrow column selection.
    """
    import pyarrow.parquet as pq

    needed = [
        "system_id",
        "entry_resolution",
        "ligand_molecular_weight",
        "entry_pdb_id",
    ]
    pf = pq.ParquetFile(parquet_path)
    available = {pf.schema_arrow.field(i).name for i in range(len(pf.schema_arrow))}
    cols = [c for c in needed if c in available]
    LOG.info("Reading parquet columns: %s", cols)
    return pf.read(columns=cols).to_pandas()


def _load_plindex() -> pd.DataFrame:
    """
    Load the full PLINDER annotation index via plinder.core API (plinder>=0.2).
    Falls back gracefully; callers should prefer _load_plindex_from_parquet when
    the local parquet path is known.
    """
    try:
        from plinder.core import get_plindex  # type: ignore
        df = get_plindex()
        return df if isinstance(df, pd.DataFrame) else df.to_pandas()
    except Exception:
        pass

    try:
        from plinder.core.scores import query_index  # type: ignore
        df = query_index(
            columns=["system_id", "entry_resolution", "ligand_molecular_weight"],
            filters=[("entry_resolution", "<", 999.0)],
        )
        if df is not None:
            return df if isinstance(df, pd.DataFrame) else df.to_pandas()
    except Exception:
        pass

    raise ImportError(
        "Could not load PLINDER index. Pass --annotation-parquet to read locally."
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

    if args.annotation_parquet:
        LOG.info("Loading PLINDER index from local parquet: %s", args.annotation_parquet)
        df = _load_plindex_from_parquet(args.annotation_parquet)
    else:
        LOG.info("Loading PLINDER index via plinder.core API")
        df = _load_plindex()
    LOG.info("Loaded index rows=%d cols=%d", len(df), len(df.columns))

    # Resolve key schema columns with robust aliases.
    plinder_id_col = _pick_column(
        df.columns,
        candidates=["system_id", "plinder_id", "id"],
        purpose="PLINDER system id",
    )
    resolution_col = _pick_column(
        df.columns,
        candidates=["entry_resolution", "resolution", "resolution_angstrom", "pdb_resolution"],
        purpose="resolution",
    )
    ligand_mw_col = _pick_column(
        df.columns,
        candidates=["ligand_molecular_weight", "system_ligand_max_molecular_weight",
                    "ligand_mw", "mol_wt", "mw"],
        purpose="ligand molecular weight",
    )

    # Apo availability: join against the apo links parquet if no has_apo column exists.
    apo_col = None
    for candidate in ["has_apo", "apo_available", "apo_present", "apo",
                      "system_has_apo_ligand", "num_apo_chains"]:
        if candidate in df.columns:
            apo_col = candidate
            break

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
    if apo_col is not None:
        if pd.api.types.is_bool_dtype(df[apo_col]) or set(df[apo_col].dropna().unique()) <= {True, False}:
            df = df[df[apo_col] == True].copy()  # noqa: E712
        else:
            normalized = (
                df[apo_col].astype(str).str.strip().str.lower()
                .isin({"1", "true", "t", "yes", "y", "apo"})
            )
            df = df[normalized].copy()
        _log_discard("Has Apo structure (column)", before=before, after=len(df))
    else:
        # Derive apo availability from the downloaded apo links parquet.
        plinder_mount = os.environ.get("PLINDER_MOUNT", "")
        release = os.environ.get("PLINDER_RELEASE", "2024-06")
        iteration = os.environ.get("PLINDER_ITERATION", "v2")
        apo_links_path = os.path.join(
            plinder_mount, release, iteration, "links", "kind=apo", "links.parquet"
        )
        if os.path.exists(apo_links_path):
            apo_df = pd.read_parquet(apo_links_path, columns=["reference_system_id"])
            apo_ids = set(apo_df["reference_system_id"].dropna().astype(str).unique())
            df = df[df[plinder_id_col].astype(str).isin(apo_ids)].copy()
            _log_discard("Has Apo structure (links parquet)", before=before, after=len(df))
        else:
            LOG.warning(
                "No apo column found and apo links parquet missing at %s — skipping apo filter.",
                apo_links_path,
            )

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
