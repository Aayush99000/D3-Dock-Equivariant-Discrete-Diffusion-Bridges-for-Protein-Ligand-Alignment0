#!/usr/bin/env python3
"""
Parallel preprocessing for D3-Dock discrete chemical states.

Pipeline per system:
1) Resolve protein PDB + ligand SDF input paths.
2) Load/sanitize ligand with RDKit; discard on sanitize failure.
3) Protonate ligand + protein at pH 7.4 via OpenBabel.
4) Save outputs:
   - <plinder_id>.rdkit.sdf
   - <plinder_id>.clean.pdb

Resume support:
- Uses a checkpoint CSV to skip already-processed systems across reruns.
"""

from __future__ import annotations

import argparse
import csv
import glob
import logging
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

# Silence noisy RDKit parsing logs in large batch runs.
RDLogger.DisableLog("rdApp.*")

LOG = logging.getLogger("d3dock_preprocess_discrete")


@dataclass
class Job:
    plinder_id: str
    base_path: str


@dataclass
class JobResult:
    plinder_id: str
    status: str
    reason: str
    ligand_input: str
    protein_input: str
    ligand_output: str
    protein_output: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parallel RDKit/OpenBabel preprocessing for D3-Dock systems."
    )
    parser.add_argument(
        "--input-csv",
        required=True,
        help="CSV from filtering step (must include plinder_id and file_path columns).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write cleaned structures.",
    )
    parser.add_argument(
        "--checkpoint-csv",
        default="preprocess_checkpoint.csv",
        help="Checkpoint CSV used for resume support.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) - 1),
        help="Number of multiprocessing workers.",
    )
    parser.add_argument(
        "--ph",
        type=float,
        default=7.4,
        help="Target protonation pH for OpenBabel.",
    )
    parser.add_argument(
        "--ligand-glob",
        default="*.sdf",
        help="Glob pattern (relative to file_path) to locate ligand SDF.",
    )
    parser.add_argument(
        "--protein-glob",
        default="*.pdb",
        help="Glob pattern (relative to file_path) to locate protein PDB.",
    )
    parser.add_argument(
        "--obabel-bin",
        default="obabel",
        help="OpenBabel executable path.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logger verbosity.",
    )
    return parser


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def resolve_single_match(base_path: str, pattern: str, kind: str) -> str:
    matches = sorted(glob.glob(os.path.join(base_path, pattern)))
    if not matches:
        raise FileNotFoundError(f"No {kind} file matched pattern '{pattern}' in {base_path}")
    return matches[0]


def sanitize_and_protonate_ligand_rdkit(ligand_sdf: str, out_rdkit_sdf: str) -> None:
    """
    Sanitize with RDKit then add hydrogens via RDKit AddHs.
    Using RDKit for both steps preserves bond orders set during sanitization —
    OpenBabel can silently re-perceive and overwrite them when writing SDF.

    NOTE: EmbedMolecule is intentionally NOT called here — it regenerates
    coordinates from scratch and would destroy the crystal-frame positions.
    AddHs(addCoords=True) places H atoms relative to existing heavy atom
    coordinates without modifying them.
    """
    supplier = Chem.SDMolSupplier(ligand_sdf, sanitize=False, removeHs=False)
    if len(supplier) == 0 or supplier[0] is None:
        raise ValueError(f"Failed to parse ligand SDF: {ligand_sdf}")

    mol = supplier[0]
    Chem.SanitizeMol(mol)

    # Remove existing Hs to avoid duplicates, re-add with crystal-frame coords.
    mol = Chem.RemoveHs(mol)
    mol = Chem.AddHs(mol, addCoords=True)

    writer = Chem.SDWriter(out_rdkit_sdf)
    writer.write(mol)
    writer.close()


def protonate_protein_with_obabel(obabel_bin: str, in_pdb: str, out_pdb: str, ph: float) -> None:
    """Use OpenBabel only for protein protonation — it handles PDB residue chemistry well."""
    cmd = [obabel_bin, in_pdb, "-O", out_pdb, "-p", str(ph)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"OpenBabel failed for {in_pdb} -> {out_pdb}. "
            f"stderr={proc.stderr.strip()}"
        )


def process_job(job: Job, output_dir: str, ph: float, ligand_glob: str, protein_glob: str, obabel_bin: str) -> JobResult:
    plinder_id = job.plinder_id
    system_dir = job.base_path

    # Skip multi-chain assemblies whose IDs exceed the 255-byte Linux filename limit.
    if len(plinder_id) > 200:
        return JobResult(
            plinder_id=plinder_id,
            status="skipped",
            reason="id_too_long",
            ligand_input="",
            protein_input="",
            ligand_output="",
            protein_output="",
        )

    system_out_dir = os.path.join(output_dir, plinder_id)
    os.makedirs(system_out_dir, exist_ok=True)

    ligand_out = os.path.join(system_out_dir, f"{plinder_id}.rdkit.sdf")
    protein_out = os.path.join(system_out_dir, f"{plinder_id}.clean.pdb")

    try:
        from plinder.core import PlinderSystem  # type: ignore
        ps = PlinderSystem(system_id=plinder_id)
        protein_in = str(ps.receptor_pdb)
        ligand_sdf_map = ps.ligand_sdfs
        if not ligand_sdf_map:
            raise ValueError("No ligand SDFs found via PlinderSystem")
        ligand_in = str(list(ligand_sdf_map.values())[0])
        if not os.path.exists(protein_in):
            raise FileNotFoundError(f"receptor.pdb missing after download: {protein_in}")
        if not os.path.exists(ligand_in):
            raise FileNotFoundError(f"ligand SDF missing after download: {ligand_in}")
    except Exception as exc:
        return JobResult(
            plinder_id=plinder_id,
            status="failed",
            reason=f"input_resolution_error: {exc}",
            ligand_input="",
            protein_input="",
            ligand_output=ligand_out,
            protein_output=protein_out,
        )

    try:
        tmp_base = os.path.join(output_dir, ".tmp")
        os.makedirs(tmp_base, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f"d3dock_{plinder_id}_", dir=tmp_base) as tmpdir:
            # Ligand: RDKit sanitize + AddHs (preserves bond orders).
            rdkit_sdf_tmp = os.path.join(tmpdir, "ligand.rdkit.sdf")
            sanitize_and_protonate_ligand_rdkit(ligand_in, rdkit_sdf_tmp)

            # Protein: OpenBabel protonation at pH 7.4 (handles residue chemistry).
            protonated_protein_tmp = os.path.join(tmpdir, "protein.protonated.pdb")
            protonate_protein_with_obabel(obabel_bin, protein_in, protonated_protein_tmp, ph)

            shutil.move(rdkit_sdf_tmp, ligand_out)
            shutil.move(protonated_protein_tmp, protein_out)

        return JobResult(
            plinder_id=plinder_id,
            status="success",
            reason="ok",
            ligand_input=ligand_in,
            protein_input=protein_in,
            ligand_output=ligand_out,
            protein_output=protein_out,
        )
    except Exception as exc:
        return JobResult(
            plinder_id=plinder_id,
            status="failed",
            reason=f"processing_error: {exc}",
            ligand_input=ligand_in,
            protein_input=protein_in,
            ligand_output=ligand_out,
            protein_output=protein_out,
        )


def load_completed_ids(checkpoint_csv: str) -> set[str]:
    if not os.path.exists(checkpoint_csv):
        return set()
    done: set[str] = set()
    with open(checkpoint_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") == "success" and row.get("plinder_id"):
                done.add(row["plinder_id"])
    return done


def append_checkpoint_row(checkpoint_csv: str, result: JobResult) -> None:
    exists = os.path.exists(checkpoint_csv)
    with open(checkpoint_csv, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "plinder_id",
                "status",
                "reason",
                "ligand_input",
                "protein_input",
                "ligand_output",
                "protein_output",
            ],
        )
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "plinder_id": result.plinder_id,
                "status": result.status,
                "reason": result.reason,
                "ligand_input": result.ligand_input,
                "protein_input": result.protein_input,
                "ligand_output": result.ligand_output,
                "protein_output": result.protein_output,
            }
        )


def main() -> int:
    args = build_parser().parse_args()
    configure_logging(args.log_level)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not shutil_which(args.obabel_bin):
        LOG.error("OpenBabel executable not found: %s", args.obabel_bin)
        LOG.error("Load OpenBabel on HPC and/or pass --obabel-bin with full path.")
        return 2

    df = pd.read_csv(args.input_csv)
    if "plinder_id" not in df.columns or "file_path" not in df.columns:
        LOG.error("Input CSV must contain 'plinder_id' and 'file_path' columns.")
        return 2

    df = df[["plinder_id", "file_path"]].dropna().drop_duplicates(subset=["plinder_id"])
    total_rows = len(df)
    LOG.info("Loaded systems from CSV: %d", total_rows)

    completed = load_completed_ids(args.checkpoint_csv)
    if completed:
        LOG.info("Resume mode: %d systems already completed; skipping.", len(completed))

    pending_df = df[~df["plinder_id"].astype(str).isin(completed)].copy()
    pending = [
        Job(plinder_id=str(row.plinder_id), base_path=str(row.file_path))
        for row in pending_df.itertuples(index=False)
    ]
    LOG.info("Pending systems to process: %d", len(pending))

    if not pending:
        LOG.info("Nothing to process. Exiting.")
        return 0

    success_count = 0
    failed_count = 0

    worker_args = [
        (job, str(out_dir), args.ph, args.ligand_glob, args.protein_glob, args.obabel_bin)
        for job in pending
    ]

    with mp.Pool(processes=args.workers) as pool:
        for result in pool.starmap(process_job, worker_args, chunksize=16):
            append_checkpoint_row(args.checkpoint_csv, result)
            if result.status == "success":
                success_count += 1
            else:
                failed_count += 1
                LOG.warning("Discarding %s | %s", result.plinder_id, result.reason)

    LOG.info("Run summary | success=%d failed=%d total_attempted=%d", success_count, failed_count, len(pending))
    LOG.info("Discarded in this run: %d", failed_count)
    LOG.info("Checkpoint updated: %s", args.checkpoint_csv)
    return 0


def shutil_which(binary: str) -> Optional[str]:
    from shutil import which

    return which(binary)


if __name__ == "__main__":
    sys.exit(main())
