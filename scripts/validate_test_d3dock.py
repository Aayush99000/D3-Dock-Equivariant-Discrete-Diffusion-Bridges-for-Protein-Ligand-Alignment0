#!/usr/bin/env python3
"""
Validation/testing script for D3-Dock.

Implements:
- Reverse diffusion inference loop (Schrodinger-Bridge style approximation).
- RMSD metrics vs ground-truth ligand poses.
- Geometry success rate (RMSD < 2.0 A).
- PoseBusters wrapper for physical validity checks.
- Surface-Ligand Overlap metric via trilinear SDF sampling.
- SDF export for PyMOL visualization.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.loader import DataLoader

from models.d3dock_model import D3DockModel
from scripts.d3dock_pyg_dataset import D3DockHeteroDataset
from scripts.train_d3dock import build_schedule


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate/test D3-Dock model.")
    p.add_argument("--checkpoint", required=True, help="Model checkpoint (.pt).")
    p.add_argument("--input-csv", required=True)
    p.add_argument("--structures-dir", required=True)
    p.add_argument("--crop-dir", required=True)
    p.add_argument("--output-dir", required=True)

    p.add_argument(
        "--split-list",
        default=None,
        help=(
            "Optional text file with tensor paths (test.txt/val.txt). "
            "Only plinder_ids present in this list are evaluated."
        ),
    )
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=2)

    p.add_argument("--T", type=int, default=1000)
    p.add_argument(
        "--schedule",
        type=str,
        default="cosine",
        choices=["cosine", "linear"],
    )
    p.add_argument("--beta-start", type=float, default=1e-4)
    p.add_argument("--beta-end", type=float, default=0.02)

    p.add_argument("--num-atom-classes", type=int, default=32)
    p.add_argument("--num-bond-classes", type=int, default=6)
    p.add_argument(
        "--rmsd-threshold",
        type=float,
        default=2.0,
        help="Success threshold for docking RMSD in Angstrom.",
    )
    p.add_argument(
        "--posebusters-bin",
        default="bust",
        help="PoseBusters CLI executable (default: bust).",
    )
    p.add_argument(
        "--surface-dir",
        default=None,
        help=(
            "Directory containing per-system .surface_awareness.npz files "
            "(output of 03_surface.sh). Required for Surface-Ligand Overlap metrics."
        ),
    )
    p.add_argument(
        "--overlap-violation-thresh",
        type=float,
        default=0.1,
        help="SDF penetration depth (Å) below which an atom is considered clean (default: 0.1).",
    )
    return p.parse_args()


def _load_model(args: argparse.Namespace, device: torch.device) -> D3DockModel:
    model = D3DockModel(
        ligand_input_dim=3,
        protein_input_dim=4,
        surface_input_dim=5,
        num_atom_classes=args.num_atom_classes,
        num_bond_classes=args.num_bond_classes,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def _extract_plinder_ids_from_split_list(split_list: str) -> set[str]:
    ids: set[str] = set()
    with open(split_list, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            # Parent directory name is the full plinder_id (e.g. 1a2c__1__1.B__1.D)
            ids.add(Path(s).parent.name)
    return ids


def _subset_dataset(dataset: D3DockHeteroDataset, keep_ids: set[str]) -> list[int]:
    idxs = []
    for i, rec in enumerate(dataset.records):
        if rec.plinder_id in keep_ids:
            idxs.append(i)
    return idxs


ATOM_CLASS_TO_Z = [
    6,   # C
    7,   # N
    8,   # O
    16,  # S
    15,  # P
    9,   # F
    17,  # Cl
    35,  # Br
    53,  # I
]

BOND_CLASS_TO_TYPE = {
    0: Chem.BondType.SINGLE,
    1: Chem.BondType.DOUBLE,
    2: Chem.BondType.TRIPLE,
    3: Chem.BondType.AROMATIC,
}


def _apply_discrete_predictions(
    mol: Chem.Mol,
    atom_logits: torch.Tensor,
    bond_logits: torch.Tensor,
    bond_edge_index: torch.Tensor,
) -> Chem.Mol:
    rw = Chem.RWMol(mol)

    atom_pred = atom_logits.argmax(dim=-1).detach().cpu().numpy().tolist()
    for i, cls in enumerate(atom_pred):
        z = ATOM_CLASS_TO_Z[cls % len(ATOM_CLASS_TO_Z)]
        rw.GetAtomWithIdx(i).SetAtomicNum(int(z))

    edge_idx = bond_edge_index.detach().cpu().numpy()
    bond_pred = bond_logits.argmax(dim=-1).detach().cpu().numpy().tolist()
    # Edge list is directed; only apply once for i<j.
    for k in range(edge_idx.shape[1]):
        i = int(edge_idx[0, k])
        j = int(edge_idx[1, k])
        if i >= j:
            continue
        b = rw.GetBondBetweenAtoms(i, j)
        if b is None:
            continue
        bt = BOND_CLASS_TO_TYPE.get(int(bond_pred[k]), Chem.BondType.SINGLE)
        b.SetBondType(bt)

    out = rw.GetMol()
    try:
        Chem.SanitizeMol(out)
    except Exception:
        # Keep best-effort geometry even if discrete update is imperfect.
        out = mol
    return out


def reverse_diffusion_sample(
    model: D3DockModel,
    data,
    T: int,
    schedule_name: Literal["cosine", "linear"],
    beta_start: float,
    beta_end: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reverse process (DDPM-like bridge approximation):
      x_{t-1} = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1-a_bar_t) * eps_theta) + sigma_t*z
    """
    sched = build_schedule(
        T=T,
        schedule=schedule_name,
        beta_start=beta_start,
        beta_end=beta_end,
        device=device,
    )

    lig = data["ligand"]
    edge = data["ligand", "bond", "ligand"]
    x_t = torch.randn_like(lig.pos)  # start from noise

    # Keep noisy discrete states in first channels for iterative refinement.
    lig.x[:, 0] = torch.randint_like(lig.x[:, 0].long(), low=1, high=10).float()
    if edge.edge_attr.size(1) > 0:
        edge.edge_attr[:, 0] = torch.randint_like(
            edge.edge_attr[:, 0].long(), low=0, high=4
        ).float()

    atom_logits = None
    bond_logits = None
    bond_edge_index = edge.edge_index

    for t in reversed(range(T)):
        lig.pos = x_t
        out = model(data)
        eps_theta = out.coord_noise

        alpha_t = sched.alphas[t]
        alpha_bar_t = sched.alpha_bars[t]
        beta_t = sched.betas[t]
        sigma_t = torch.sqrt(beta_t)

        coeff = beta_t / torch.sqrt(1.0 - alpha_bar_t + 1e-8)
        mean = (x_t - coeff * eps_theta) / torch.sqrt(alpha_t + 1e-8)
        if t > 0:
            x_t = mean + sigma_t * torch.randn_like(x_t)
        else:
            x_t = mean

        # Discrete denoising step: feed argmax states into next iteration.
        atom_logits = out.atom_type_logits
        bond_logits = out.bond_type_logits
        bond_edge_index = out.bond_edge_index
        lig.x[:, 0] = atom_logits.argmax(dim=-1).float() + 1.0
        if edge.edge_attr.size(1) > 0:
            edge.edge_attr[:, 0] = bond_logits.argmax(dim=-1).float()

    return x_t, atom_logits, bond_logits, bond_edge_index


def _set_mol_coords(mol: Chem.Mol, coords: np.ndarray) -> Chem.Mol:
    mol = Chem.Mol(mol)
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(
            i,
            Chem.rdGeometry.Point3D(
                float(coords[i, 0]), float(coords[i, 1]), float(coords[i, 2])
            ),
        )
    return mol


def _compute_rmsd(ref_mol: Chem.Mol, pred_mol: Chem.Mol) -> float:
    # Use heavy-atom best RMSD.
    ref = Chem.RemoveHs(ref_mol)
    pred = Chem.RemoveHs(pred_mol)
    try:
        return float(AllChem.GetBestRMS(ref, pred))
    except Exception:
        return float("nan")


def run_posebusters(
    poserbusters_bin: str,
    sdf_path: str,
    protein_pdb: str,
    output_json: str,
) -> dict:
    """
    Wrapper with robust fallbacks for PoseBusters CLI variants.
    """
    cmds = [
        [posebusters_bin, sdf_path, "--protein", protein_pdb, "--outfmt", "json"],
        [posebusters_bin, "--mol", sdf_path, "--protein", protein_pdb, "--format", "json"],
        [posebusters_bin, sdf_path, "--protein", protein_pdb],
    ]

    for cmd in cmds:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                continue
            text = proc.stdout.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except Exception:
                # Some CLIs print non-JSON; store raw output.
                payload = {"raw_output": text}
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            return payload
        except FileNotFoundError:
            break

    payload = {
        "error": "PoseBusters execution failed or CLI not found",
        "hint": f"Check --posebusters-bin ({posebusters_bin}) and installed PoseBusters version.",
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


@dataclass
class SdfMeta:
    sdf_grid: np.ndarray   # (X, Y, Z) float32, world-space SDF values
    origin: np.ndarray     # (3,) float32, world-space grid origin
    spacing: float         # voxel size in Angstroms


def load_sdf_meta(surface_dir: str, plinder_id: str) -> Optional[SdfMeta]:
    """Load the precomputed SDF grid from the surface_awareness.npz for one system."""
    path = Path(surface_dir) / f"{plinder_id}.surface_awareness.npz"
    if not path.exists():
        return None
    d = np.load(str(path), allow_pickle=True)
    return SdfMeta(
        sdf_grid=d["sdf_grid"].astype(np.float32),
        origin=d["origin"].astype(np.float32),
        spacing=float(d["spacing"]),
    )


def _trilinear_sample(grid: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """
    Trilinear interpolation of a 3-D scalar grid at fractional index positions.

    grid : (X, Y, Z) float32
    idx  : (N, 3) float  — fractional grid indices (x, y, z)
    returns (N,) float32 SDF values
    """
    X, Y, Z = grid.shape
    x0 = np.floor(idx[:, 0]).astype(np.int32)
    y0 = np.floor(idx[:, 1]).astype(np.int32)
    z0 = np.floor(idx[:, 2]).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    # Clamp indices to valid range
    x0c = np.clip(x0, 0, X - 1); x1c = np.clip(x1, 0, X - 1)
    y0c = np.clip(y0, 0, Y - 1); y1c = np.clip(y1, 0, Y - 1)
    z0c = np.clip(z0, 0, Z - 1); z1c = np.clip(z1, 0, Z - 1)

    # Interpolation weights — clamped to [0,1] for atoms outside the grid
    xd = np.clip(idx[:, 0] - x0, 0.0, 1.0)
    yd = np.clip(idx[:, 1] - y0, 0.0, 1.0)
    zd = np.clip(idx[:, 2] - z0, 0.0, 1.0)

    return (
        grid[x0c, y0c, z0c] * (1 - xd) * (1 - yd) * (1 - zd)
        + grid[x1c, y0c, z0c] * xd       * (1 - yd) * (1 - zd)
        + grid[x0c, y1c, z0c] * (1 - xd) * yd       * (1 - zd)
        + grid[x0c, y0c, z1c] * (1 - xd) * (1 - yd) * zd
        + grid[x1c, y1c, z0c] * xd       * yd       * (1 - zd)
        + grid[x1c, y0c, z1c] * xd       * (1 - yd) * zd
        + grid[x0c, y1c, z1c] * (1 - xd) * yd       * zd
        + grid[x1c, y1c, z1c] * xd       * yd       * zd
    ).astype(np.float32)


def compute_surface_overlap(
    ligand_pos_world: np.ndarray,
    sdf_meta: SdfMeta,
    violation_thresh: float = 0.1,
) -> dict:
    """
    Measure how much the predicted ligand penetrates the protein surface.

    ligand_pos_world : (N_atoms, 3) float — heavy-atom positions in world space (Å)
    sdf_meta         : precomputed SDF grid in world space
    violation_thresh : penetration depth (Å) below which an atom is "clean"

    Returns a dict with per-system Surface-Ligand Overlap metrics.

    Convention: SDF < 0  →  atom is inside the protein (violation).
    """
    # Transform world coordinates → fractional grid indices
    # grid_idx = (world_pos - origin) / spacing
    idx = (ligand_pos_world - sdf_meta.origin[None, :]) / sdf_meta.spacing  # (N, 3)
    sdf_vals = _trilinear_sample(sdf_meta.sdf_grid, idx)                     # (N,)

    n_atoms = len(sdf_vals)
    # Penetration depth per atom: positive when inside protein, 0 otherwise
    penetration = np.maximum(0.0, -sdf_vals)

    n_violations    = int((sdf_vals < 0).sum())
    # Clean: violation depth < violation_thresh (includes atoms outside protein)
    n_clean         = int((sdf_vals > -violation_thresh).sum())

    return {
        "slo_n_atoms":              n_atoms,
        "slo_n_violations":         n_violations,
        "slo_violation_fraction":   round(float(n_violations / n_atoms), 4),
        "slo_mean_overlap_depth_A": round(float(penetration.mean()), 4),
        "slo_max_overlap_depth_A":  round(float(penetration.max()), 4),
        "slo_clean_surface_rate":   round(float(n_clean / n_atoms), 4),
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_sdf_dir = out_dir / "pred_sdf"
    pred_sdf_dir.mkdir(parents=True, exist_ok=True)
    poser_dir = out_dir / "posebusters"
    poser_dir.mkdir(parents=True, exist_ok=True)

    dataset = D3DockHeteroDataset(
        input_csv=args.input_csv,
        structures_dir=args.structures_dir,
        crop_dir=args.crop_dir,
        transform=None,
    )
    indices = list(range(len(dataset)))
    if args.split_list:
        keep_ids = _extract_plinder_ids_from_split_list(args.split_list)
        indices = _subset_dataset(dataset, keep_ids)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    model = _load_model(args, device)

    all_metrics = []
    rmsd_values = []
    success_count = 0
    slo_records: list[dict] = []   # surface-ligand overlap per sample

    # Process samples one-by-one for robust molecule serialization.
    for idx in indices:
        rec = dataset.records[idx]
        data = dataset[idx].to(device)

        pred_pos, atom_logits, bond_logits, bond_edge_index = reverse_diffusion_sample(
            model=model,
            data=data,
            T=args.T,
            schedule_name=args.schedule,
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            device=device,
        )

        ref_supplier = Chem.SDMolSupplier(rec.ligand_sdf, sanitize=False, removeHs=False)
        if len(ref_supplier) == 0 or ref_supplier[0] is None:
            continue
        ref_mol = ref_supplier[0]
        pred_mol = _set_mol_coords(
            ref_mol, pred_pos.detach().cpu().numpy().astype(np.float64)
        )
        pred_mol = _apply_discrete_predictions(
            pred_mol, atom_logits, bond_logits, bond_edge_index
        )

        rmsd = _compute_rmsd(ref_mol, pred_mol)
        success = bool(np.isfinite(rmsd) and rmsd < args.rmsd_threshold)
        if np.isfinite(rmsd):
            rmsd_values.append(rmsd)
        if success:
            success_count += 1

        sdf_out = pred_sdf_dir / f"{rec.plinder_id}.pred.sdf"
        w = Chem.SDWriter(str(sdf_out))
        w.write(pred_mol)
        w.close()

        protein_pdb = rec.holo_pdb
        poser_json = poser_dir / f"{rec.plinder_id}.posebusters.json"
        pb = run_posebusters(
            poserbusters_bin=args.posebusters_bin,
            sdf_path=str(sdf_out),
            protein_pdb=protein_pdb,
            output_json=str(poser_json),
        )

        # ── Surface-Ligand Overlap ────────────────────────────────────────────
        slo_metrics: dict = {}
        if args.surface_dir:
            sdf_meta = load_sdf_meta(args.surface_dir, rec.plinder_id)
            if sdf_meta is not None:
                # pred_pos is in COM-normalised space; add the ligand COM (from
                # crop.npz) to recover world-space coordinates before sampling.
                crop_path = (
                    Path(args.crop_dir)
                    / rec.plinder_id
                    / f"{rec.plinder_id}.crop.npz"
                )
                try:
                    crop = np.load(str(crop_path), allow_pickle=True)
                    com = crop["ligand_center_of_mass"].astype(np.float32)  # (3,)
                    pred_world = pred_pos.detach().cpu().numpy().astype(np.float32) + com
                    slo_metrics = compute_surface_overlap(
                        ligand_pos_world=pred_world,
                        sdf_meta=sdf_meta,
                        violation_thresh=args.overlap_violation_thresh,
                    )
                    slo_records.append(slo_metrics)
                except Exception as e:
                    slo_metrics = {"slo_error": str(e)}
            else:
                slo_metrics = {"slo_error": "surface_awareness.npz not found"}

        metric_row = {
            "plinder_id": rec.plinder_id,
            "rmsd": float(rmsd) if np.isfinite(rmsd) else None,
            "success_rmsd_lt_2A": int(success),
            "pred_sdf": str(sdf_out),
            "posebusters_json": str(poser_json),
            "posebusters_status": "ok" if "error" not in pb else "error",
            **slo_metrics,
        }
        all_metrics.append(metric_row)
        slo_str = (
            f" | slo_viol={slo_metrics.get('slo_violation_fraction', 'n/a')}"
            f" clean={slo_metrics.get('slo_clean_surface_rate', 'n/a')}"
            f" max_depth={slo_metrics.get('slo_max_overlap_depth_A', 'n/a')}Å"
            if slo_metrics and "slo_error" not in slo_metrics else ""
        )
        print(f"[{rec.plinder_id}] rmsd={metric_row['rmsd']} success={success}{slo_str}")

    total = len(all_metrics)
    success_rate = float(success_count / total) if total > 0 else 0.0
    mean_rmsd = float(np.mean(rmsd_values)) if len(rmsd_values) > 0 else float("nan")
    median_rmsd = (
        float(np.median(rmsd_values)) if len(rmsd_values) > 0 else float("nan")
    )

    metrics_csv = out_dir / "metrics.csv"
    import pandas as pd

    pd.DataFrame(all_metrics).to_csv(metrics_csv, index=False)
    # ── Aggregate Surface-Ligand Overlap stats ────────────────────────────────
    slo_summary: dict = {}
    if slo_records:
        def _mean(key: str) -> float:
            vals = [r[key] for r in slo_records if key in r]
            return round(float(np.mean(vals)), 4) if vals else float("nan")

        slo_summary = {
            "slo_samples_evaluated":        len(slo_records),
            "slo_mean_violation_fraction":  _mean("slo_violation_fraction"),
            "slo_mean_overlap_depth_A":     _mean("slo_mean_overlap_depth_A"),
            "slo_mean_max_overlap_depth_A": _mean("slo_max_overlap_depth_A"),
            "slo_mean_clean_surface_rate":  _mean("slo_clean_surface_rate"),
            # Fraction of poses with zero violations (fully clash-free)
            "slo_clash_free_rate": round(
                float(
                    sum(1 for r in slo_records if r.get("slo_n_violations", 1) == 0)
                    / len(slo_records)
                ),
                4,
            ),
        }

    summary = {
        "total_samples": total,
        "mean_rmsd": mean_rmsd,
        "median_rmsd": median_rmsd,
        "success_rate_rmsd_lt_2A": success_rate,
        "rmsd_threshold": args.rmsd_threshold,
        **slo_summary,
        "metrics_csv": str(metrics_csv),
        "pred_sdf_dir": str(pred_sdf_dir),
        "posebusters_dir": str(poser_dir),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
