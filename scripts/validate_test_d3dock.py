#!/usr/bin/env python3
"""
Validation/testing script for D3-Dock.

Implements:
- Reverse diffusion inference loop (Schrodinger-Bridge style approximation).
- RMSD metrics vs ground-truth ligand poses.
- Geometry success rate (RMSD < 2.0 A).
- PoseBusters wrapper for physical validity checks.
- SDF export for PyMOL visualization.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import tempfile
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
            stem = Path(s).stem
            pid = stem.split(".")[0]
            ids.add(pid)
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

        metric_row = {
            "plinder_id": rec.plinder_id,
            "rmsd": float(rmsd) if np.isfinite(rmsd) else None,
            "success_rmsd_lt_2A": int(success),
            "pred_sdf": str(sdf_out),
            "posebusters_json": str(poser_json),
            "posebusters_status": "ok" if "error" not in pb else "error",
        }
        all_metrics.append(metric_row)
        print(f"[{rec.plinder_id}] rmsd={metric_row['rmsd']} success={success}")

    total = len(all_metrics)
    success_rate = float(success_count / total) if total > 0 else 0.0
    mean_rmsd = float(np.mean(rmsd_values)) if len(rmsd_values) > 0 else float("nan")
    median_rmsd = (
        float(np.median(rmsd_values)) if len(rmsd_values) > 0 else float("nan")
    )

    metrics_csv = out_dir / "metrics.csv"
    import pandas as pd

    pd.DataFrame(all_metrics).to_csv(metrics_csv, index=False)
    summary = {
        "total_samples": total,
        "mean_rmsd": mean_rmsd,
        "median_rmsd": median_rmsd,
        "success_rate_rmsd_lt_2A": success_rate,
        "rmsd_threshold": args.rmsd_threshold,
        "metrics_csv": str(metrics_csv),
        "pred_sdf_dir": str(pred_sdf_dir),
        "posebusters_dir": str(poser_dir),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
