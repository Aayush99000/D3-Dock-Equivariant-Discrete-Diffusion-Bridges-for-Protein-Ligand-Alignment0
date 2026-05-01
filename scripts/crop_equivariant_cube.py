#!/usr/bin/env python3
"""
Crop protein-ligand systems into local 20A cubes centered at ligand COM.

For each system:
1) Pocket atoms: protein atoms within 10A of any ligand atom.
2) Surface extraction: keep SES points associated with the pocket region.
3) Crop SDF grid to a 20A cube centered on ligand COM.
4) Normalize coordinates by subtracting ligand COM.
5) Write outputs under: <output_dir>/<plinder_id>/

Expected upstream files (from previous scripts):
- cleaned structures in <structures_dir>/<plinder_id>/<plinder_id>.clean.pdb
- ligand SDF in <structures_dir>/<plinder_id>/<plinder_id>.rdkit.sdf
- surface bundle in <surface_dir>/<plinder_id>.surface_awareness.npz
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from rdkit import Chem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create ligand-centered local equivariant crops for D3-Dock."
    )
    parser.add_argument(
        "--input-csv",
        required=True,
        help="CSV containing plinder_id column (filtered subset list).",
    )
    parser.add_argument(
        "--structures-dir",
        required=True,
        help="Directory with per-system cleaned structures.",
    )
    parser.add_argument(
        "--surface-dir",
        required=True,
        help="Directory containing per-system surface .npz files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for cropped systems (organized by plinder_id).",
    )
    parser.add_argument(
        "--cube-size",
        type=float,
        default=20.0,
        help="Cube edge length in Angstrom (default: 20.0).",
    )
    parser.add_argument(
        "--pocket-cutoff",
        type=float,
        default=10.0,
        help="Pocket atom cutoff in Angstrom from any ligand atom (default: 10.0).",
    )
    parser.add_argument(
        "--surface-assoc-cutoff",
        type=float,
        default=2.0,
        help="SES-to-pocket association cutoff in Angstrom (default: 2.0).",
    )
    return parser.parse_args()


def _load_ligand_coords(ligand_sdf: str) -> np.ndarray:
    supplier = Chem.SDMolSupplier(ligand_sdf, sanitize=False, removeHs=False)
    if len(supplier) == 0 or supplier[0] is None:
        raise ValueError(f"Cannot parse ligand SDF: {ligand_sdf}")
    mol = supplier[0]
    conf = mol.GetConformer()
    coords = np.zeros((mol.GetNumAtoms(), 3), dtype=np.float32)
    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        coords[i] = [p.x, p.y, p.z]
    return coords


def _load_protein_atoms(pdb_path: str) -> tuple[np.ndarray, list]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    atoms = [atom for atom in structure.get_atoms()]
    if not atoms:
        raise ValueError(f"No atoms parsed from PDB: {pdb_path}")
    coords = np.asarray([atom.get_coord() for atom in atoms], dtype=np.float32)
    return coords, atoms


def _min_distance_mask(query: np.ndarray, ref: np.ndarray, cutoff: float, chunk: int = 4096) -> np.ndarray:
    """
    Returns boolean mask for query points whose min distance to ref <= cutoff.
    Memory-safe chunked distance computation.
    """
    out = np.zeros(query.shape[0], dtype=bool)
    cutoff2 = cutoff * cutoff
    ref32 = ref.astype(np.float32)
    for start in range(0, query.shape[0], chunk):
        end = min(start + chunk, query.shape[0])
        q = query[start:end].astype(np.float32)  # (B,3)
        d = q[:, None, :] - ref32[None, :, :]  # (B,N,3)
        d2 = np.sum(d * d, axis=-1)  # (B,N)
        out[start:end] = np.min(d2, axis=1) <= cutoff2
    return out


def _cube_mask(points: np.ndarray, center: np.ndarray, cube_size: float) -> np.ndarray:
    half = cube_size * 0.5
    return np.all(np.abs(points - center[None, :]) <= half, axis=1)


def _crop_sdf_grid(
    sdf_grid: np.ndarray,
    origin: np.ndarray,
    spacing: float,
    center: np.ndarray,
    cube_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    half = cube_size * 0.5
    # Desired world-space crop box:
    low = center - half
    high = center + half

    # Convert to grid index ranges.
    start = np.floor((low - origin) / spacing).astype(int)
    end = np.ceil((high - origin) / spacing).astype(int)

    shape = np.asarray(sdf_grid.shape, dtype=int)
    start = np.clip(start, 0, shape - 1)
    end = np.clip(end, 1, shape)

    # Ensure non-empty slices.
    end = np.maximum(end, start + 1)
    cropped = sdf_grid[start[0] : end[0], start[1] : end[1], start[2] : end[2]]
    cropped_origin = origin + start.astype(np.float32) * spacing
    return cropped.astype(np.float32), cropped_origin.astype(np.float32)


def _write_pocket_pdb(original_pdb: str, selected_serials: set[int], out_pdb: str) -> None:
    """
    Writes ATOM/HETATM records with serial numbers in selected_serials.
    Keeps file formatting from the original PDB lines.
    """
    with open(original_pdb, "r", encoding="utf-8") as fin, open(out_pdb, "w", encoding="utf-8") as fout:
        for line in fin:
            rec = line[:6].strip()
            if rec not in {"ATOM", "HETATM"}:
                continue
            serial_txt = line[6:11].strip()
            if not serial_txt:
                continue
            try:
                serial = int(serial_txt)
            except ValueError:
                continue
            if serial in selected_serials:
                fout.write(line)
        fout.write("END\n")


def process_system(
    plinder_id: str,
    structures_dir: str,
    surface_dir: str,
    output_dir: str,
    cube_size: float,
    pocket_cutoff: float,
    surface_assoc_cutoff: float,
) -> tuple[str, str]:
    sys_dir = os.path.join(structures_dir, plinder_id)
    ligand_sdf = os.path.join(sys_dir, f"{plinder_id}.rdkit.sdf")
    protein_pdb = os.path.join(sys_dir, f"{plinder_id}.clean.pdb")
    surface_npz = os.path.join(surface_dir, f"{plinder_id}.surface_awareness.npz")

    if not os.path.exists(ligand_sdf):
        return plinder_id, f"missing_ligand: {ligand_sdf}"
    if not os.path.exists(protein_pdb):
        return plinder_id, f"missing_protein: {protein_pdb}"
    if not os.path.exists(surface_npz):
        return plinder_id, f"missing_surface: {surface_npz}"

    lig_coords = _load_ligand_coords(ligand_sdf)
    lig_com = lig_coords.mean(axis=0).astype(np.float32)

    prot_coords, prot_atoms = _load_protein_atoms(protein_pdb)
    pocket_mask = _min_distance_mask(prot_coords, lig_coords, cutoff=pocket_cutoff)
    if pocket_mask.sum() == 0:
        return plinder_id, "no_pocket_atoms"

    pocket_coords = prot_coords[pocket_mask]
    pocket_atoms = [atom for atom, keep in zip(prot_atoms, pocket_mask) if keep]
    serials = {int(atom.get_serial_number()) for atom in pocket_atoms if atom.get_serial_number() is not None}

    surf = np.load(surface_npz)
    surface_points = surf["surface_points"].astype(np.float32)
    surface_normals = surf["surface_normals"].astype(np.float32)
    sdf_grid = surf["sdf_grid"].astype(np.float32)
    sdf_origin = surf["origin"].astype(np.float32)
    spacing = float(surf["spacing"])

    # Surface points associated with pocket atoms and inside cube.
    surf_assoc = _min_distance_mask(surface_points, pocket_coords, cutoff=surface_assoc_cutoff)
    surf_cube = _cube_mask(surface_points, lig_com, cube_size=cube_size)
    surf_keep = surf_assoc & surf_cube
    kept_surface_points = surface_points[surf_keep]
    kept_surface_normals = surface_normals[surf_keep]

    pocket_cube = _cube_mask(pocket_coords, lig_com, cube_size=cube_size)
    pocket_coords_cube = pocket_coords[pocket_cube]
    if pocket_coords_cube.shape[0] == 0:
        return plinder_id, "no_pocket_atoms_in_cube"

    sdf_crop, sdf_crop_origin = _crop_sdf_grid(
        sdf_grid=sdf_grid,
        origin=sdf_origin,
        spacing=spacing,
        center=lig_com,
        cube_size=cube_size,
    )

    # Translation invariance: subtract ligand COM.
    lig_norm = lig_coords - lig_com[None, :]
    pocket_norm = pocket_coords_cube - lig_com[None, :]
    surface_norm_coords = kept_surface_points - lig_com[None, :]
    sdf_crop_origin_norm = sdf_crop_origin - lig_com

    out_dir = os.path.join(output_dir, plinder_id)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # PDB of selected pocket atoms (original coordinates retained for interoperability).
    out_pdb = os.path.join(out_dir, f"{plinder_id}.pocket.raw.pdb")
    _write_pocket_pdb(protein_pdb, serials, out_pdb)

    # Normalized crop bundle for model input.
    out_npz = os.path.join(out_dir, f"{plinder_id}.crop.npz")
    np.savez_compressed(
        out_npz,
        plinder_id=np.asarray([plinder_id]),
        ligand_center_of_mass=lig_com.astype(np.float32),
        ligand_coords_normalized=lig_norm.astype(np.float32),
        pocket_atom_coords_normalized=pocket_norm.astype(np.float32),
        surface_points_normalized=surface_norm_coords.astype(np.float32),
        surface_normals=kept_surface_normals.astype(np.float32),
        sdf_grid=sdf_crop.astype(np.float32),
        sdf_origin_normalized=sdf_crop_origin_norm.astype(np.float32),
        sdf_spacing=np.float32(spacing),
        cube_size=np.float32(cube_size),
        pocket_cutoff=np.float32(pocket_cutoff),
        surface_assoc_cutoff=np.float32(surface_assoc_cutoff),
    )

    return plinder_id, "ok"


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.input_csv)
    if "plinder_id" not in df.columns:
        raise ValueError("Input CSV must contain 'plinder_id' column.")
    ids = [str(x) for x in df["plinder_id"].dropna().unique()]

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    manifest_path = os.path.join(args.output_dir, "crop_manifest.csv")

    rows = []
    for pid in ids:
        pid, status = process_system(
            plinder_id=pid,
            structures_dir=args.structures_dir,
            surface_dir=args.surface_dir,
            output_dir=args.output_dir,
            cube_size=args.cube_size,
            pocket_cutoff=args.pocket_cutoff,
            surface_assoc_cutoff=args.surface_assoc_cutoff,
        )
        rows.append({"plinder_id": pid, "status": status})
        print(f"[{pid}] {status}")

    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"Completed {len(rows)} systems | ok={ok} failed={len(rows)-ok}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
