#!/usr/bin/env python3
"""
Generate D3-Dock surface-awareness constraints from cleaned structures.

Outputs a compressed .npz containing:
- surface_points: (N, 3) SES point coordinates (Angstrom)
- surface_normals: (N, 3) SES normals
- sdf_grid: (nx, ny, nz) signed distance values (Angstrom)
- origin: (3,) grid origin
- spacing: float grid spacing (Angstrom)
- grid_shape: (3,) voxel grid shape
- center: (3,) ligand center used for voxelization

SES generation uses MSMS (external binary).
SDF follows:
    Phi(x) = min_i ( ||x - a_i|| - r_i )
where a_i are atom centers and r_i are VdW radii.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from Bio.PDB import PDBParser
from rdkit import Chem


# Bondi-like radii in Angstroms (with practical fallbacks for proteins).
VDW_RADII = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "P": 1.80,
    "S": 1.80,
    "CL": 1.75,
    "BR": 1.85,
    "I": 1.98,
    "ZN": 1.39,
    "MG": 1.73,
    "CA": 1.94,
    "FE": 1.56,
    "CU": 1.40,
    "MN": 1.61,
}
DEFAULT_VDW_RADIUS = 1.70


@dataclass
class ProteinAtoms:
    coords: np.ndarray  # (N, 3)
    radii: np.ndarray  # (N,)
    elements: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SES point cloud + ligand-centered SDF grid for D3-Dock."
    )
    parser.add_argument("--protein-pdb", required=True, help="Path to cleaned protein PDB.")
    parser.add_argument(
        "--ligand-sdf",
        default=None,
        help="Ligand SDF used to compute grid center (preferred).",
    )
    parser.add_argument(
        "--center",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Manual center if ligand SDF is not provided.",
    )
    parser.add_argument("--output-npz", required=True, help="Output .npz filepath.")
    parser.add_argument(
        "--grid-resolution",
        type=float,
        default=0.5,
        help="Voxel spacing in Angstrom (default: 0.5).",
    )
    parser.add_argument(
        "--pocket-padding",
        type=float,
        default=8.0,
        help="Extra margin (Angstrom) around ligand bounding box.",
    )
    parser.add_argument(
        "--probe-radius",
        type=float,
        default=1.5,
        help="MSMS probe radius in Angstrom.",
    )
    parser.add_argument(
        "--surface-density",
        type=float,
        default=3.0,
        help="MSMS surface density value.",
    )
    parser.add_argument(
        "--msms-bin",
        default="msms",
        help="Path to MSMS executable (default: msms).",
    )
    return parser.parse_args()


def _norm_element(element: str | None, atom_name: str) -> str:
    if element:
        return element.strip().upper()
    name = atom_name.strip().upper()
    if not name:
        return "C"
    if len(name) >= 2 and name[:2] in {"CL", "BR", "ZN", "MG", "CA", "FE", "CU", "MN"}:
        return name[:2]
    return name[0]


def load_protein_atoms(protein_pdb: str) -> ProteinAtoms:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", protein_pdb)
    coords: list[np.ndarray] = []
    radii: list[float] = []
    elements: list[str] = []

    for atom in structure.get_atoms():
        # Keep hydrogens in case protein is already protonated.
        element = _norm_element(getattr(atom, "element", None), atom.get_name())
        coords.append(atom.get_coord().astype(np.float32))
        radii.append(float(VDW_RADII.get(element, DEFAULT_VDW_RADIUS)))
        elements.append(element)

    if not coords:
        raise ValueError(f"No atoms parsed from protein PDB: {protein_pdb}")

    return ProteinAtoms(
        coords=np.asarray(coords, dtype=np.float32),
        radii=np.asarray(radii, dtype=np.float32),
        elements=elements,
    )


def load_ligand_coords(ligand_sdf: str) -> np.ndarray:
    supplier = Chem.SDMolSupplier(ligand_sdf, sanitize=False, removeHs=False)
    if len(supplier) == 0 or supplier[0] is None:
        raise ValueError(f"Could not parse ligand SDF: {ligand_sdf}")
    mol = supplier[0]
    conf = mol.GetConformer()
    pts = []
    for idx in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(idx)
        pts.append([p.x, p.y, p.z])
    return np.asarray(pts, dtype=np.float32)


def compute_center_and_extents(
    ligand_coords: np.ndarray | None, manual_center: list[float] | None, pocket_padding: float
) -> tuple[np.ndarray, np.ndarray]:
    if ligand_coords is not None:
        lig_min = ligand_coords.min(axis=0)
        lig_max = ligand_coords.max(axis=0)
        center = (lig_min + lig_max) * 0.5
        extent = (lig_max - lig_min) + 2.0 * pocket_padding
    elif manual_center is not None:
        center = np.asarray(manual_center, dtype=np.float32)
        extent = np.asarray([2.0 * pocket_padding] * 3, dtype=np.float32)
    else:
        raise ValueError("Provide either --ligand-sdf or --center X Y Z.")

    # Enforce minimum cube edge for numerical stability.
    extent = np.maximum(extent, np.asarray([12.0, 12.0, 12.0], dtype=np.float32))
    return center.astype(np.float32), extent.astype(np.float32)


def build_grid(center: np.ndarray, extent: np.ndarray, spacing: float) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    half = extent * 0.5
    origin = center - half
    nx = int(np.ceil(extent[0] / spacing)) + 1
    ny = int(np.ceil(extent[1] / spacing)) + 1
    nz = int(np.ceil(extent[2] / spacing)) + 1

    xs = origin[0] + np.arange(nx, dtype=np.float32) * spacing
    ys = origin[1] + np.arange(ny, dtype=np.float32) * spacing
    zs = origin[2] + np.arange(nz, dtype=np.float32) * spacing
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    points = np.stack((xx, yy, zz), axis=-1).reshape(-1, 3).astype(np.float32)
    return origin.astype(np.float32), points, (nx, ny, nz)


def compute_sdf(
    grid_points: np.ndarray,
    atom_coords: np.ndarray,
    atom_radii: np.ndarray,
    grid_shape: tuple[int, int, int],
    chunk_size: int = 4096,
) -> np.ndarray:
    """
    Exact SDF:
        Phi(x) = min_i ( ||x - a_i|| - r_i )
    computed in chunks for memory safety.
    """
    n_points = grid_points.shape[0]
    out = np.empty(n_points, dtype=np.float32)
    coords = atom_coords.astype(np.float32)
    radii = atom_radii.astype(np.float32)

    for start in range(0, n_points, chunk_size):
        end = min(start + chunk_size, n_points)
        pts = grid_points[start:end]  # (B, 3)
        deltas = pts[:, None, :] - coords[None, :, :]  # (B, N, 3)
        dist = np.sqrt(np.sum(deltas * deltas, axis=-1), dtype=np.float32)  # (B, N)
        phi = np.min(dist - radii[None, :], axis=1)
        out[start:end] = phi.astype(np.float32)

    return out.reshape(grid_shape)


def _write_msms_xyzr(path: str, coords: np.ndarray, radii: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for (x, y, z), r in zip(coords, radii):
            f.write(f"{x:.5f} {y:.5f} {z:.5f} {r:.4f}\n")


def _parse_msms_vertices(vert_path: str) -> tuple[np.ndarray, np.ndarray]:
    points: list[list[float]] = []
    normals: list[list[float]] = []
    with open(vert_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            cols = s.split()
            # MSMS vert rows typically include x y z nx ny nz ...
            if len(cols) < 6:
                continue
            try:
                x, y, z = float(cols[0]), float(cols[1]), float(cols[2])
                nx, ny, nz = float(cols[3]), float(cols[4]), float(cols[5])
            except ValueError:
                continue
            points.append([x, y, z])
            normals.append([nx, ny, nz])

    if not points:
        raise ValueError(f"No SES points parsed from MSMS vert file: {vert_path}")
    return np.asarray(points, dtype=np.float32), np.asarray(normals, dtype=np.float32)


def run_msms(
    msms_bin: str,
    atom_coords: np.ndarray,
    atom_radii: np.ndarray,
    probe_radius: float,
    surface_density: float,
) -> tuple[np.ndarray, np.ndarray]:
    with tempfile.TemporaryDirectory(prefix="d3dock_msms_") as tmpdir:
        xyzr = os.path.join(tmpdir, "protein.xyzr")
        out_prefix = os.path.join(tmpdir, "ses")
        _write_msms_xyzr(xyzr, atom_coords, atom_radii)

        cmd = [
            msms_bin,
            "-if",
            xyzr,
            "-of",
            out_prefix,
            "-probe_radius",
            str(probe_radius),
            "-density",
            str(surface_density),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"MSMS failed (exit={proc.returncode}). "
                f"stderr={proc.stderr.strip()} stdout={proc.stdout.strip()}"
            )

        vert_path = out_prefix + ".vert"
        if not os.path.exists(vert_path):
            raise FileNotFoundError(f"MSMS output missing vert file: {vert_path}")

        return _parse_msms_vertices(vert_path)


def _subset_atoms_near_pocket(
    coords: np.ndarray, radii: np.ndarray, center: np.ndarray, extent: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    half_diag = float(np.linalg.norm(extent * 0.5))
    cutoff = half_diag + float(np.max(radii)) + 2.0
    d = np.linalg.norm(coords - center[None, :], axis=1)
    mask = d <= cutoff
    if np.sum(mask) == 0:
        return coords, radii
    return coords[mask], radii[mask]


def save_npz(
    output_npz: str,
    surface_points: np.ndarray,
    surface_normals: np.ndarray,
    sdf_grid: np.ndarray,
    origin: np.ndarray,
    spacing: float,
    grid_shape: tuple[int, int, int],
    center: np.ndarray,
) -> None:
    Path(output_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        surface_points=surface_points.astype(np.float32),
        surface_normals=surface_normals.astype(np.float32),
        sdf_grid=sdf_grid.astype(np.float32),
        origin=origin.astype(np.float32),
        spacing=np.float32(spacing),
        grid_shape=np.asarray(grid_shape, dtype=np.int32),
        center=center.astype(np.float32),
    )


def main() -> None:
    args = parse_args()

    protein = load_protein_atoms(args.protein_pdb)
    ligand_coords = load_ligand_coords(args.ligand_sdf) if args.ligand_sdf else None
    center, extent = compute_center_and_extents(
        ligand_coords=ligand_coords,
        manual_center=args.center,
        pocket_padding=args.pocket_padding,
    )

    pocket_coords, pocket_radii = _subset_atoms_near_pocket(
        coords=protein.coords,
        radii=protein.radii,
        center=center,
        extent=extent,
    )

    surface_points, surface_normals = run_msms(
        msms_bin=args.msms_bin,
        atom_coords=pocket_coords,
        atom_radii=pocket_radii,
        probe_radius=args.probe_radius,
        surface_density=args.surface_density,
    )

    origin, grid_points, grid_shape = build_grid(
        center=center,
        extent=extent,
        spacing=args.grid_resolution,
    )
    sdf_grid = compute_sdf(
        grid_points=grid_points,
        atom_coords=pocket_coords,
        atom_radii=pocket_radii,
        grid_shape=grid_shape,
    )

    save_npz(
        output_npz=args.output_npz,
        surface_points=surface_points,
        surface_normals=surface_normals,
        sdf_grid=sdf_grid,
        origin=origin,
        spacing=args.grid_resolution,
        grid_shape=grid_shape,
        center=center,
    )
    print(
        f"Wrote surface-awareness data: {args.output_npz} "
        f"(surface_points={surface_points.shape[0]}, grid_shape={grid_shape})"
    )


if __name__ == "__main__":
    main()
