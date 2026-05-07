#!/usr/bin/env python3
"""
PyTorch Geometric dataset utilities for D3-Dock.

Each sample returns a HeteroData with:
- data["ligand"].x
- data["ligand", "bond", "ligand"].edge_index
- data["ligand", "bond", "ligand"].edge_attr
- data["protein_atoms"].x
- data["protein_surface"].x
- data["sdf_grid"] (tensor field on the root object)

Includes RandomApoHoloSwap transform to improve docking robustness during training.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from Bio.PDB import PDBParser
from rdkit import Chem
from scipy.spatial import cKDTree
from torch_geometric.data import Dataset, HeteroData


AMINO_ACIDS = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
]
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
AA_UNKNOWN = len(AMINO_ACIDS)

# Coarse residue hydrophobicity (Kyte-Doolittle scale, normalized later).
AA_HYDRO = {
    "ILE": 4.5,
    "VAL": 4.2,
    "LEU": 3.8,
    "PHE": 2.8,
    "CYS": 2.5,
    "MET": 1.9,
    "ALA": 1.8,
    "GLY": -0.4,
    "THR": -0.7,
    "SER": -0.8,
    "TRP": -0.9,
    "TYR": -1.3,
    "PRO": -1.6,
    "HIS": -3.2,
    "GLU": -3.5,
    "GLN": -3.5,
    "ASP": -3.5,
    "ASN": -3.5,
    "LYS": -3.9,
    "ARG": -4.5,
}

# Very coarse integer charge model for electrostatic proxy at pH 7.4.
AA_CHARGE = {
    "ASP": -1.0,
    "GLU": -1.0,
    "LYS": 1.0,
    "ARG": 1.0,
    "HIS": 0.1,
}

# Bond/chirality/hybridization encodings.
CHIRALITY_TO_IDX = {
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED: 0,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW: 1,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW: 2,
    Chem.rdchem.ChiralType.CHI_OTHER: 3,
}
HYBRID_TO_IDX = {
    Chem.rdchem.HybridizationType.UNSPECIFIED: 0,
    Chem.rdchem.HybridizationType.S: 1,
    Chem.rdchem.HybridizationType.SP: 2,
    Chem.rdchem.HybridizationType.SP2: 3,
    Chem.rdchem.HybridizationType.SP3: 4,
    Chem.rdchem.HybridizationType.SP3D: 5,
    Chem.rdchem.HybridizationType.SP3D2: 6,
}
BOND_TO_IDX = {
    Chem.rdchem.BondType.SINGLE: 0,
    Chem.rdchem.BondType.DOUBLE: 1,
    Chem.rdchem.BondType.TRIPLE: 2,
    Chem.rdchem.BondType.AROMATIC: 3,
}


@dataclass
class SystemPaths:
    plinder_id: str
    ligand_sdf: str
    holo_pdb: str
    apo_pdb: Optional[str]
    crop_npz: str


class D3DockHeteroDataset(Dataset):
    """
    Final training dataset yielding HeteroData objects for D3-Dock.
    """

    def __init__(
        self,
        input_csv: str,
        structures_dir: str,
        crop_dir: str,
        transform=None,
        pre_transform=None,
        apo_required: bool = False,
    ) -> None:
        self.input_csv = input_csv
        self.structures_dir = structures_dir
        self.crop_dir = crop_dir
        self.apo_required = apo_required

        self.records = self._build_records()
        super().__init__(root="", transform=transform, pre_transform=pre_transform)

    def _build_records(self) -> list[SystemPaths]:
        df = pd.read_csv(self.input_csv)
        if "plinder_id" not in df.columns:
            raise ValueError("Input CSV must contain 'plinder_id'.")

        records: list[SystemPaths] = []
        for pid in df["plinder_id"].dropna().astype(str).unique():
            sys_dir = os.path.join(self.structures_dir, pid)
            ligand_sdf = os.path.join(sys_dir, f"{pid}.rdkit.sdf")
            holo_pdb = os.path.join(sys_dir, f"{pid}.clean.pdb")
            crop_npz = os.path.join(self.crop_dir, pid, f"{pid}.crop.npz")
            apo_pdb = self._resolve_apo_path(sys_dir, pid)

            if not os.path.exists(ligand_sdf):
                continue
            if not os.path.exists(holo_pdb):
                continue
            if not os.path.exists(crop_npz):
                continue
            if self.apo_required and apo_pdb is None:
                continue

            records.append(
                SystemPaths(
                    plinder_id=pid,
                    ligand_sdf=ligand_sdf,
                    holo_pdb=holo_pdb,
                    apo_pdb=apo_pdb,
                    crop_npz=crop_npz,
                )
            )
        if len(records) == 0:
            raise ValueError("No valid systems found for dataset construction.")
        return records

    @staticmethod
    def _resolve_apo_path(system_dir: str, plinder_id: str) -> Optional[str]:
        candidates = [
            os.path.join(system_dir, f"{plinder_id}.apo.clean.pdb"),
            os.path.join(system_dir, f"{plinder_id}.apo.pdb"),
            os.path.join(system_dir, "apo.pdb"),
            os.path.join(system_dir, "apo.clean.pdb"),
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return None

    def len(self) -> int:
        return len(self.records)

    def get(self, idx: int) -> HeteroData:
        rec = self.records[idx]
        ligand_mol = _load_rdkit_mol(rec.ligand_sdf)
        holo_info = _load_protein_features(rec.holo_pdb)
        apo_info = _load_protein_features(rec.apo_pdb) if rec.apo_pdb else None
        crop = np.load(rec.crop_npz)

        data = HeteroData()
        data["plinder_id"] = rec.plinder_id

        # Ligand node features: atomic number, chirality, hybridization.
        lig_x, lig_pos = _build_ligand_node_features(ligand_mol)
        edge_index, edge_attr = _build_ligand_edges(ligand_mol)
        data["ligand"].x = lig_x
        data["ligand"].pos = lig_pos
        data["ligand", "bond", "ligand"].edge_index = edge_index
        data["ligand", "bond", "ligand"].edge_attr = edge_attr

        # Protein atoms (default to holo).
        data["protein_atoms"].x = holo_info["x"]
        data["protein_atoms"].pos = holo_info["pos"]
        data["protein_atoms_holo"].x = holo_info["x"]
        data["protein_atoms_holo"].pos = holo_info["pos"]
        if apo_info is not None:
            data["protein_atoms_apo"].x = apo_info["x"]
            data["protein_atoms_apo"].pos = apo_info["pos"]
            data["has_apo"] = torch.tensor([1], dtype=torch.long)
        else:
            data["has_apo"] = torch.tensor([0], dtype=torch.long)

        # Surface features: normals + hydrophobicity + electrostatic potential.
        surface_pos = torch.tensor(crop["surface_points_normalized"], dtype=torch.float32)
        surface_normals = torch.tensor(crop["surface_normals"], dtype=torch.float32)
        hydro, electro = _surface_scalar_features(
            surface_pos.numpy(), holo_info["pos"].numpy(), holo_info["aa_idx"].numpy()
        )
        hydro_t = torch.tensor(hydro[:, None], dtype=torch.float32)
        electro_t = torch.tensor(electro[:, None], dtype=torch.float32)
        data["protein_surface"].pos = surface_pos
        data["protein_surface"].x = torch.cat([surface_normals, hydro_t, electro_t], dim=1)

        # SDF voxel grid stored as a flattened "global" node feature so PyG
        # can batch it correctly (concatenates along dim 0 → [B, flat_dim]).
        # Grids vary slightly in size; pad to SDF_GRID_SIZE^3 with a large
        # positive value (= far outside protein = no collision penalty).
        SDF_GRID_SIZE = 41
        sdf_np = crop["sdf_grid"]
        sdf_padded = np.full((SDF_GRID_SIZE,) * 3, fill_value=10.0, dtype=np.float32)
        x, y, z = sdf_np.shape
        sdf_padded[:x, :y, :z] = sdf_np
        data["global"].sdf_grid = torch.tensor(
            sdf_padded.ravel()[None], dtype=torch.float32
        )  # (1, SDF_GRID_SIZE^3)
        data["global"].sdf_spacing = torch.tensor(
            [[float(crop["sdf_spacing"])]], dtype=torch.float32
        )  # (1, 1)
        data["global"].sdf_origin = torch.tensor(
            crop["sdf_origin_normalized"][None], dtype=torch.float32
        )  # (1, 3)

        return data


class RandomApoHoloSwap:
    """
    Randomly swaps protein_atoms skeleton from Holo to Apo (if available).

    This acts on data["protein_atoms"] while preserving
    data["protein_atoms_holo"] and data["protein_atoms_apo"].
    """

    def __init__(self, p_swap: float = 0.5) -> None:
        if p_swap < 0.0 or p_swap > 1.0:
            raise ValueError("p_swap must be in [0, 1].")
        self.p_swap = p_swap

    def __call__(self, data: HeteroData) -> HeteroData:
        has_apo = bool(int(data["has_apo"][0].item())) if "has_apo" in data else False
        if (not has_apo) or ("protein_atoms_apo" not in data.node_types):
            data["used_apo"] = torch.tensor([0], dtype=torch.long)
            return data

        if torch.rand(1).item() < self.p_swap:
            data["protein_atoms"].x = data["protein_atoms_apo"].x
            data["protein_atoms"].pos = data["protein_atoms_apo"].pos
            data["used_apo"] = torch.tensor([1], dtype=torch.long)
        else:
            data["protein_atoms"].x = data["protein_atoms_holo"].x
            data["protein_atoms"].pos = data["protein_atoms_holo"].pos
            data["used_apo"] = torch.tensor([0], dtype=torch.long)
        return data


def _load_rdkit_mol(sdf_path: str) -> Chem.Mol:
    supplier = Chem.SDMolSupplier(sdf_path, sanitize=False, removeHs=False)
    if len(supplier) == 0 or supplier[0] is None:
        raise ValueError(f"Cannot parse ligand SDF: {sdf_path}")
    mol = supplier[0]
    Chem.SanitizeMol(mol)
    return mol


def _build_ligand_node_features(mol: Chem.Mol) -> tuple[torch.Tensor, torch.Tensor]:
    conf = mol.GetConformer()
    x = []
    pos = []
    for i, atom in enumerate(mol.GetAtoms()):
        chirality = CHIRALITY_TO_IDX.get(atom.GetChiralTag(), 0)
        hybr = HYBRID_TO_IDX.get(atom.GetHybridization(), 0)
        x.append([atom.GetAtomicNum(), chirality, hybr])
        p = conf.GetAtomPosition(i)
        pos.append([p.x, p.y, p.z])
    return (
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(pos, dtype=torch.float32),
    )


def _build_ligand_edges(mol: Chem.Mol) -> tuple[torch.Tensor, torch.Tensor]:
    edges = []
    attrs = []
    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()
        bt = BOND_TO_IDX.get(b.GetBondType(), 0)
        conjugated = 1.0 if b.GetIsConjugated() else 0.0
        aromatic = 1.0 if b.GetIsAromatic() else 0.0
        feat = [float(bt), conjugated, aromatic]
        edges.append([i, j])
        edges.append([j, i])
        attrs.append(feat)
        attrs.append(feat)

    if len(edges) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 3), dtype=torch.float32)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(attrs, dtype=torch.float32)
    return edge_index, edge_attr


def _normalize_aa_hydro(v: float) -> float:
    # Range approximately [-4.5, +4.5] -> [-1, 1].
    return float(v) / 4.5


def _load_protein_features(pdb_path: Optional[str]) -> Optional[dict[str, torch.Tensor]]:
    if pdb_path is None:
        return None
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)

    pos = []
    x = []
    aa_idx_list = []
    for atom in structure.get_atoms():
        coord = atom.get_coord().astype(np.float32)
        residue = atom.get_parent()
        resname = residue.get_resname().strip().upper()
        aa_idx = AA_TO_IDX.get(resname, AA_UNKNOWN)
        b_factor = float(atom.get_bfactor()) if atom.get_bfactor() is not None else 0.0
        atomic_num = _atomic_number_from_biopython_atom(atom)
        is_backbone = 1.0 if atom.get_name().strip() in {"N", "CA", "C", "O"} else 0.0

        # Requested: amino acid type + B-factor (plus two helpful structural channels).
        x.append([float(aa_idx), b_factor, float(atomic_num), is_backbone])
        pos.append(coord)
        aa_idx_list.append(aa_idx)

    if len(pos) == 0:
        raise ValueError(f"No atoms found in protein PDB: {pdb_path}")

    return {
        "pos": torch.tensor(np.asarray(pos), dtype=torch.float32),
        "x": torch.tensor(np.asarray(x), dtype=torch.float32),
        "aa_idx": torch.tensor(np.asarray(aa_idx_list), dtype=torch.long),
    }


def _atomic_number_from_biopython_atom(atom) -> int:
    raw = getattr(atom, "element", "") or atom.get_name()
    s = str(raw).strip().upper()
    if len(s) == 0:
        return 6
    # Minimal mapping sufficient for proteins.
    mapping = {
        "H": 1,
        "C": 6,
        "N": 7,
        "O": 8,
        "S": 16,
        "P": 15,
        "FE": 26,
        "ZN": 30,
        "MG": 12,
        "CA": 20,
    }
    if s in mapping:
        return mapping[s]
    if s[0] in mapping:
        return mapping[s[0]]
    return 6


def _aa_from_idx(idx: int) -> str:
    if idx < 0 or idx >= len(AMINO_ACIDS):
        return "UNK"
    return AMINO_ACIDS[idx]


def _surface_scalar_features(
    surface_pos: np.ndarray,
    protein_pos: np.ndarray,
    protein_aa_idx: np.ndarray,
    k_nn: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute simple local surface scalar channels:
    - hydrophobicity: average normalized KD score of nearby residues
    - electrostatic potential: coarse sum(q / r) using residue-level charges
    """
    if surface_pos.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    tree = cKDTree(protein_pos)
    dists, idxs = tree.query(surface_pos, k=min(k_nn, protein_pos.shape[0]))

    if dists.ndim == 1:
        dists = dists[:, None]
        idxs = idxs[:, None]

    hydro = np.zeros((surface_pos.shape[0],), dtype=np.float32)
    electro = np.zeros((surface_pos.shape[0],), dtype=np.float32)

    eps = 1e-3
    for i in range(surface_pos.shape[0]):
        neigh_idx = np.asarray(idxs[i], dtype=int)
        neigh_dist = np.asarray(dists[i], dtype=np.float32)
        aa_names = [_aa_from_idx(int(protein_aa_idx[j])) for j in neigh_idx]
        h_vals = np.asarray(
            [_normalize_aa_hydro(AA_HYDRO.get(a, 0.0)) for a in aa_names], dtype=np.float32
        )
        q_vals = np.asarray([AA_CHARGE.get(a, 0.0) for a in aa_names], dtype=np.float32)

        hydro[i] = float(np.mean(h_vals))
        electro[i] = float(np.sum(q_vals / np.maximum(neigh_dist, eps)))
    return hydro.astype(np.float32), electro.astype(np.float32)


if __name__ == "__main__":
    # Minimal smoke usage example:
    # dataset = D3DockHeteroDataset(
    #     input_csv="/scratch/$USER/d3dock/filtered_plinder_systems.csv",
    #     structures_dir="/scratch/$USER/d3dock/preprocessed",
    #     crop_dir="/scratch/$USER/d3dock/crops",
    #     transform=RandomApoHoloSwap(p_swap=0.5),
    # )
    # print(dataset[0])
    pass
