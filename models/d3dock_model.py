#!/usr/bin/env python3
"""
D3-Dock dual-branch equivariant architecture.

Core blocks:
- Shared E(3)-equivariant encoder for ligand/protein-atom graphs (e3nn Irreps).
- Surface point-cloud transformer for protein surface points + normals/scalars.
- Cross-attention fusion: surface → ligand, protein-atoms → ligand.
- Sinusoidal timestep embedding injected into scalar features (critical for diffusion).
- Dual heads:
  - Continuous coordinate score/noise head (SE(3)-equivariant).
  - Discrete D3PM heads for atom-type and bond-order logits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from e3nn import o3
from torch_scatter import scatter
from torch_geometric.data import HeteroData
from torch_geometric.nn import radius_graph
from torch.utils.checkpoint import checkpoint as grad_checkpoint


def _get_batch(node_store, n: int, device: torch.device) -> torch.Tensor:
    if hasattr(node_store, "batch") and node_store.batch is not None:
        return node_store.batch
    return torch.zeros(n, dtype=torch.long, device=device)


class SinusoidalTimestepEmbedding(nn.Module):
    """
    Sinusoidal positional encoding for diffusion timesteps, followed by a 2-layer MLP.
    Standard in DDPM / DiffSBDD / DiffDock. Projects t in [0, T-1] → R^dim.
    """

    def __init__(self, dim: int, max_period: int = 10000) -> None:
        super().__init__()
        assert dim % 2 == 0, "dim must be even for sinusoidal embedding"
        self.dim = dim
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32) / (half - 1)
        )
        self.register_buffer("freqs", freqs)  # (half,)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) long or float
        x = t.float().unsqueeze(-1) * self.freqs.unsqueeze(0)  # (B, half)
        emb = torch.cat([x.sin(), x.cos()], dim=-1)            # (B, dim)
        return self.mlp(emb)                                    # (B, dim)


class RadialMLP(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return self.net(r)


class EquivariantMessageLayer(nn.Module):
    """
    Irreps-preserving E(3)-equivariant message passing layer.
    """

    def __init__(
        self,
        irreps_node: o3.Irreps,
        irreps_sh: o3.Irreps,
        radial_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.irreps_node = irreps_node
        self.irreps_sh = irreps_sh

        self.tp = o3.FullyConnectedTensorProduct(
            irreps_in1=irreps_node,
            irreps_in2=irreps_sh,
            irreps_out=irreps_node,
            shared_weights=True,
        )
        self.lin_self = o3.Linear(irreps_node, irreps_node)
        self.lin_msg = o3.Linear(irreps_node, irreps_node)
        self.radial = RadialMLP(radial_hidden_dim, irreps_node.dim)
        self.norm = nn.LayerNorm(irreps_node.dim)

    def forward(
        self, feat: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        src, dst = edge_index
        edge_vec = pos[dst] - pos[src]
        edge_len = edge_vec.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        sh = o3.spherical_harmonics(
            self.irreps_sh, edge_vec, normalize=True, normalization="component"
        )
        msg = self.tp(feat[src], sh)
        msg = msg * self.radial(edge_len)

        agg = scatter(msg, dst, dim=0, dim_size=feat.size(0), reduce="mean")
        out = self.lin_self(feat) + self.lin_msg(agg)
        out = feat + out
        return self.norm(out)


class SharedEquivariantEncoder(nn.Module):
    def __init__(
        self,
        input_scalar_dim: int,
        num_scalar_channels: int = 128,
        num_vector_channels: int = 32,
        lmax: int = 2,
        num_layers: int = 4,
        radial_hidden_dim: int = 128,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.scalar_dim = num_scalar_channels
        self.vector_dim = num_vector_channels * 3
        self.use_checkpoint = use_checkpoint
        self.irreps_hidden = o3.Irreps(
            f"{num_scalar_channels}x0e + {num_vector_channels}x1o"
        )
        self.irreps_in = o3.Irreps(f"{input_scalar_dim}x0e")
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax=lmax)

        self.input_proj = o3.Linear(self.irreps_in, self.irreps_hidden)
        self.layers = nn.ModuleList(
            [
                EquivariantMessageLayer(
                    irreps_node=self.irreps_hidden,
                    irreps_sh=self.irreps_sh,
                    radial_hidden_dim=radial_hidden_dim,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self, x_scalar: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        h = self.input_proj(x_scalar)
        for layer in self.layers:
            if self.use_checkpoint:
                h = grad_checkpoint(layer, h, pos, edge_index, use_reentrant=False)
            else:
                h = layer(h, pos, edge_index)
        return h


class SurfacePointTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        model_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
        ff_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.feat_proj = nn.Linear(input_dim, model_dim)
        self.pos_proj = nn.Sequential(
            nn.Linear(3, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    def forward(
        self, surface_pos: torch.Tensor, surface_x: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        h = self.feat_proj(surface_x) + self.pos_proj(surface_pos)
        out = torch.zeros_like(h)
        for b in batch.unique(sorted=True):
            idx = (batch == b).nonzero(as_tuple=False).view(-1)
            hb = h[idx].unsqueeze(0)  # (1, Ns, C)
            out[idx] = self.encoder(hb).squeeze(0)
        return out


class CrossAttentionFusion(nn.Module):
    """
    Generic cross-attention: ligand atoms (query) attend to a context (key/value).
    Used for both surface→ligand and protein-atoms→ligand fusion.
    """

    def __init__(
        self, ligand_dim: int, context_dim: int, num_heads: int = 8
    ) -> None:
        super().__init__()
        self.query_proj = nn.Linear(ligand_dim, ligand_dim)
        self.kv_proj = nn.Linear(context_dim, ligand_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=ligand_dim, num_heads=num_heads, batch_first=True
        )
        self.out_norm = nn.LayerNorm(ligand_dim)
        self.out_mlp = nn.Sequential(
            nn.Linear(ligand_dim, ligand_dim),
            nn.GELU(),
            nn.Linear(ligand_dim, ligand_dim),
        )

    def forward(
        self,
        ligand_scalar: torch.Tensor,
        ligand_batch: torch.Tensor,
        context: torch.Tensor,
        context_batch: torch.Tensor,
    ) -> torch.Tensor:
        out = ligand_scalar.clone()
        for b in ligand_batch.unique(sorted=True):
            lig_idx = (ligand_batch == b).nonzero(as_tuple=False).view(-1)
            ctx_idx = (context_batch == b).nonzero(as_tuple=False).view(-1)
            if lig_idx.numel() == 0 or ctx_idx.numel() == 0:
                continue
            q = self.query_proj(ligand_scalar[lig_idx]).unsqueeze(0)
            kv = self.kv_proj(context[ctx_idx]).unsqueeze(0)
            fused, _ = self.attn(q, kv, kv, need_weights=False)
            fused = fused.squeeze(0)
            out[lig_idx] = self.out_norm(ligand_scalar[lig_idx] + self.out_mlp(fused))
        return out


@dataclass
class D3DockOutput:
    coord_noise: torch.Tensor
    atom_type_logits: torch.Tensor
    bond_type_logits: torch.Tensor
    bond_edge_index: torch.Tensor


class D3DockModel(nn.Module):
    """
    Dual-branch D3-Dock model with equivariant coordinate head and D3PM logits.

    forward(data, t) — t is required: per-graph timestep tensor (B,) long.
    Without timestep conditioning the noise predictor cannot distinguish
    denoising at t=999 (pure noise) from t=1 (near-clean), making the
    reverse diffusion chain diverge.
    """

    def __init__(
        self,
        ligand_input_dim: int = 3,
        protein_input_dim: int = 4,
        surface_input_dim: int = 5,
        num_atom_classes: int = 32,
        num_bond_classes: int = 6,
        hidden_scalar_channels: int = 128,
        hidden_vector_channels: int = 32,
        lmax: int = 2,
        num_gnn_layers: int = 4,
        protein_radius: float = 6.0,
        protein_max_neighbors: int = 64,
        ligand_radius: float = 4.5,
        ligand_max_neighbors: int = 32,
        use_gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.protein_radius = protein_radius
        self.protein_max_neighbors = protein_max_neighbors
        self.ligand_radius = ligand_radius
        self.ligand_max_neighbors = ligand_max_neighbors
        self.scalar_dim = hidden_scalar_channels

        self.shared_encoder_lig = SharedEquivariantEncoder(
            input_scalar_dim=ligand_input_dim,
            num_scalar_channels=hidden_scalar_channels,
            num_vector_channels=hidden_vector_channels,
            lmax=lmax,
            num_layers=num_gnn_layers,
            use_checkpoint=use_gradient_checkpointing,
        )
        self.shared_encoder_prot = SharedEquivariantEncoder(
            input_scalar_dim=protein_input_dim,
            num_scalar_channels=hidden_scalar_channels,
            num_vector_channels=hidden_vector_channels,
            lmax=lmax,
            num_layers=num_gnn_layers,
            use_checkpoint=use_gradient_checkpointing,
        )

        self.irreps_hidden = self.shared_encoder_lig.irreps_hidden

        # Sinusoidal timestep embedding — projects t → scalar_dim additive bias.
        self.time_embed = SinusoidalTimestepEmbedding(dim=hidden_scalar_channels)

        self.surface_encoder = SurfacePointTransformer(
            input_dim=surface_input_dim, model_dim=128, num_heads=8, num_layers=3
        )
        # Surface cross-attention (surface 128-dim → ligand scalar_dim).
        self.surface_to_ligand = CrossAttentionFusion(
            ligand_dim=self.scalar_dim, context_dim=128, num_heads=8
        )
        # Protein-atom cross-attention (protein scalar_dim → ligand scalar_dim).
        self.protein_to_ligand = CrossAttentionFusion(
            ligand_dim=self.scalar_dim, context_dim=self.scalar_dim, num_heads=8
        )

        # Continuous SE(3)-equivariant coordinate head (vector output).
        self.coord_head = o3.Linear(self.irreps_hidden, o3.Irreps("1x1o"))

        # Discrete D3PM logits for atom types.
        self.atom_head = nn.Sequential(
            nn.Linear(self.scalar_dim, self.scalar_dim),
            nn.GELU(),
            nn.Linear(self.scalar_dim, num_atom_classes),
        )

        # Discrete D3PM logits for bond orders/types.
        self.bond_head = nn.Sequential(
            nn.Linear(self.scalar_dim * 2 + 1, self.scalar_dim),
            nn.GELU(),
            nn.Linear(self.scalar_dim, num_bond_classes),
        )

    def _protein_edges(self, pos: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        return radius_graph(
            x=pos,
            r=self.protein_radius,
            batch=batch,
            loop=False,
            max_num_neighbors=self.protein_max_neighbors,
        )

    def _ligand_spatial_edges(
        self, pos: torch.Tensor, batch: torch.Tensor, bond_edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Union of chemical bond edges and spatial radius edges (r=ligand_radius)."""
        radius_ei = radius_graph(
            x=pos,
            r=self.ligand_radius,
            batch=batch,
            loop=False,
            max_num_neighbors=self.ligand_max_neighbors,
        )
        combined = torch.cat([bond_edge_index, radius_ei], dim=1)
        N = pos.size(0)
        flat = combined[0] * N + combined[1]
        flat = torch.unique(flat)
        return torch.stack([flat // N, flat % N], dim=0)

    def forward(self, data: HeteroData, t: torch.Tensor) -> D3DockOutput:
        """
        data : HeteroData batch (all coordinates COM-normalized)
        t    : (B,) long — diffusion timestep per graph in the batch
        """
        lig = data["ligand"]
        prot = data["protein_atoms"]
        surf = data["protein_surface"]

        lig_x = lig.x.float()
        lig_pos = lig.pos.float()
        lig_batch = _get_batch(lig, lig_x.size(0), lig_x.device)

        prot_x = prot.x.float()
        prot_pos = prot.pos.float()
        prot_batch = _get_batch(prot, prot_x.size(0), prot_x.device)

        surf_x = surf.x.float()
        surf_pos = surf.pos.float()
        surf_batch = _get_batch(surf, surf_x.size(0), surf_x.device)

        bond_edge_index = data["ligand", "bond", "ligand"].edge_index.long()
        protein_edge_index = self._protein_edges(prot_pos, prot_batch)
        ligand_edge_index = self._ligand_spatial_edges(lig_pos, lig_batch, bond_edge_index)

        # Equivariant encoders.
        lig_feat = self.shared_encoder_lig(lig_x, lig_pos, ligand_edge_index)
        prot_feat = self.shared_encoder_prot(prot_x, prot_pos, protein_edge_index)

        lig_scalar = lig_feat[:, : self.scalar_dim]
        prot_scalar = prot_feat[:, : self.scalar_dim]

        # Timestep embedding: inject into per-atom scalar features via lig_batch index.
        time_emb = self.time_embed(t)               # (B, scalar_dim)
        lig_scalar = lig_scalar + time_emb[lig_batch]

        # Surface cross-attention.
        surface_latent = self.surface_encoder(surf_pos, surf_x, surf_batch)
        fused_scalar = self.surface_to_ligand(
            ligand_scalar=lig_scalar,
            ligand_batch=lig_batch,
            context=surface_latent,
            context_batch=surf_batch,
        )

        # Protein-atom cross-attention: gives ligand atoms direct structural context.
        fused_scalar = self.protein_to_ligand(
            ligand_scalar=fused_scalar,
            ligand_batch=lig_batch,
            context=prot_scalar,
            context_batch=prot_batch,
        )

        # Inject fused scalar channels back into full irreps feature for coord head.
        fused_feat = lig_feat.clone()
        fused_feat[:, : self.scalar_dim] = fused_scalar

        coord_noise = self.coord_head(fused_feat)       # (N_lig, 3)
        atom_type_logits = self.atom_head(fused_scalar)

        src, dst = bond_edge_index
        bond_dist = (lig_pos[src] - lig_pos[dst]).norm(dim=-1, keepdim=True)
        bond_feat = torch.cat([fused_scalar[src], fused_scalar[dst], bond_dist], dim=-1)
        bond_type_logits = self.bond_head(bond_feat)

        return D3DockOutput(
            coord_noise=coord_noise,
            atom_type_logits=atom_type_logits,
            bond_type_logits=bond_type_logits,
            bond_edge_index=bond_edge_index,
        )
