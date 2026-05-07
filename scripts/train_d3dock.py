#!/usr/bin/env python3
"""
DDP-capable training loop for D3-Dock (Neural Schrödinger Bridge style).

Implements:
- Time scheduling (linear/cosine) for T diffusion steps.
- Discrete D3PM-style noising + cross-entropy loss.
- Continuous coordinate score/noise MSE loss.
- Physical overlap penalty via precomputed SDF grid.
- AdamW optimizer + OneCycleLR scheduler.
"""

from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader

from models.d3dock_model import D3DockModel
from scripts.d3dock_pyg_dataset import D3DockHeteroDataset, RandomApoHoloSwap


@dataclass
class DiffusionSchedule:
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bars: torch.Tensor
    sigmas: torch.Tensor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train D3-Dock with DDP support.")
    p.add_argument("--input-csv", required=True)
    p.add_argument("--structures-dir", required=True)
    p.add_argument("--crop-dir", required=True)
    p.add_argument("--output-dir", required=True)

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--max-grad-norm", type=float, default=1.0)

    p.add_argument("--T", type=int, default=1000, help="Diffusion time steps.")
    p.add_argument(
        "--schedule",
        type=str,
        default="cosine",
        choices=["cosine", "linear"],
        help="Noise schedule type.",
    )
    p.add_argument("--beta-start", type=float, default=1e-4)
    p.add_argument("--beta-end", type=float, default=0.02)

    p.add_argument("--num-atom-classes", type=int, default=32)
    p.add_argument("--num-bond-classes", type=int, default=6)

    p.add_argument("--loss-w-cont", type=float, default=1.0)
    p.add_argument("--loss-w-atom", type=float, default=1.0)
    p.add_argument("--loss-w-bond", type=float, default=1.0)
    p.add_argument("--loss-w-phys", type=float, default=0.5)
    p.add_argument(
        "--apo-swap-prob",
        type=float,
        default=0.5,
        help="Probability to swap holo with apo skeleton in training transform.",
    )

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-every", type=int, default=1)
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint .pt file to resume from, or 'auto' to load latest in --output-dir.",
    )
    return p.parse_args()


def set_seed(seed: int, rank: int = 0) -> None:
    s = seed + rank
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def init_distributed() -> tuple[bool, int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


def cleanup_distributed(enabled: bool) -> None:
    if enabled and dist.is_initialized():
        dist.destroy_process_group()


def build_schedule(
    T: int,
    schedule: Literal["cosine", "linear"],
    beta_start: float,
    beta_end: float,
    device: torch.device,
) -> DiffusionSchedule:
    if schedule == "linear":
        betas = torch.linspace(beta_start, beta_end, T, device=device)
    else:
        # Cosine schedule from improved DDPM-style alpha_bar parameterization.
        s = 0.008
        t = torch.linspace(0, T, T + 1, device=device) / T
        alpha_bar = torch.cos(((t + s) / (1 + s)) * math.pi * 0.5) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1 - (alpha_bar[1:] / alpha_bar[:-1]).clamp(min=1e-5, max=0.999)
        betas = betas.clamp(min=1e-6, max=0.999)

    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    sigmas = torch.sqrt(1.0 - alpha_bars)
    return DiffusionSchedule(
        betas=betas, alphas=alphas, alpha_bars=alpha_bars, sigmas=sigmas
    )


def sample_timesteps(batch_size: int, T: int, device: torch.device) -> torch.Tensor:
    return torch.randint(0, T, (batch_size,), device=device, dtype=torch.long)


def d3pm_transition_matrix(num_classes: int, beta_t: torch.Tensor) -> torch.Tensor:
    """
    Simple D3PM-style transition:
        Q_t = (1 - beta_t) * I + beta_t * U
    where U is uniform transition matrix.
    """
    I = torch.eye(num_classes, device=beta_t.device).unsqueeze(0)  # [B,K,K]
    U = torch.full((beta_t.size(0), num_classes, num_classes), 1.0 / num_classes, device=beta_t.device)
    b = beta_t.view(-1, 1, 1)
    return (1.0 - b) * I + b * U


def apply_discrete_noising(
    clean_labels: torch.Tensor,
    node_batch: torch.Tensor,
    schedule: DiffusionSchedule,
    t_per_graph: torch.Tensor,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
    - noisy labels z_t sampled from Q_t(z_t | z_0)
    - clean labels z_0 as CE targets
    """
    B = t_per_graph.size(0)
    beta_t = schedule.betas[t_per_graph]  # [B]
    Q = d3pm_transition_matrix(num_classes, beta_t)  # [B,K,K]

    z0 = clean_labels.long().clamp(min=0, max=num_classes - 1)
    probs = Q[node_batch, z0, :]  # [N,K]
    noisy = torch.multinomial(probs, num_samples=1).squeeze(-1)
    return noisy, z0


def sample_sdf_trilinear(
    sdf_grid: torch.Tensor,
    sdf_origin: torch.Tensor,
    sdf_spacing: torch.Tensor,
    points: torch.Tensor,
    point_batch: torch.Tensor,
) -> torch.Tensor:
    """
    Trilinear sample SDF at points for each graph in batch.
    Assumes sdf_grid shape [B, X, Y, Z].
    """
    B = sdf_grid.size(0)
    out = torch.zeros(points.size(0), device=points.device, dtype=points.dtype)
    for b in range(B):
        idx = (point_batch == b).nonzero(as_tuple=False).view(-1)
        if idx.numel() == 0:
            continue
        p = points[idx]  # [Nb,3], normalized coordinates
        origin = sdf_origin[b]  # [3]
        spacing = sdf_spacing[b].view(1)
        g = sdf_grid[b]  # [X,Y,Z]
        X, Y, Z = g.shape

        xyz = (p - origin) / spacing
        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]

        x0 = torch.floor(x).long().clamp(0, X - 1)
        y0 = torch.floor(y).long().clamp(0, Y - 1)
        z0 = torch.floor(z).long().clamp(0, Z - 1)
        x1 = (x0 + 1).clamp(0, X - 1)
        y1 = (y0 + 1).clamp(0, Y - 1)
        z1 = (z0 + 1).clamp(0, Z - 1)

        xd = (x - x0.float()).clamp(0, 1)
        yd = (y - y0.float()).clamp(0, 1)
        zd = (z - z0.float()).clamp(0, 1)

        c000 = g[x0, y0, z0]
        c001 = g[x0, y0, z1]
        c010 = g[x0, y1, z0]
        c011 = g[x0, y1, z1]
        c100 = g[x1, y0, z0]
        c101 = g[x1, y0, z1]
        c110 = g[x1, y1, z0]
        c111 = g[x1, y1, z1]

        c00 = c000 * (1 - xd) + c100 * xd
        c01 = c001 * (1 - xd) + c101 * xd
        c10 = c010 * (1 - xd) + c110 * xd
        c11 = c011 * (1 - xd) + c111 * xd
        c0 = c00 * (1 - yd) + c10 * yd
        c1 = c01 * (1 - yd) + c11 * yd
        c = c0 * (1 - zd) + c1 * zd
        out[idx] = c
    return out


def maybe_reduce_loss(loss: torch.Tensor, distributed: bool) -> torch.Tensor:
    if not distributed:
        return loss
    with torch.no_grad():
        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        loss /= dist.get_world_size()
    return loss


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.OneCycleLR,
    schedule: DiffusionSchedule,
    args: argparse.Namespace,
    device: torch.device,
    distributed: bool,
) -> dict[str, float]:
    model.train()
    total = {"loss": 0.0, "cont": 0.0, "atom": 0.0, "bond": 0.0, "phys": 0.0}
    n_steps = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)

        lig = batch["ligand"]
        edge_store = batch["ligand", "bond", "ligand"]
        lig_batch = lig.batch
        B = int(lig_batch.max().item()) + 1 if lig_batch.numel() > 0 else 1

        # Sample diffusion timestep per graph.
        t = sample_timesteps(B, args.T, device)
        sigma_t = schedule.sigmas[t]  # [B]
        alpha_bar_t = schedule.alpha_bars[t]  # [B]

        # Continuous forward noising (x_t = sqrt(a_bar) x0 + sqrt(1-a_bar) eps).
        x0 = lig.pos
        eps = torch.randn_like(x0)
        node_sigma = sigma_t[lig_batch].unsqueeze(-1)
        node_alpha = torch.sqrt(alpha_bar_t[lig_batch]).unsqueeze(-1)
        x_t = node_alpha * x0 + node_sigma * eps
        lig.pos = x_t

        # Discrete forward noising for atom classes and bond classes.
        atom_clean = (lig.x[:, 0].long() - 1).clamp(min=0, max=args.num_atom_classes - 1)
        atom_noisy, atom_target = apply_discrete_noising(
            clean_labels=atom_clean,
            node_batch=lig_batch,
            schedule=schedule,
            t_per_graph=t,
            num_classes=args.num_atom_classes,
        )
        # Feed noisy atom-state back as first channel.
        lig.x[:, 0] = atom_noisy.float() + 1.0

        e_src = edge_store.edge_index[0]
        edge_batch = lig_batch[e_src]
        bond_clean = edge_store.edge_attr[:, 0].long().clamp(
            min=0, max=args.num_bond_classes - 1
        )
        bond_noisy, bond_target = apply_discrete_noising(
            clean_labels=bond_clean,
            node_batch=edge_batch,
            schedule=schedule,
            t_per_graph=t,
            num_classes=args.num_bond_classes,
        )
        edge_store.edge_attr[:, 0] = bond_noisy.float()

        out = model(batch)

        # Continuous loss (score/noise prediction).
        loss_cont = F.mse_loss(out.coord_noise, eps)

        # Discrete D3PM losses.
        loss_atom = F.cross_entropy(out.atom_type_logits, atom_target)
        loss_bond = F.cross_entropy(out.bond_type_logits, bond_target)

        # Physical penalty using SDF (negative SDF => overlap/collision).
        x0_hat = x_t - node_sigma * out.coord_noise
        sdf_grid_3d = batch["global"].sdf_grid.view(-1, 41, 41, 41)  # (B, 41, 41, 41)
        sdf_vals = sample_sdf_trilinear(
            sdf_grid=sdf_grid_3d.float(),
            sdf_origin=batch["global"].sdf_origin.float(),
            sdf_spacing=batch["global"].sdf_spacing.float(),
            points=x0_hat,
            point_batch=lig_batch,
        )
        loss_phys = torch.relu(-sdf_vals).mean()

        loss = (
            args.loss_w_cont * loss_cont
            + args.loss_w_atom * loss_atom
            + args.loss_w_bond * loss_bond
            + args.loss_w_phys * loss_phys
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()

        total["loss"] += float(loss.detach().item())
        total["cont"] += float(loss_cont.detach().item())
        total["atom"] += float(loss_atom.detach().item())
        total["bond"] += float(loss_bond.detach().item())
        total["phys"] += float(loss_phys.detach().item())
        n_steps += 1

    for k in total:
        total[k] = total[k] / max(n_steps, 1)

    # Only reduce scalars for logging; gradients already synced by DDP.
    if distributed:
        for k in total:
            tval = torch.tensor(total[k], device=device)
            tval = maybe_reduce_loss(tval, distributed=True)
            total[k] = float(tval.item())
    return total


def build_loader(
    args: argparse.Namespace, distributed: bool, rank: int, world_size: int
) -> tuple[DataLoader, Optional[DistributedSampler], int]:
    dataset = D3DockHeteroDataset(
        input_csv=args.input_csv,
        structures_dir=args.structures_dir,
        crop_dir=args.crop_dir,
        transform=RandomApoHoloSwap(p_swap=args.apo_swap_prob),
    )

    sampler = None
    if distributed:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False
        )
        shuffle = False
    else:
        shuffle = True

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    return loader, sampler, len(dataset)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.OneCycleLR,
    epoch: int,
    out_dir: str,
    is_ddp: bool,
) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model": (model.module.state_dict() if is_ddp else model.state_dict()),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    torch.save(state, os.path.join(out_dir, f"checkpoint_epoch_{epoch:04d}.pt"))


def find_latest_checkpoint(out_dir: str) -> Optional[str]:
    ckpts = sorted(Path(out_dir).glob("checkpoint_epoch_*.pt"))
    return str(ckpts[-1]) if ckpts else None


def main() -> None:
    args = parse_args()
    distributed, rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed, rank)

    if rank == 0:
        print(f"[D3-Dock] Distributed={distributed} world_size={world_size} device={device}")

    loader, sampler, dataset_len = build_loader(args, distributed, rank, world_size)
    if rank == 0:
        print(f"[D3-Dock] Dataset size: {dataset_len}")

    model = D3DockModel(
        ligand_input_dim=3,
        protein_input_dim=4,
        surface_input_dim=5,
        num_atom_classes=args.num_atom_classes,
        num_bond_classes=args.num_bond_classes,
    ).to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    schedule = build_schedule(
        T=args.T,
        schedule=args.schedule,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        device=device,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    steps_per_epoch = max(len(loader), 1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
        div_factor=10.0,
        final_div_factor=100.0,
    )

    start_epoch = 1
    resume_path = args.resume
    if resume_path == "auto":
        resume_path = find_latest_checkpoint(args.output_dir)
    if resume_path is not None:
        if rank == 0:
            print(f"[D3-Dock] Resuming from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        raw_model = model.module if distributed else model
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        if rank == 0:
            print(f"[D3-Dock] Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)

        stats = train_one_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            scheduler=scheduler,
            schedule=schedule,
            args=args,
            device=device,
            distributed=distributed,
        )

        if rank == 0:
            print(
                f"[Epoch {epoch:03d}] "
                f"loss={stats['loss']:.4f} "
                f"cont={stats['cont']:.4f} atom={stats['atom']:.4f} "
                f"bond={stats['bond']:.4f} phys={stats['phys']:.4f} "
                f"lr={scheduler.get_last_lr()[0]:.3e}"
            )
            if epoch % args.save_every == 0:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    out_dir=args.output_dir,
                    is_ddp=distributed,
                )

    cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
