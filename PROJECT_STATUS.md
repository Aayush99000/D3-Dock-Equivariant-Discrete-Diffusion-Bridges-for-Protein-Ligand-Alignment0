# D3-Dock — What We've Built & Where We Are

> Written in plain language. Think of this as your "what is happening and why" reference.

---

## What is D3-Dock trying to do?

You're building a model that predicts **how a drug molecule (ligand) fits inside a protein**.
This is called **molecular docking** and it's a core step in drug discovery — if you know how a drug binds to a protein, you can figure out if it'll work.

Most existing models (like DiffDock or MiDi) have a problem: they predict atoms in positions that are physically impossible — like atoms clipping through the protein wall, or bonds that violate chemistry rules. D3-Dock is designed to fix that by:

1. Enforcing **chemistry rules** (correct atom types and bond orders) using a technique called D3PM
2. Evolving the **3D positions** of the molecule through a process called Neural Schrödinger Bridge
3. Being **aware of the protein surface** (its "skin") so the ligand doesn't clip through it
4. Using a **physics penalty** (SDF grid) to push the ligand away if it overlaps with the protein

---

## The Data — PLINDER

**What it is:** A large public database of protein-ligand pairs from real crystal structures (i.e., experimentally verified poses).

**Why we use it:** It's the best available dataset for this task — tens of thousands of real examples of how drugs bind to proteins.

**Where it lives on the cluster:** `/scratch/katoch.aa/plinder_data/`

---

## What We've Built (Code)

### 1. The Model — `models/d3dock_model.py`
This is the brain of D3-Dock. It has two branches working together:

| Branch | What it does |
|---|---|
| **Discrete branch (D3PM)** | Decides what atom types and bond types the ligand should have. Think of it like spell-checking the chemistry. |
| **Continuous branch (NSB)** | Figures out where in 3D space each atom should sit. |
| **Surface encoder** | Reads the protein's surface shape and chemistry, so the model knows what it's docking into. |
| **Equivariant backbone (e3nn)** | Makes sure predictions don't change just because you rotated or flipped the molecule. Physics doesn't care about orientation, so neither should the model. |

### 2. The Dataset Loader — `scripts/d3dock_pyg_dataset.py`
Reads the preprocessed data from disk and hands it to the model during training. Handles things like loading the ligand, protein atoms, and surface points into the right format.

### 3. The Training Loop — `scripts/train_d3dock.py`
Runs the actual training. Uses multiple GPUs at once (DDP = Distributed Data Parallel) to speed things up. Optimizes 4 losses simultaneously:
- MSE loss → atoms should be in the right 3D position
- Atom-type loss → atom types should be chemically correct
- Bond-type loss → bond orders should be correct
- Physics penalty → ligand should not be inside the protein

### 4. The Validation Script — `scripts/validate_test_d3dock.py`
After training, this runs the model on held-out test data and measures:
- **RMSD** — how far off is our prediction from the real pose (lower = better)
- **PoseBusters** — does the predicted pose pass real chemistry validity checks

---

## What We've Run (Pipeline on the HPC Cluster)

The cluster is at Northeastern Explorer HPC. All data and scripts live under `/scratch/katoch.aa/d3dock/`.

Here's the preprocessing pipeline we ran, step by step:

### Step 1 — Filter PLINDER (`01_filter`) ✅ Done
**Script:** `scripts/filter_plinder_subset.py`

**What it does:** PLINDER has hundreds of thousands of entries. Not all of them are usable — some have missing atoms, weird ligands, or broken structures. This step filters down to a clean, usable subset.

**Output:** `outputs/filtered_systems.csv` — a list of ~52,782 valid protein-ligand systems.

---

### Step 2 — Discrete Preprocessing (`02_preprocess`) ✅ Done
**Script:** `scripts/preprocess_discrete_states.py`

**What it does:** For each system, extracts the ligand's atom types and bond types in a format the model can learn from. Basically: converts raw chemistry into numbers.

**Output:** `outputs/preprocessed/` — one folder per system, 52,782 total.

---

### Step 3 — Surface Generation (`03_surface`) ✅ Done
**Script:** `scripts/generate_surface_awareness.py`

**What it does:** Computes the protein's "skin" — called the Solvent-Excluded Surface (SES). Also computes a 3D distance grid (SDF) that tells the model how far any point in space is from the protein surface. This is what prevents the ligand from clipping into the protein.

**Output:** `outputs/surface/` — one `.surface_awareness.npz` per system, 52,762 total.

---

### Step 4 — Equivariant Cropping (`04_crop`) ✅ Done
**Script:** `scripts/crop_equivariant_cube.py`

**What it does:** Cuts out a 20Å cube around the ligand's center of mass. We only keep:
- Protein atoms within 10Å of the ligand (the "pocket")
- Surface points within 2Å of the pocket
- The SDF grid within that cube

This makes training tractable — instead of feeding the entire protein, we only feed the relevant binding pocket.

**Why 18,389 out of 52,782?** Many systems were dropped because their pocket had too few atoms inside the 20Å cube (tagged `no_pocket_atoms`). These are low-quality or edge-case structures not worth training on.

**Output:** `outputs/crops/` — 18,389 usable systems, each with a `.crop.npz` file.

---

### Step 5 — PL50 Train/Val/Test Split (`05_partition`) ⏳ Running Right Now (Job 6602035)
**Script:** `scripts/partition_pl50_split.py`

**What it does:** Splits the 18,389 systems into training, validation, and test sets — but with a strict rule: **no test protein can be more than 40% similar (in sequence) to any training protein.** This is called the PL50 split.

**Why does this matter?** If a test protein looks almost identical to one the model trained on, the model could "cheat" by memorizing it rather than truly generalizing. The 40% threshold ensures the test set is genuinely unseen.

**Output:** `outputs/splits/train.txt`, `val.txt`, `test.txt` — file lists for each split.

---

### Step 6 — Training (`train_d3dock.py`) ⏸ Not Started Yet
Will launch after the split is ready. Multi-GPU DDP job on the cluster.

### Step 7 — Validation (`validate_test_d3dock.py`) ⏸ Not Started Yet
Will run after training completes. Produces RMSD scores and PoseBusters results.

---

## Current Status

```
Filter → Preprocess → Surface → Crop → [SPLIT RUNNING] → Train → Validate
  ✅          ✅          ✅        ✅           ⏳                ⏸        ⏸
```

You are **one step away from training**. Once job 6602035 finishes and produces the train/val/test lists, the next step is submitting the training SLURM job.

---

## Quick Reference — Where Things Live on the Cluster

| What | Path |
|---|---|
| All scripts | `/scratch/katoch.aa/d3dock/scripts/` |
| Model code | `/scratch/katoch.aa/d3dock/models/` |
| SLURM job scripts | `/scratch/katoch.aa/d3dock/slurm/` |
| Job logs | `/scratch/katoch.aa/d3dock/logs/` |
| Filtered systems | `/scratch/katoch.aa/d3dock/outputs/filtered_systems.csv` |
| Crop files | `/scratch/katoch.aa/d3dock/outputs/crops/` |
| Split lists | `/scratch/katoch.aa/d3dock/outputs/splits/` (after job finishes) |
| PLINDER raw data | `/scratch/katoch.aa/plinder_data/` |
