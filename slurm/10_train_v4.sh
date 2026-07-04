#!/bin/bash
#SBATCH --job-name=d3dock_v4
#SBATCH --partition=courses-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/katoch.aa/d3dock/logs/10_train_v4_%j.out
#SBATCH --error=/scratch/katoch.aa/d3dock/logs/10_train_v4_%j.err

set -e
source /shared/EL9/explorer/miniconda3/25.9.1/miniconda3/etc/profile.d/conda.sh
conda activate d3dock

BASE=/scratch/katoch.aa/d3dock
SPLITS=$BASE/outputs/splits
OUTPUT=$BASE/outputs/checkpoints_v4_capacity
mkdir -p $OUTPUT

echo "[$(date)] Starting D3-Dock v4 training (128-dim, radius graph, grad ckpt)"
echo "Job ID: $SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd $BASE
export PYTHONPATH=$BASE:$PYTHONPATH

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=1 \
    scripts/train_d3dock.py \
    --input-csv  $SPLITS/train_systems.csv \
    --structures-dir $BASE/outputs/preprocessed \
    --crop-dir   $BASE/outputs/crops \
    --output-dir $OUTPUT \
    --epochs     200 \
    --batch-size 2 \
    --num-workers 4 \
    --lr         1e-3 \
    --T          1000 \
    --schedule   cosine \
    --loss-w-cont  1.0 \
    --loss-w-atom  1.0 \
    --loss-w-bond  1.0 \
    --loss-w-phys  0.5 \
    --save-every 5 \
    --seed       42

echo "[$(date)] Training complete."
