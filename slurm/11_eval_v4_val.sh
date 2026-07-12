#!/bin/bash
#SBATCH --job-name=d3dock_v4_val
#SBATCH --partition=courses-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/katoch.aa/d3dock/logs/11_eval_v4_val_%j.out
#SBATCH --error=/scratch/katoch.aa/d3dock/logs/11_eval_v4_val_%j.err

set -e
source /shared/EL9/explorer/miniconda3/25.9.1/miniconda3/etc/profile.d/conda.sh
conda activate d3dock

BASE=/scratch/katoch.aa/d3dock
CKPT=$BASE/outputs/checkpoints_v4_capacity/checkpoint_epoch_0200.pt
SPLITS=$BASE/outputs/splits
STRUCTS=$BASE/outputs/preprocessed
CROPS=$BASE/outputs/crops
OUT=$BASE/outputs/eval_val_v4_T200

mkdir -p $OUT

echo "[$(date)] Evaluating v4 ep200 on VAL split at T=200"
echo "Job ID: $SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd $BASE
export PYTHONPATH=$BASE:$PYTHONPATH

python scripts/validate_test_d3dock.py \
    --checkpoint      $CKPT \
    --input-csv       $SPLITS/val_systems.csv \
    --structures-dir  $STRUCTS \
    --crop-dir        $CROPS \
    --output-dir      $OUT \
    --T               200

echo "[$(date)] Val eval complete."
echo "=== SUMMARY ==="
cat $OUT/summary.json
