#!/bin/bash
#SBATCH --job-name=d3dock_crop
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/katoch.aa/d3dock/logs/04_crop_%j.out
#SBATCH --error=/scratch/katoch.aa/d3dock/logs/04_crop_%j.err

set -e
source /shared/EL9/explorer/miniconda3/25.9.1/miniconda3/etc/profile.d/conda.sh
conda activate d3dock

BASE=/scratch/katoch.aa/d3dock

mkdir -p $BASE/outputs/crops $BASE/logs

echo "[$(date)] Generating equivariant pocket crops"
echo "Job ID: $SLURM_JOB_ID"

cd $BASE
export PYTHONPATH=$BASE:$PYTHONPATH

python scripts/crop_equivariant_cube.py \
    --input-csv      $BASE/outputs/filtered_systems.csv \
    --structures-dir $BASE/outputs/preprocessed \
    --surface-dir    $BASE/outputs/surface \
    --output-dir     $BASE/outputs/crops \
    --cube-size      20.0 \
    --pocket-cutoff  10.0 \
    --surface-assoc-cutoff 2.0

echo "[$(date)] Done."
echo "Crops generated: $(find $BASE/outputs/crops -name '*.crop.npz' | wc -l)"
