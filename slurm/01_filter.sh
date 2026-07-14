#!/bin/bash
#SBATCH --job-name=d3dock_filter
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/katoch.aa/d3dock/logs/01_filter_%j.out
#SBATCH --error=/scratch/katoch.aa/d3dock/logs/01_filter_%j.err

set -e
source /shared/EL9/explorer/miniconda3/25.9.1/miniconda3/etc/profile.d/conda.sh
conda activate d3dock

BASE=/scratch/katoch.aa/d3dock
PLINDER_MOUNT=/scratch/katoch.aa/plinder_data

mkdir -p $BASE/logs $BASE/outputs $PLINDER_MOUNT

echo "[$(date)] Filtering PLINDER systems"
echo "Job ID: $SLURM_JOB_ID"

cd $BASE
export PYTHONPATH=$BASE:$PYTHONPATH
export PLINDER_MOUNT=$PLINDER_MOUNT

# plinder.core will auto-download the annotation index to PLINDER_MOUNT
python scripts/filter_plinder_subset.py \
    --plinder-data-root  $PLINDER_MOUNT/plinder/2024-06/v2/systems \
    --output-csv         $BASE/outputs/filtered_systems.csv \
    --max-resolution     2.5 \
    --min-ligand-mw      100 \
    --max-ligand-mw      800

echo "[$(date)] Done."
wc -l $BASE/outputs/filtered_systems.csv
