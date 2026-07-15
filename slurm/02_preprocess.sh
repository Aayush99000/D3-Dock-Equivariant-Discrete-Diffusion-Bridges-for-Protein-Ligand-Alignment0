#!/bin/bash
#SBATCH --job-name=d3dock_preprocess
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/katoch.aa/d3dock/logs/02_preprocess_%j.out
#SBATCH --error=/scratch/katoch.aa/d3dock/logs/02_preprocess_%j.err

set -e
source /shared/EL9/explorer/miniconda3/25.9.1/miniconda3/etc/profile.d/conda.sh
conda activate d3dock

BASE=/scratch/katoch.aa/d3dock

mkdir -p $BASE/outputs/preprocessed $BASE/logs

echo "[$(date)] Preprocessing discrete states (RDKit + OpenBabel)"
echo "Job ID: $SLURM_JOB_ID"

cd $BASE
export PYTHONPATH=$BASE:$PYTHONPATH

# Install openbabel if missing
if ! command -v obabel &>/dev/null; then
    echo "[$(date)] Installing openbabel into d3dock env..."
    conda install -y -c conda-forge openbabel
fi

python scripts/preprocess_discrete_states.py \
    --input-csv      $BASE/outputs/filtered_systems.csv \
    --output-dir     $BASE/outputs/preprocessed \
    --checkpoint-csv $BASE/outputs/preprocessed/preprocess_checkpoint.csv \
    --workers        15

echo "[$(date)] Done."
echo "Systems preprocessed: $(ls $BASE/outputs/preprocessed | grep -v checkpoint | wc -l)"
