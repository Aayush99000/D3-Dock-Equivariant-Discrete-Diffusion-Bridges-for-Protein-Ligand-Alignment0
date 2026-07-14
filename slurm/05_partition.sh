#!/bin/bash
#SBATCH --job-name=d3dock_partition
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/katoch.aa/d3dock/logs/05_partition_%j.out
#SBATCH --error=/scratch/katoch.aa/d3dock/logs/05_partition_%j.err

set -e
source /shared/EL9/explorer/miniconda3/25.9.1/miniconda3/etc/profile.d/conda.sh
conda activate d3dock

BASE=/scratch/katoch.aa/d3dock
PLINDER=/scratch/katoch.aa/plinder_data/plinder/2024-06/v2

mkdir -p $BASE/outputs/splits $BASE/logs

echo "[$(date)] Partitioning into PL50 train/val/test splits"
echo "Job ID: $SLURM_JOB_ID"

cd $BASE
export PYTHONPATH=$BASE:$PYTHONPATH

# Find the split parquet (try both common locations)
SPLIT_PARQUET=""
for CANDIDATE in \
    "$PLINDER/splits/split.parquet" \
    "$PLINDER/splits/splits.parquet" \
    "$PLINDER/index/splits.parquet"; do
    if [ -f "$CANDIDATE" ]; then
        SPLIT_PARQUET="$CANDIDATE"
        break
    fi
done

if [ -z "$SPLIT_PARQUET" ]; then
    echo "ERROR: Could not find split parquet. Searching..."
    find /scratch/katoch.aa/plinder_data -name "*.parquet" | grep -i split | head -5
    exit 1
fi

echo "Using split parquet: $SPLIT_PARQUET"

python scripts/partition_pl50_split.py \
    --processed-dir    $BASE/outputs/crops \
    --split-parquet    $SPLIT_PARQUET \
    --output-dir       $BASE/outputs/splits \
    --pt-glob          "**/*.crop.npz" \
    --id-regex         "^(.+)\.crop$" \
    --identity-threshold 0.40

# Convert .txt path lists to plinder_id CSVs for use with validate_test_d3dock.py
echo "plinder_id" > $BASE/outputs/splits/train_systems.csv
sed 's|.*/||; s|\.crop\.npz||' $BASE/outputs/splits/train.txt >> $BASE/outputs/splits/train_systems.csv

echo "plinder_id" > $BASE/outputs/splits/val_systems.csv
sed 's|.*/||; s|\.crop\.npz||' $BASE/outputs/splits/val.txt >> $BASE/outputs/splits/val_systems.csv

echo "plinder_id" > $BASE/outputs/splits/test_systems.csv
sed 's|.*/||; s|\.crop\.npz||' $BASE/outputs/splits/test.txt >> $BASE/outputs/splits/test_systems.csv

echo "[$(date)] Done."
echo "Train: $(wc -l < $BASE/outputs/splits/train.txt) systems"
echo "Val:   $(wc -l < $BASE/outputs/splits/val.txt) systems"
echo "Test:  $(wc -l < $BASE/outputs/splits/test.txt) systems"
