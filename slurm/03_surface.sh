#!/bin/bash
#SBATCH --job-name=d3dock_surface
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/katoch.aa/d3dock/logs/03_surface_%j.out
#SBATCH --error=/scratch/katoch.aa/d3dock/logs/03_surface_%j.err

set -e
source /shared/EL9/explorer/miniconda3/25.9.1/miniconda3/etc/profile.d/conda.sh
conda activate d3dock

BASE=/scratch/katoch.aa/d3dock
STRUCTS=$BASE/outputs/preprocessed
SURF=$BASE/outputs/surface

mkdir -p $SURF $BASE/logs

echo "[$(date)] Generating surface awareness files (MSMS + SDF grid)"
echo "Job ID: $SLURM_JOB_ID"

cd $BASE
export PYTHONPATH=$BASE:$PYTHONPATH

# Run up to 15 systems in parallel; wait when pool is full
MAX_JOBS=15
RUNNING=0
FAILED=0

for SYS_DIR in $STRUCTS/*/; do
    PID=$(basename $SYS_DIR)
    PDB=$SYS_DIR/${PID}.clean.pdb
    SDF=$SYS_DIR/${PID}.rdkit.sdf
    OUT=$SURF/${PID}.surface_awareness.npz

    # Skip if already done
    [ -f "$OUT" ] && continue
    # Skip if cleaned PDB missing (system failed preprocessing)
    [ -f "$PDB" ] || continue

    python scripts/generate_surface_awareness.py \
        --protein-pdb "$PDB" \
        --ligand-sdf  "$SDF" \
        --output-npz  "$OUT" \
        --grid-resolution 0.5 \
        --pocket-padding  8.0 \
        --probe-radius    1.5 \
        --surface-density 3.0 \
        2>/dev/null &

    RUNNING=$((RUNNING + 1))
    if [ $RUNNING -ge $MAX_JOBS ]; then
        wait -n 2>/dev/null || wait
        RUNNING=$((RUNNING - 1))
    fi
done

# Wait for remaining background jobs
wait

echo "[$(date)] Done."
echo "Surface files generated: $(ls $SURF/*.npz 2>/dev/null | wc -l)"
