#!/bin/bash
#SBATCH --job-name=d3dock_surface
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --array=0-49%25
#SBATCH --output=/scratch/katoch.aa/d3dock/logs/03_surface_%A_%a.out
#SBATCH --error=/scratch/katoch.aa/d3dock/logs/03_surface_%A_%a.err

# Array of 50 tasks, max 25 running at once.
# Each task handles 1/50 of systems with 4 parallel MSMS workers.
# Total parallelism: 25 tasks × 4 workers = 100 concurrent MSMS processes.

set -e
source /shared/EL9/explorer/miniconda3/25.9.1/miniconda3/etc/profile.d/conda.sh
conda activate d3dock

BASE=/scratch/katoch.aa/d3dock
STRUCTS=$BASE/outputs/preprocessed
SURF=$BASE/outputs/surface

mkdir -p $SURF $BASE/logs

echo "[$(date)] Surface array task $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_MAX"
echo "Job: $SLURM_ARRAY_JOB_ID  Task: $SLURM_ARRAY_TASK_ID"

cd $BASE
export PYTHONPATH=$BASE:$PYTHONPATH
export PLINDER_MOUNT=/scratch/katoch.aa/plinder_data

# Build list of all systems that have a clean PDB
ALL_SYSTEMS=( $(find $STRUCTS -name "*.clean.pdb" -printf "%h\n" | sort) )
TOTAL=${#ALL_SYSTEMS[@]}

# Divide into 50 equal chunks
TASK_ID=$SLURM_ARRAY_TASK_ID
N_TASKS=50
CHUNK=$(( (TOTAL + N_TASKS - 1) / N_TASKS ))
START=$(( TASK_ID * CHUNK ))
END=$(( START + CHUNK ))
[ $END -gt $TOTAL ] && END=$TOTAL

echo "Processing systems $START to $END of $TOTAL total"

MAX_JOBS=4
RUNNING=0

for (( i=START; i<END; i++ )); do
    SYS_DIR=${ALL_SYSTEMS[$i]}
    PID=$(basename $SYS_DIR)
    PDB=$SYS_DIR/${PID}.clean.pdb
    SDF=$SYS_DIR/${PID}.rdkit.sdf
    OUT=$SURF/${PID}.surface_awareness.npz

    [ -f "$OUT" ] && continue
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

wait

DONE=$(ls $SURF/*.npz 2>/dev/null | wc -l)
echo "[$(date)] Task $TASK_ID done. Total surface files so far: $DONE"
