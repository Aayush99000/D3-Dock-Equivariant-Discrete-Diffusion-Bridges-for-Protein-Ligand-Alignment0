#!/bin/bash
#SBATCH --job-name=d3dock_demo
#SBATCH --partition=courses-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/katoch.aa/d3dock/logs/13_demo_%j.out
#SBATCH --error=/scratch/katoch.aa/d3dock/logs/13_demo_%j.err

set -e
source /shared/EL9/explorer/miniconda3/25.9.1/miniconda3/etc/profile.d/conda.sh
conda activate d3dock

BASE=/scratch/katoch.aa/d3dock
CKPT=$BASE/outputs/checkpoints_v4_capacity/checkpoint_epoch_0200.pt
SPLITS=$BASE/outputs/splits
STRUCTS=$BASE/outputs/preprocessed
CROPS=$BASE/outputs/crops
OUT=$BASE/outputs/eval_demo

mkdir -p $OUT

# Write a CSV with only the 4 successful val systems
DEMO_CSV=$OUT/demo_systems.csv
echo "plinder_id" > $DEMO_CSV
echo "2ci5__1__1.A__1.C"  >> $DEMO_CSV
echo "7tuu__1__1.A__1.C"  >> $DEMO_CSV
echo "6ioh__1__1.A__1.C"  >> $DEMO_CSV
echo "7oso__1__1.A__1.E"  >> $DEMO_CSV

echo "[$(date)] Running demo trajectory eval on 4 successful val systems"
echo "Job ID: $SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd $BASE
export PYTHONPATH=$BASE:$PYTHONPATH

# Copy staged scripts (home dir is accessible from compute nodes)
cp ~/.d3dock_staging/validate_test_d3dock.py scripts/validate_test_d3dock.py
cp ~/.d3dock_staging/demo_viewer.py          scripts/demo_viewer.py

python scripts/validate_test_d3dock.py \
    --checkpoint      $CKPT \
    --input-csv       $DEMO_CSV \
    --structures-dir  $STRUCTS \
    --crop-dir        $CROPS \
    --output-dir      $OUT \
    --T               200 \
    --save-trajectory \
    --trajectory-stride 5

echo "[$(date)] Trajectory eval complete. Generating HTML demos..."

# RMSD values from val eval
declare -A RMSD
RMSD["2ci5__1__1.A__1.C"]="1.949"
RMSD["7tuu__1__1.A__1.C"]="1.994"
RMSD["6ioh__1__1.A__1.C"]="1.967"
RMSD["7oso__1__1.A__1.E"]="1.661"

mkdir -p $OUT/html

for PID in "2ci5__1__1.A__1.C" "7tuu__1__1.A__1.C" "6ioh__1__1.A__1.C" "7oso__1__1.A__1.E"; do
    echo "  Generating demo for $PID..."
    python scripts/demo_viewer.py \
        --plinder-id  "$PID" \
        --protein-pdb $STRUCTS/$PID/$PID.clean.pdb \
        --true-sdf    $STRUCTS/$PID/$PID.rdkit.sdf \
        --pred-sdf    $OUT/pred_sdf/$PID.pred.sdf \
        --trajectory  $OUT/trajectory/${PID}_trajectory.npz \
        --rmsd        ${RMSD[$PID]} \
        --output      $OUT/html/${PID}_demo.html
done

echo "[$(date)] Done. HTML demos at $OUT/html/"
ls -lh $OUT/html/
