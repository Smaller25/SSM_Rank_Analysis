#!/bin/bash
#SBATCH --job-name=sh_stage4_recall
#SBATCH --partition=main
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --time=6:00:00
#SBATCH --output=/home/sohyung/sh_logs/stage4_recall_%j.log
# Stage 4 (recall-role) gdn2-1.3B induction probe on the 100B paper-matched checkpoint.
# Mirrors 260726_stage2_pruning_asymmetry/run_stage2.sbatch. Submit via `sbatch` (NEVER direct CUDA
# on the login node: cuInit=100).
#
# ONE SEED PER JOB (PIN-6): each seed governs passage sampling + the random-mask draw. There is NO
# autoregressive decode here (teacher-forced forward only) so this is MUCH faster than Stage 2 NIAH;
# the dominant cost is the classifier's CPU-side per-head SVD. Expected < 60 min/seed; 6h is ample.
# Submit all three with:
#   bash submit_recall_role_seeds.sh          # -> sbatch run_recall_role.sbatch <seed> for 0 1 2
# or one at a time:
#   sbatch run_recall_role.sbatch 0
set -e
source /home/sohyung/sh_gdn2_venv/bin/activate
export TRITON_CACHE_DIR=/home/sohyung/.triton_cache HF_HUB_DISABLE_XET=1
export HF_HOME=/data2/sohyung/hf_home HF_DATASETS_CACHE=/data2/sohyung/hf_home/datasets  # scratch on /data2, not root disk
export GDN2_CKPT_PATH=/home/sohyung/models/gdn2_1.3B_100b.pth   # [PIN-1] 100B paper-matched (10B rejected)
# Per-head SVD is CPU-side (classifier) -> cap BLAS threads to avoid oversubscription.
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
echo "node $(hostname) gpu $CUDA_VISIBLE_DEVICES"
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.device_count())"

HERE=/home/sohyung/SSM_Rank_Analysis/.sh_exp_worktrees/recall-role-induction/260728_recall_role
if [ ! -f "$HERE/stage4_recall_role.py" ]; then
    echo "ABORT: $HERE/stage4_recall_role.py not found" >&2; exit 1
fi

# [PIN-6] ONE seed per job. Aggregate the three per-seed out dirs post-hoc. --resume survives
# interruption (incremental per-condition flush).
SEED=${1:-${SH_SEED:-0}}
export PYTHONHASHSEED=$SEED
OUT=$HERE/results/recall_role_100b_seed${SEED}
mkdir -p "$OUT"
echo "=== SEED $SEED -> $OUT ==="
# [PIN-5] passage_len=1024 -> probe seq=[A][A]=2048 tok (within block_size 4096), FROZEN.
python "$HERE/stage4_recall_role.py" \
    --seed $SEED --n-seq 16 --cls-seq-len 2048 \
    --passage-len 1024 --n-passages 16 \
    --require-real-data --resume --out "$OUT"
echo "STAGE4_SEED${SEED}_DONE"
