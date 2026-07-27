#!/bin/bash
#SBATCH --job-name=sh_stage3_mech
#SBATCH --partition=main
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --time=6:00:00
#SBATCH --output=/home/sohyung/sh_logs/stage3_mech_%j.log
# Stage 3 (G3) gdn2-1.3B rank=GENUINE-CAPACITY mechanism test on the 100B paper-matched checkpoint.
# Mirrors 260726_stage2_pruning_asymmetry/run_stage2.sbatch. Submit via `sbatch` ONLY (NEVER direct
# CUDA on the login node: cuInit=100 -> MEMORY greenbeard/run-via-slurm).
#
# ONE SEED PER JOB (blocker-3): PPL is teacher-forced (fast), but the SEGMENTED intervention adds
# per-segment kernel launches + CPU SVD per (layer,head) per boundary. int-2 (6 ratios) + int-3 (4
# ladder steps) x 3 groups x segmented forwards over 16 seqs x 3 domains dominate; one seed per 6h
# allocation keeps each job independent and --resume-safe. Submit all three + the aggregate with:
#   bash submit_stage3_seeds.sh          # -> sbatch run_stage3.sbatch <seed> for seed in 0 1 2 (+agg)
# or one at a time:  sbatch run_stage3.sbatch 0
set -e
source /home/sohyung/sh_gdn2_venv/bin/activate
export TRITON_CACHE_DIR=/home/sohyung/.triton_cache HF_HUB_DISABLE_XET=1
export HF_HOME=/data2/sohyung/hf_home HF_DATASETS_CACHE=/data2/sohyung/hf_home/datasets  # scratch on /data2, NOT root disk
export GDN2_CKPT_PATH=/home/sohyung/models/gdn2_1.3B_100b.pth   # [PIN-1] 100B paper-matched (10B REJECTED)
# Per-head SVD (surgery + classifier) is CPU-side -> cap BLAS threads to avoid oversubscription.
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
echo "node $(hostname) gpu $CUDA_VISIBLE_DEVICES"
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.device_count())"

# Optional int-4 linear probe (scikit-learn) + optional S-NIAH runtime deps (wonderwords/nltk).
# Provisioned defensively so a fresh compute node works; HF cache reused offline once populated.
python -c "import sklearn" 2>/dev/null || pip install -q scikit-learn
# NIAH is de-scoped by default (PPL is primary; NIAH origin=0.18 floored). Uncomment if running --niah:
# python -c "import wonderwords, nltk" 2>/dev/null || pip install -q wonderwords nltk
# python -c "import nltk; nltk.data.find('tokenizers/punkt_tab')" 2>/dev/null || python -c "import nltk; nltk.download('punkt_tab'); nltk.download('punkt')"

HERE=/home/sohyung/SSM_Rank_Analysis/.sh_exp_worktrees/gdn2-rank-genuine-capacity/260727_stage3_rank_mechanism
if [ ! -f "$HERE/stage3_mechanism.py" ]; then
    echo "ABORT: $HERE/stage3_mechanism.py not found" >&2; exit 1
fi

# [PIN-6] ONE seed per job. seed governs random-group draw + data sampling + spectrum-noise RNG.
# --resume survives interruption (incremental per-intervention flush). segment_len FROZEN = 256 (PREREG).
SEED=${1:-${SH_SEED:-0}}
export PYTHONHASHSEED=$SEED
OUT=$HERE/results/stage3_100b_seed${SEED}
mkdir -p "$OUT"
echo "=== SEED $SEED -> $OUT ==="
python "$HERE/stage3_mechanism.py" \
    --seed $SEED --n-seq 16 --ppl-n-seq 16 --seq-len 2048 --segment-len 256 \
    --require-real-data --resume --out "$OUT"
echo "STAGE3_SEED${SEED}_DONE"
