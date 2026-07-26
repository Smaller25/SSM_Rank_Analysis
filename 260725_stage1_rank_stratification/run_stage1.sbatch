#!/bin/bash
#SBATCH --job-name=sh_stage1_rank
#SBATCH --partition=main
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --time=4:00:00
#SBATCH --output=/home/sohyung/sh_logs/stage1_rank_%j.log
# Stage 1 (G1) gdn2-1.3B rank-stratification reproduction on the 100B paper-matched checkpoint.
# Adapted from 260722_exp/run_f6f7s14.sbatch. Submit via `sbatch` (never direct CUDA on login node).
set -e
source /home/sohyung/sh_gdn2_venv/bin/activate
export TRITON_CACHE_DIR=/home/sohyung/.triton_cache HF_HUB_DISABLE_XET=1
export GDN2_CKPT_PATH=/home/sohyung/models/gdn2_1.3B_100b.pth   # [PIN-1] 100B paper-matched (10B rejected)
# SVD is CPU-side and called per head many times -> cap BLAS threads to avoid oversubscription.
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
echo "node $(hostname) gpu $CUDA_VISIBLE_DEVICES"
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.device_count())"

# [BLOCKER-FIX] Point HERE at the WORKTREE that holds the 4 credibility fixes
# (growth-regime rho, paper min(t,d) bound, nuclear-norm cosine, fallback->INVALID gate).
# The main working tree (.../260725_stage1_rank_stratification) is the pre-fix code (commit 86849e3)
# and MUST NOT be run for this experiment.
HERE=/home/sohyung/SSM_Rank_Analysis/.sh_exp_worktrees/gdn2-rank-stratification-repro/260725_stage1_rank_stratification
# Guard: hard-fail unless the fixed code is actually present (2 paper_theory_bound refs expected).
if [ ! -f "$HERE/stage1_repro.py" ]; then
    echo "ABORT: $HERE/stage1_repro.py not found" >&2; exit 1
fi
if [ "$(grep -c paper_theory_bound "$HERE/stage1_repro.py")" -lt 1 ]; then
    echo "ABORT: $HERE/stage1_repro.py lacks the credibility fixes (paper_theory_bound marker absent)" >&2; exit 1
fi
# --stride 16 -> dense t<d=128 growth grid + separated late anchors (256/512/1024/2048) for [B].
# --require-real-data hard-fails on synthetic fallback. [PIN-6] >=3 seeds (data sampling is now
# seed-dependent -> real multi-seed variance); each seed to its own out dir, aggregated post-hoc.
for SEED in 0 1 2; do
    export PYTHONHASHSEED=$SEED
    OUT=$HERE/results/stage1_100b_seed${SEED}
    mkdir -p "$OUT"
    echo "=== SEED $SEED -> $OUT ==="
    python "$HERE/stage1_repro.py" \
        --n-seq 16 --seq-len 2048 --layer-stride 1 --stride 16 --seed $SEED \
        --require-real-data --resume --out "$OUT"
done
echo "STAGE1_DONE"
