#!/bin/bash
#SBATCH --job-name=sh_stage2_prune
#SBATCH --partition=main
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --time=6:00:00
#SBATCH --output=/home/sohyung/sh_logs/stage2_prune_%j.log
# Stage 2 (G2) gdn2-1.3B pruning-ASYMMETRY test on the 100B paper-matched checkpoint.
# Mirrors 260725_stage1_rank_stratification/run_stage1.sbatch. Submit via `sbatch` (NEVER direct
# CUDA on the login node: cuInit=100).
set -e
source /home/sohyung/sh_gdn2_venv/bin/activate
export TRITON_CACHE_DIR=/home/sohyung/.triton_cache HF_HUB_DISABLE_XET=1
export GDN2_CKPT_PATH=/home/sohyung/models/gdn2_1.3B_100b.pth   # [PIN-1] 100B paper-matched (10B rejected)
# Per-head SVD is CPU-side (classifier) -> cap BLAS threads to avoid oversubscription.
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
echo "node $(hostname) gpu $CUDA_VISIBLE_DEVICES"
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.device_count())"

# S-NIAH (niah_ruler) runtime deps: wonderwords + nltk(punkt_tab) + the paul_graham essay dataset.
# These were pre-provisioned on the login node (venv + ~/nltk_data + HF cache); the checks below are
# defensive so a fresh compute node still works. HF cache is reused offline once populated.
python -c "import wonderwords, nltk" 2>/dev/null || pip install -q wonderwords nltk
python -c "import nltk; nltk.data.find('tokenizers/punkt_tab')" 2>/dev/null || python -c "import nltk; nltk.download('punkt_tab'); nltk.download('punkt')"

HERE=/home/sohyung/SSM_Rank_Analysis/.sh_exp_worktrees/gdn2-pruning-asymmetry/260726_stage2_pruning_asymmetry
if [ ! -f "$HERE/stage2_pruning.py" ]; then
    echo "ABORT: $HERE/stage2_pruning.py not found" >&2; exit 1
fi

# [PIN-6] >=3 seeds (0,1,2): seed governs random-mask draw + data/needle sampling. Each seed to its
# own out dir; --resume survives interruption (incremental per-condition flush). Aggregate post-hoc.
for SEED in 0 1 2; do
    export PYTHONHASHSEED=$SEED
    OUT=$HERE/results/stage2_100b_seed${SEED}
    mkdir -p "$OUT"
    echo "=== SEED $SEED -> $OUT ==="
    python "$HERE/stage2_pruning.py" \
        --seed $SEED --n-seq 16 --ppl-n-seq 16 --seq-len 2048 \
        --niah-samples 20 --gen-tokens 128 \
        --require-real-data --resume --out "$OUT"
done
echo "STAGE2_DONE"
