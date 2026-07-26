#!/bin/bash
#SBATCH --job-name=sh_stage2_prune
#SBATCH --partition=main
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --time=6:00:00
#SBATCH --output=/home/sohyung/sh_logs/stage2_prune_%j.log
# Stage 2 (G2) gdn2-1.3B pruning-ASYMMETRY test on the 100B paper-matched checkpoint.
# Mirrors 260725_stage1_rank_stratification/run_stage1.sbatch. Submit via `sbatch` (NEVER direct
# CUDA on the login node: cuInit=100).
#
# [fix blocker-3] ONE SEED PER JOB. The S-NIAH greedy decode (autoregressive, full re-forward of the
# ~2k-token prompt each step) dominates wall time; running seeds 0/1/2 SERIALLY inside a single 6h
# allocation risked a mid-run SLURM kill on seed 2 (and wasted the whole GPU allocation on timeout).
# This script now runs exactly ONE seed (from $1 or $SH_SEED, default 0), so each seed gets its own
# fresh 6h budget and the three jobs queue/run independently. Submit all three with:
#   bash submit_stage2_seeds.sh            # -> sbatch run_stage2.sbatch <seed> for seed in 0 1 2
# or one at a time:
#   sbatch run_stage2.sbatch 0
set -e
source /home/sohyung/sh_gdn2_venv/bin/activate
export TRITON_CACHE_DIR=/home/sohyung/.triton_cache HF_HUB_DISABLE_XET=1
export HF_HOME=/data2/sohyung/hf_home HF_DATASETS_CACHE=/data2/sohyung/hf_home/datasets  # [major] scratch on /data2, not root disk
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

# [PIN-6] ONE seed per job. seed governs random-mask draw + data/needle sampling. Aggregate the three
# per-seed out dirs post-hoc. --resume survives interruption (incremental per-condition flush).
SEED=${1:-${SH_SEED:-0}}
export PYTHONHASHSEED=$SEED
OUT=$HERE/results/stage2_100b_seed${SEED}
mkdir -p "$OUT"
echo "=== SEED $SEED -> $OUT ==="
# [fix blocker-3] --gen-tokens 48 (was 128): the needle value is a short 7-digit number the
# substring-recall scorer only needs to SEE, and greedy decode early-stops on a blank-line run; 48
# tokens is ample headroom while cutting the dominant autoregressive-decode cost ~2.7x. gen_tokens is
# NOT a pinned CONSTANT (PIN-1/2/5/6 fix model/rank-metric/domains/seeds+k; not decode length).
python "$HERE/stage2_pruning.py" \
    --seed $SEED --n-seq 16 --ppl-n-seq 16 --seq-len 2048 \
    --niah-samples 20 --gen-tokens 48 \
    --require-real-data --resume --out "$OUT"
echo "STAGE2_SEED${SEED}_DONE"
