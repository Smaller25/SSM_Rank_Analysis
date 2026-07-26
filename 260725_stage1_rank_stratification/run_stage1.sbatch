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

HERE=/home/sohyung/SSM_Rank_Analysis/260725_stage1_rank_stratification
OUT=$HERE/results/stage1_100b
mkdir -p "$OUT"
# --stride 16 -> dense t<d=128 growth grid for [B]; --require-real-data hard-fails on synthetic fallback.
python "$HERE/stage1_repro.py" \
    --n-seq 16 --seq-len 2048 --layer-stride 1 --stride 16 --seed 0 \
    --require-real-data --resume --out "$OUT"
echo "STAGE1_DONE"
