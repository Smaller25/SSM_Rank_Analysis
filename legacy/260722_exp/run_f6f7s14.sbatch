#!/bin/bash
#SBATCH --job-name=sh_f6f7s14
#SBATCH --partition=main
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --time=1:00:00
#SBATCH --output=/home/sohyung/sh_logs/f6f7s14_%j.log
set -e
source /home/sohyung/sh_gdn2_venv/bin/activate
export TRITON_CACHE_DIR=/home/sohyung/.triton_cache HF_HUB_DISABLE_XET=1
export GDN2_CKPT_PATH=/home/sohyung/models/gdn2_1.3B_100b.pth
echo "node $(hostname) gpu $CUDA_VISIBLE_DEVICES"
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.device_count())"
python /home/sohyung/SSM_Rank_Analysis/legacy/260722_exp/rank_decomp_100b.py
