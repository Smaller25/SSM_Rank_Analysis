#!/bin/bash
#SBATCH --job-name=sh_rankdecay
#SBATCH --partition=main
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --time=1:00:00
#SBATCH --output=/home/sohyung/sh_logs/rankdecay_%j.log
set -e
source /home/sohyung/sh_gdn2_venv/bin/activate
export TRITON_CACHE_DIR=/home/sohyung/.triton_cache HF_HUB_DISABLE_XET=1 HF_HOME=/data2/sohyung/hf_home
export GDN2_CKPT_PATH=/home/sohyung/models/gdn2_1.3B_100b.pth OMP_NUM_THREADS=4
python /home/sohyung/SSM_Rank_Analysis/260728_rank_decay_dist/rank_decay_dist.py
echo RANKDECAY_DONE
