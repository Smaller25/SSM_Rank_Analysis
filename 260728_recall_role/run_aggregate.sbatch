#!/bin/bash
#SBATCH --job-name=sh_stage4_agg
#SBATCH --partition=main
#SBATCH --time=0:20:00
#SBATCH --output=/home/sohyung/sh_logs/stage4_agg_%j.log
set -e
source /home/sohyung/sh_gdn2_venv/bin/activate
HERE=/home/sohyung/SSM_Rank_Analysis/.sh_exp_worktrees/recall-role-induction/260728_recall_role
python "$HERE/aggregate_seeds.py" --results-root "$HERE/results" --base recall_role_100b --seeds 0,1,2 \
    --out "$HERE/results/aggregate_stage4.json"
echo "STAGE4_AGGREGATE_DONE"
