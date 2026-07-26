#!/bin/bash
#SBATCH --job-name=sh_stage2_agg
#SBATCH --partition=main
#SBATCH --time=0:20:00
#SBATCH --output=/home/sohyung/sh_logs/stage2_agg_%j.log
set -e
source /home/sohyung/sh_gdn2_venv/bin/activate
HERE=/home/sohyung/SSM_Rank_Analysis/.sh_exp_worktrees/gdn2-pruning-asymmetry/260726_stage2_pruning_asymmetry
python "$HERE/aggregate_seeds.py" --results-root "$HERE/results" --base stage2_100b --seeds 0,1,2 \
    --out "$HERE/results/aggregate_stage2.json"
echo "STAGE2_AGGREGATE_DONE"
