#!/bin/bash
#SBATCH --job-name=sh_stage3_agg
#SBATCH --partition=main
#SBATCH --time=0:20:00
#SBATCH --output=/home/sohyung/sh_logs/stage3_agg_%j.log
# Stage 3 aggregate: mean+-std dose x group curves + delta(high)-delta(low)/random + final G3 verdict.
# Runs AFTER all seed jobs succeed (submit_stage3_seeds.sh sets --dependency=afterok).
set -e
source /home/sohyung/sh_gdn2_venv/bin/activate
HERE=/home/sohyung/SSM_Rank_Analysis/.sh_exp_worktrees/gdn2-rank-genuine-capacity/260727_stage3_rank_mechanism
python "$HERE/aggregate_seeds3.py" --results-root "$HERE/results" --base stage3_100b --seeds 0,1,2 \
    --out "$HERE/results/aggregate_stage3.json"
echo "STAGE3_AGGREGATE_DONE"
