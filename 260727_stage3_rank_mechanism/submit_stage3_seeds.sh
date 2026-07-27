#!/bin/bash
# Submit Stage 3 seeds 0/1/2 as independent sbatch jobs (each own 6h budget, --resume-safe), then an
# aggregate job that runs AFTER all seed jobs succeed (--dependency=afterok) to compute the final G3.
set -e
HERE=/home/sohyung/SSM_Rank_Analysis/.sh_exp_worktrees/gdn2-rank-genuine-capacity/260727_stage3_rank_mechanism
SEEDS=("$@"); [ ${#SEEDS[@]} -eq 0 ] && SEEDS=(0 1 2)
IDS=""
for SEED in "${SEEDS[@]}"; do
    JID=$(sbatch --parsable "$HERE/run_stage3.sbatch" "$SEED")
    echo "seed $SEED -> job $JID"
    IDS="${IDS}:${JID}"
done
AGG=$(sbatch --parsable --dependency=afterok${IDS} "$HERE/run_aggregate3.sbatch")
echo "aggregate -> job $AGG (afterok${IDS})"
