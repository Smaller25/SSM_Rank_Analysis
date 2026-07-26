#!/bin/bash
# Submit Stage 2 seeds 0/1/2 as independent sbatch jobs (each own 6h budget, --resume-safe), then an
# aggregate job that runs AFTER all seed jobs succeed (--dependency=afterok) to compute the final G2.
set -e
HERE=/home/sohyung/SSM_Rank_Analysis/.sh_exp_worktrees/gdn2-pruning-asymmetry/260726_stage2_pruning_asymmetry
SEEDS=("$@"); [ ${#SEEDS[@]} -eq 0 ] && SEEDS=(0 1 2)
IDS=""
for SEED in "${SEEDS[@]}"; do
    JID=$(sbatch --parsable "$HERE/run_stage2.sbatch" "$SEED")
    echo "seed $SEED -> job $JID"
    IDS="${IDS}:${JID}"
done
AGG=$(sbatch --parsable --dependency=afterok${IDS} "$HERE/run_aggregate.sbatch")
echo "aggregate -> job $AGG (afterok${IDS})"
