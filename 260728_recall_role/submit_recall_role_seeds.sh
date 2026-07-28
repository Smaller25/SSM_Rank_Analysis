#!/bin/bash
# Submit Stage 4 seeds 0/1/2 as independent sbatch jobs (each own 6h budget, --resume-safe), then an
# aggregate job that runs AFTER all seed jobs succeed (--dependency=afterok) to compute the final
# recall-role verdict. Mirrors 260726_stage2_pruning_asymmetry/submit_stage2_seeds.sh.
set -e
HERE=/home/sohyung/SSM_Rank_Analysis/.sh_exp_worktrees/recall-role-induction/260728_recall_role
SEEDS=("$@"); [ ${#SEEDS[@]} -eq 0 ] && SEEDS=(0 1 2)
IDS=""
for SEED in "${SEEDS[@]}"; do
    JID=$(sbatch --parsable "$HERE/run_recall_role.sbatch" "$SEED")
    echo "seed $SEED -> job $JID"
    IDS="${IDS}:${JID}"
done
AGG=$(sbatch --parsable --dependency=afterok${IDS} "$HERE/run_aggregate.sbatch")
echo "aggregate -> job $AGG (afterok${IDS})"
