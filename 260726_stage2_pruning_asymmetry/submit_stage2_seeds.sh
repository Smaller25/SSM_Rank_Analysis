#!/bin/bash
# [fix blocker-3] Submit Stage 2 seeds 0/1/2 as THREE INDEPENDENT sbatch jobs, each with its own
# fresh 6h budget, instead of a single serial-3-seed allocation that risked a mid-run timeout kill on
# seed 2. The jobs queue/run independently (concurrently if the partition has capacity); each --resume
# survives interruption via incremental per-condition JSON flush. Aggregate the three per-seed out
# dirs post-hoc.
#
# Usage:
#   bash submit_stage2_seeds.sh          # submit seeds 0 1 2
#   bash submit_stage2_seeds.sh 0 1      # submit a subset
set -e
HERE=/home/sohyung/SSM_Rank_Analysis/.sh_exp_worktrees/gdn2-pruning-asymmetry/260726_stage2_pruning_asymmetry
SEEDS=("$@")
if [ ${#SEEDS[@]} -eq 0 ]; then
    SEEDS=(0 1 2)
fi
for SEED in "${SEEDS[@]}"; do
    echo "sbatch run_stage2.sbatch $SEED"
    sbatch "$HERE/run_stage2.sbatch" "$SEED"
done
