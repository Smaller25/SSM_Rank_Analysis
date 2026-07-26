# Stage 1 — gdn2-1.3B rank-stratification reproduction (arXiv:2602.02195)

Judges whether the paper's per-head **State Rank Stratification** (low-vs-high spectral bifurcation)
and **temporal order-preservation** reproduce in pure GDN (gdn2-1.3B, **100B paper-matched
checkpoint** [PIN-1]; 18 layers × 16 heads). Stage 0 audit: `STAGE0_AUDIT.md`. Pre-registered
thresholds: `PREREGISTRATION.md`.

A **single logging forward** per (domain, sequence) produces all three Stage-1 analyses (efficiency,
not deferred judgement):
- **[A]** threshold-rank `Rank_eff` (Eq.6, ε=1e-4) stratification → bimodality (Sarle BC + valley/low/high).
- **[B]** per-head time-consistency measured in the **growth regime** (prefix pairs with `t < d =
  min(dk,dv)`, where rank is still rising) — **two separate axes**: (i) Spearman ρ on the per-head
  **rank** vector, (ii) **norm-cosine** on the per-head **nuclear-norm** vector (magnitude
  order-preservation, Thm 4.4/Eq.13). Saturated prefixes (Spearman NaN) are **counted** as a separate
  regime, not averaged away. (credibility-fix 1 + 3)
- **[C]** r̄ = exp(E_t log a_t) → OLS R² for each of the 3 rank metrics + residual vs the **paper
  bound `min(t, d)`** (Thm 3.1). The old `min(d, e/(1−r̄))` is a local nb3 heuristic **not in
  arXiv:2602.02195**; kept only as an explicitly-labelled auxiliary diagnostic. (credibility-fix 2)
- **[G1c]** entropy eRank vs threshold-rank head-ranking Spearman (metric-artifact guard).
- **[Dom]** cross-domain head-classification (low/high) agreement.

**Data authenticity gate (credibility-fix 4):** if any natural domain (wikitext/github/arxiv) falls
back to the synthetic corpus, `verdict.G1a` is forced **INVALID (`null`)** and routing halts;
`--require-real-data` makes that an immediate hard failure.

## Files
- `rank_metrics.py` — metrics: `threshold_rank` (Eq.6, ε=1e-4) + `entropy_erank`/`stable_rank`
  (imported from `capacity_utils` [PIN-3]) + `nuclear_norm` (`svdvals().sum()`, norm axis, fix-3).
  Every result carries metric name + cap `d`.
- `data_stage1.py` — WikiText-103 / GitHub / arXiv (≥16 seqs × 1024–2048) + App.D attacks ×2.
  HF parquet-native; deterministic synthetic fallback if offline (flagged `source=fallback` +
  `is_fallback=True` — this trips the fix-4 authenticity gate).
- `loader_gdn2.py` — thin adapter over `260722_exp/common.py` (canonical gdn2-1.3B loader [PIN-1],
  **100B checkpoint enforced**: 10B rejected, 95B accepted). a_t capture = kernel-intercept on
  `fused_recurrent_gdn2` (`a_t = exp(g)`) with a deterministic reconstruction fallback; adds a
  per-head r̄ decay probe.
- `stage1_repro.py` — driver + pre-registered verdict (G1a/G1b/G1c). CPU smoke via `--smoke`.
- `run_stage1.sbatch` — greenbeard SLURM launcher (100B, all layers, dense growth grid).
- `requirements.lock` — `pip freeze` of the gdn2 venv used for the run.

## Run (greenbeard SLURM, gdn2 venv)
```bash
sbatch run_stage1.sbatch          # 100B checkpoint, all 5 domains, dense growth grid
# or directly on a GPU node:
source /home/sohyung/sh_gdn2_venv/bin/activate
export TRITON_CACHE_DIR=/home/sohyung/.triton_cache HF_HUB_DISABLE_XET=1
export GDN2_CKPT_PATH=/home/sohyung/models/gdn2_1.3B_100b.pth   # 100B paper-matched [PIN-1]
python stage1_repro.py --n-seq 16 --seq-len 2048 --layer-stride 1 --stride 16 --seed 0 \
       --require-real-data --out results/stage1
# output: results/stage1/stage1_report_<yymmdd>_<tag>_<git>.json  (+ stable alias stage1_report.json)
#         partial_<dom>.json flushed per completed domain (resume with --resume)
```
`--stride 16` gives a dense `t < d = 128` growth grid for [B]. Reduce cost if needed: `--layer-stride 2`
(fewer layers), `--domains wikitext,github,arxiv` (skip attacks). Full-layer logging is recommended.
The loader **rejects the 10B checkpoint** and accepts 95B as a paper-matched near-equivalent [PIN-1].

## Smoke test (CPU, seconds, no model / no GPU)
```bash
python rank_metrics.py            # SVD-rank selftest (planted rank-3, full-rank, zero)
python stage1_repro.py --smoke    # synthetic planted-bimodal states -> whole pipeline + verdict JSON
```
The smoke plants a 50/50 low/high per-head rank split with prefix-growing (order-preserving) ranks and
per-head r̄, so [A]/[B]/[C]/[G1c]/cross-domain/verdict all execute and assert. Confirms wiring +
thresholds + JSON — NOT the model.

## Stage gating [PIN-7]
Stage 2/3 (head masking/pruning, planted-MQAR SIR/oracle/MR, operator-composition C_t) are **not
started** — gated behind G1a. `verdict.next` states the routing.
