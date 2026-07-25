# Stage 1 — gdn2-1.3B rank-stratification reproduction (arXiv:2602.02195)

Judges whether the paper's per-head **State Rank Stratification** (low-vs-high spectral bifurcation)
and **temporal order-preservation** reproduce in pure GDN (gdn2-1.3B, checkpoint-10B). Stage 0 audit:
`STAGE0_AUDIT.md`. Pre-registered thresholds: `PREREGISTRATION.md`.

A **single logging forward** per (domain, sequence) produces all three Stage-1 analyses (efficiency,
not deferred judgement):
- **[A]** threshold-rank `Rank_eff` (Eq.6, ε=1e-4) stratification → bimodality (Sarle BC + valley/low/high).
- **[B]** per-head rank-vector time-consistency: Spearman ρ(r_t1,r_t2) + norm consistency + cos sim,
  over a prefix sweep.
- **[C]** r̄ = exp(E_t log a_t) → OLS R² for each of the 3 rank metrics + residual vs theory curve
  `min(d, e/(1−r̄))`.
- **[G1c]** entropy eRank vs threshold-rank head-ranking Spearman (metric-artifact guard).
- **[Dom]** cross-domain head-classification (low/high) agreement.

## Files
- `rank_metrics.py` — 3 metrics: `threshold_rank` (Eq.6, ε=1e-4) + `entropy_erank`/`stable_rank`
  (imported from `capacity_utils` [PIN-3]). Every result carries metric name + cap `d`.
- `data_stage1.py` — WikiText-103 / GitHub / arXiv (≥16 seqs × 1024–2048) + App.D attacks ×2.
  HF parquet-native; deterministic synthetic fallback if offline (flagged `source=fallback`).
- `loader_gdn2.py` — thin adapter over `260722_exp/common.py` (canonical gdn2-1.3B loader [PIN-1]);
  adds a per-head r̄ decay probe.
- `stage1_repro.py` — driver + pre-registered verdict (G1a/G1b/G1c). CPU smoke via `--smoke`.

## Run (VESSL A100, gdn2 env)
```bash
bash /root/smaller/sh_rebuild/sh_setup.sh          # restore env if /root was reset
export TRITON_CACHE_DIR=/root/triton_cache HF_HUB_DISABLE_XET=1
export GDN2_CKPT_PATH=/root/gdn2_1.3B_10B.pth       # checkpoint-10B [PIN-1]
python stage1_repro.py --n-seq 16 --seq-len 2048 --layer-stride 1 --out results/stage1
# output: results/stage1/stage1_report.json  (per-domain [A][B][C][G1c] + cross_domain + verdict)
```
Reduce cost if needed: `--stride 256` (fewer time snapshots), `--layer-stride 2` (fewer layers),
`--domains wikitext,github,arxiv` (skip attacks). Full-layer logging is recommended for the judgement.

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
