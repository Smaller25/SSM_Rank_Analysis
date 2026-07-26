# Pre-registration — gdn2-1.3B rank-stratification reproduction (Stage 1 / G1)

Registered before the logging run. Post-hoc changes require a dated rationale here [PIN-4].
Constants mirror the plan §6 + consistencyNotes PINs and are echoed into `stage1_report.json["config"]`.

> **Threshold provenance caveat ([PIN-4], added 2026-07-26).** Every decision threshold below
> (ρ>0.90, norm-cos>0.98, cos>0.97, R², rank-corr, MR) is the value the paper *observed* on
> **Qwen3-Next (48 layers, large post-trained)**. The model under test here is **gdn2-1.3B, 18 layers
> × 16 heads, 100B-base** — outside the paper's scope. These thresholds are therefore adopted as
> **reproduction TARGET lines, not pass standards**: falling short is a generalization limitation to
> record (routing → "G1a=NO, limitation"), not a code failure.

## Model / data constants (CONSTANT across sessions)
- **[PIN-1] model** = gdn2-1.3B, `Config.from_name("gdn2_1.3B")`, `load_state_dict(strict=False)`,
  bf16, `mode="fused_recurrent"`. Checkpoint = **100B paper-matched** `model-100b.pth`
  (`/home/sohyung/models/gdn2_1.3B_100b.pth`); **95B accepted** as a paper-matched near-equivalent;
  **10B REJECTED** by the loader (different training amount → different state rank). No regression to
  370m. Architecture: **18 layers × 16 heads** (`num_v_heads == num_heads == 16`, no GVA).
- **[PIN-2] threshold-rank** ε = **1e-4**, `Rank_eff = Σ_i I(σ_i > ε·σ_1)` (paper Eq.6). Every rank
  number is reported with metric name + implementation + cap `d = min(dk,dv)`.
- **[PIN-3] entropy eRank / stable rank** = `capacity_utils.effective_rank` / `stable_rank` (imported,
  not re-implemented).
- **[PIN-5] data** = public 3 domains (WikiText-103 / GitHub code / arXiv), ≥16 seqs × 1024–2048 tok
  each, + App.D repeat attacks ×2 (rare-char, common-number). RankViz is unpublished → public
  substitution (rationale recorded).

## Decision thresholds (hypotheses) — all TARGET lines (Qwen3-Next-observed), not pass standards
| Gate | Metric | Threshold |
|---|---|---|
| **G1a** | stratification bimodality (Sarle BC on normalized threshold-rank) | BC > 0.555 in ≥50% of natural domains |
| **G1a** | temporal order-preservation Spearman ρ on the **rank** vector, **growth regime `t<d`** (fix-1) | ρ > **0.90** in ≥50% of layers (mean over natural domains) |
| (aux) | **norm-cosine on the per-head nuclear-norm vector** (fix-3) / cos sim on rank | > **0.98** / > **0.97** (paper App., Thm 4.4/Eq.13) |
| **G1b** | r̄ → rank regression R² (threshold-rank) + paper bound `min(t,d)` residual (fix-2) | pass ≥ **0.7**, weak ≥ **0.3**, else fail |
| **G1c** | eRank vs threshold-rank head-ranking Spearman | strong ≥ **0.8**, pass ≥ **0.6**, else fail |
| (Stage3) | MQAR miss-rate MR | < **0.2** (pinned; not evaluated in Stage 1) |
| classify | low/high head split threshold θ_R (normalized rank) | **0.5** |
| **valid** | data authenticity (fix-4) | any natural domain on synthetic fallback → **G1a = INVALID (`null`)**, routing halts |

## Outcome routing [PIN-7]
- **G1a = INVALID (`null`)** → a natural domain fell back to synthetic data; verdict untrustworthy →
  re-run with real data (`--require-real-data`). Stage 2 stays gated.
- **G1a = YES** → the paper's stratification + temporal consistency reproduces on pure GDN → Stage 2
  may start; paper access permitted.
- **G1a = NO (unimodal)** → record limitation (hybrid / scale / post-training dependence); do NOT
  train hybrid; Stage 2 stays gated.

## Notes
- Stage-1 natural-language observation is deterministic per forward; variance comes from ≥16 seqs/domain.
  Seed is pinned (torch / numpy / PYTHONHASHSEED) via `--seed` before any data is drawn [PIN-6].
- All gdn2 layers logged (`--layer-stride 1`) for the reproduction judgement. The model has **18
  layers × 16 heads**; the paper's "48 layers" refers to Qwen3-Next, not to this pure-GDN model.
- **Time consistency [B]** is measured on GROWTH prefix pairs (`t < d = min(dk,dv)`) where rank is
  still rising; saturated pairs (Spearman NaN) are counted and reported as a separate regime, never
  silently averaged (fix-1). RANK axis = Spearman ρ; NORM axis = norm-cosine on nuclear norms (fix-3).
- **Theory curve [C]** uses the paper bound `min(t,d)` (Thm 3.1); the old `min(d, e/(1−r̄))` is a
  local nb3 heuristic **not in arXiv:2602.02195**, retained only as a labelled auxiliary (fix-2).
- G1c exists because entropy eRank ≠ threshold-rank: head classifications may diverge; both are reported
  (metric-artifact guard).
