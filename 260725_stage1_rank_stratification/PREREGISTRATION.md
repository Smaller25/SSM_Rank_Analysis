# Pre-registration — gdn2-1.3B rank-stratification reproduction (Stage 1 / G1)

Registered before the logging run. Post-hoc changes require a dated rationale here [PIN-4].
Constants mirror the plan §6 + consistencyNotes PINs and are echoed into `stage1_report.json["config"]`.

## Model / data constants (CONSTANT across sessions)
- **[PIN-1] model** = gdn2-1.3B, `Config.from_name("gdn2_1.3B")`, `load_state_dict(strict=False)`,
  bf16, `mode="fused_recurrent"`. Checkpoint = **checkpoint-10B** `/root/gdn2_1.3B_10B.pth`.
  No regression to 370m.
- **[PIN-2] threshold-rank** ε = **1e-4**, `Rank_eff = Σ_i I(σ_i > ε·σ_1)` (paper Eq.6). Every rank
  number is reported with metric name + implementation + cap `d = min(dk,dv)`.
- **[PIN-3] entropy eRank / stable rank** = `capacity_utils.effective_rank` / `stable_rank` (imported,
  not re-implemented).
- **[PIN-5] data** = public 3 domains (WikiText-103 / GitHub code / arXiv), ≥16 seqs × 1024–2048 tok
  each, + App.D repeat attacks ×2 (rare-char, common-number). RankViz is unpublished → public
  substitution (rationale recorded).

## Decision thresholds (hypotheses)
| Gate | Metric | Threshold |
|---|---|---|
| **G1a** | stratification bimodality (Sarle BC on normalized threshold-rank) | BC > 0.555 in ≥50% of natural domains |
| **G1a** | temporal order-preservation Spearman ρ(r_t1,r_t2) | ρ > **0.90** in ≥50% of layers (mean over natural domains) |
| (aux) | norm consistency / cos sim | > **0.98** / > **0.97** (paper App.) |
| **G1b** | r̄ → rank regression R² (threshold-rank) | pass ≥ **0.7**, weak ≥ **0.3**, else fail |
| **G1c** | eRank vs threshold-rank head-ranking Spearman | strong ≥ **0.8**, pass ≥ **0.6**, else fail |
| (Stage3) | MQAR miss-rate MR | < **0.2** (pinned; not evaluated in Stage 1) |
| classify | low/high head split threshold θ_R (normalized rank) | **0.5** |

## Outcome routing [PIN-7]
- **G1a = YES** → the paper's stratification + temporal consistency reproduces on pure GDN → Stage 2
  may start; paper access permitted.
- **G1a = NO (unimodal)** → record limitation (hybrid / scale / post-training dependence); do NOT
  train hybrid; Stage 2 stays gated.

## Notes
- Stage-1 natural-language observation is deterministic per forward; variance comes from ≥16 seqs/domain.
- All gdn2 layers logged (`--layer-stride 1`) for the reproduction judgement (paper visualizes all 48 layers).
- G1c exists because entropy eRank ≠ threshold-rank: head classifications may diverge; both are reported
  (metric-artifact guard).
