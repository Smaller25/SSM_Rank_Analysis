# Stage 2 — GDN pruning ASYMMETRY (gdn2-1.3B, 100B paper-matched)

Tests the paper's (arXiv:2602.02195 §5, Table 1) pruning **asymmetry** on pure GDN: does pruning
**low-rank** state heads hurt far more than pruning **high-rank** heads, *beyond a random
equal-count control*? Stage 1 (G1a=YES, 3 seeds) unlocked this per [PIN-7].

## Gate G2
`delta = degradation vs origin`. Asymmetry confirmed if **low-rank pruning hurts more than
high-rank pruning AND low-rank damage exceeds the random control** (and high-rank is no worse than
random), on aggregate across ≥3 seeds. Primary DV = S-NIAH retrieval accuracy; secondary = per-domain
PPL. If NOT reproduced → record hybrid/post-training-specific possibility, re-decide framing.

## Four count-matched conditions (identical k)
`origin` (no mask) · `high` (zero k high-rank heads) · `low` (zero k low-rank heads, predicted worst)
· `random` (zero k random heads; seed governs the draw — the critical control).

## Files
- `head_classifier.py` — RE-DERIVES low/high head sets from code (Stage 1 JSON lost). Runs
  `loader_gdn2.load().states` over the 3 natural domains, computes `rank_metrics.threshold_rank`
  (eps=1e-4) per (layer,head), normalizes by cap, splits at θ_R=0.5, count-matches low/high to k.
  Emits a cross-domain agreement sanity gate (~0.97 expected).
- `head_mask.py` — `HeadMasker`: forward hooks on each `GatedDeltaNet2.o_norm` submodule that zero
  the `(b,t,h,d)` mixer output `o` at the exact gdn2.py:391 site (after `o_norm`, before
  `rearrange`/`o_proj`). Empty mask == origin.
- `niah_retrieval.py` — S-NIAH RETRIEVAL scorer (the real gap): builds the RULER multikey prompt via
  `niah_ruler.make_mk_niah`, greedy-decodes ~128 tokens via `bundle.logits`, scores needle-value
  substring recall, swept over needle depth.
- `ppl_eval.py` — per-domain PPL via `analysis.token_nll_bits` (reused, not re-implemented).
- `stage2_pruning.py` — driver: classify → 4 conditions × (NIAH + PPL) → G2 verdict; incremental
  per-condition JSON flush; provenance logged.
- `run_stage2.sbatch` — SLURM (partition main, gpu:rtx6000:1, seeds 0/1/2, --require-real-data).

## Pins held CONSTANT (Stage 1 PREREGISTRATION + plan §6)
- **[PIN-1]** gdn2-1.3B, **100B paper-matched checkpoint ONLY** (10B rejected by
  `resolve_and_assert_ckpt`), Config.from_name + strict=False, bf16, fused_recurrent, 18×16 heads.
- **[PIN-2]** threshold-rank eps=1e-4 (Eq.6), θ_R=0.5 — same classifier Stage 1 used.
- **[PIN-5]** 3 natural domains from `data_cache/` real texts; `--require-real-data` hard-fails on
  fallback.
- **[PIN-6]** seeds 0/1/2 govern random-mask draw + data/needle sampling; torch/np/PYTHONHASHSEED
  pinned; **k IDENTICAL across high/low/random**.

## Pre-registration caveat (echo verbatim)
Paper's 93.8 / 46.9 / 90.6 NIAH + KV 38.9% are **Qwen3-Next (48-layer, POST-TRAINED)** observations
adopted as reproduction **TARGET lines, NOT pass standards** for 18-layer gdn2. Falling short =
generalization limitation to record, not code failure.

## Run
```bash
# smoke (CPU, no model — checks classifier count-match, masking API, niah recall, ppl, verdict)
python stage2_pruning.py --smoke

# greenbeard SLURM (NEVER direct CUDA on login node)
sbatch run_stage2.sbatch                 # seeds 0/1/2, incremental flush, --resume-safe
# or one seed:
export GDN2_CKPT_PATH=/home/sohyung/models/gdn2_1.3B_100b.pth
export TRITON_CACHE_DIR=/home/sohyung/.triton_cache HF_HUB_DISABLE_XET=1
python stage2_pruning.py --seed 0 --require-real-data --resume --out results/stage2_100b_seed0
```
Output: `results/stage2_100b_seed{0,1,2}/` — `head_classification.json`, `cond_{origin,high,low,
random}.json` (incremental), `stage2_report_<YYMMDD>_<tag>_seed<N>_<githead>.json`.
