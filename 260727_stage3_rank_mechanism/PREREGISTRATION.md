# Pre-registration — Stage 3 (G3): rank = GENUINE CAPACITY vs oversaturated junk (pure gdn2-1.3B, 100B)

Registered BEFORE the logging run. Post-hoc changes require a dated rationale in this file [PIN-4]
(mirrors Stage 1). Constants are echoed into `stage3_report.json` (grids, segment_len, provenance).

## Question / hypothesis
Refute the paper's (arXiv:2602.02195 §5) chain **"full rank ⟹ oversaturated ⟹ prunable"** by
dissociating two definitions of saturation on the SAME heads Stage 2 used:
- **(A) linear-algebraic saturation** = full threshold-rank (paper Eq.6).
- **(B) functional saturation** = capacity exceeded, information destroyed by interference.

Stage 2 already showed the REVERSAL (high-rank KV-state pruning is catastrophic; low ≈ random). Stage
3 asks whether high-rank heads **USE** that rank capacity (genuine) or carry saturated junk, via
post-hoc **spectral-content surgery** on each head's recurrent state `S_h ∈ (dk,dv)`.

**H (G3):** high-rank heads carry information filling their state's spectral capacity, so degrading
the spectral CONTENT of high-rank `S_h` (SVD top-r truncation; spectrum-matched singular-vector
randomization) degrades LM substantially more than the same op on low-rank / count-matched random
heads; low-rank heads are already low-dimensional and near-insensitive.
**NULL/reverse** (high-rank flat under top-r/spectrum-noise) = rank genuinely idle/redundant → recorded
as a generalization LIMITATION, not a code failure.

## DV
- **PRIMARY = macro PPL delta vs origin** (2^mean-bits/token over wikitext/github/arxiv real text) as a
  function of intervention dose, per head group {high, low, random}.
- **SECONDARY (headroom-gated) = S-NIAH RULER-multikey retrieval accuracy** — reported ONLY if origin
  ≥ **0.30**, else **UNTESTABLE_FLOOR** (Stage 2 origin was 0.18). De-scoped by default (`--niah`
  off): PPL is the sole signal.

## CONSTANTS (CONSTANT across sessions — do NOT tune per run)
- **[PIN-1]** model = gdn2-1.3B, `Config.from_name("gdn2_1.3B")`, `load_state_dict(strict=False)`,
  bf16, `mode="fused_recurrent"`. Checkpoint = **100B paper-matched** `model-100b.pth`
  (`/home/sohyung/models/gdn2_1.3B_100b.pth`, sha256[:16]=4b03319f); **95B accepted** near-equivalent;
  **10B HARD-REJECTED** by `resolve_and_assert_ckpt`. Architecture: **18 layers × 16 heads = 288
  heads**, `num_v_heads == num_heads == 16` (no GVA). Confirmed from config: `gdn2_per_layer=1` ⇒ all
  18 layers are `GatedDeltaNet2` (PURE GDN, no attention layers ⇒ no cross-segment RoPE/KV offset;
  state carries purely via the recurrent Cache). NO regression to 370m, NO shallow/untrained model.
- **[PIN-2]** head rank metric = `threshold_rank` ε=**1e-4** (`Rank_eff = Σ_i I(σ_i > ε·σ_1)`, Eq.6),
  cap `d = min(dk,dv)`; classification threshold **θ_R = 0.5** on normalized rank. Groups re-derived
  in-process via `head_classifier.classify` → **k = min(#low,#high)** DISJOINT count-matched bottom-k
  / top-k sets (`isdisjoint` asserted). Cross-domain agreement sanity gate ≈ 0.97 (Stage 1 = 0.971).
  **DECLARED DEVIATION** from JRNP Eq.14 `S_h = α·(r̄/d)+(1−α)·(n̄/max n̄)`: α is UNPUBLISHED, so we
  classify on normalized threshold-rank alone.
- **[PIN-5]** data = `data_cache/` real 3 domains (wikitext-103 / codeparrot-clean / ccdv-arxiv), ≥16
  seqs × up to 2048 tok, `--require-real-data` (synthetic fallback → INVALID).
- **[PIN-6]** seeds **0/1/2**, `PYTHONHASHSEED/random/numpy/torch/cuda` pinned per `--seed`; ONE seed
  per sbatch job + a dependency aggregate (mean±std). Spectrum-noise RNG is derived per (seed, layer,
  head) so it is reproducible per seed.

## NEW Stage-3 constants — FROZEN here BEFORE the logging run
- **Intervention grids** (`stage3_mechanism.py`):
  - **int-1 prune-fraction** (KV-state v-zeroing via `head_mask.HeadMasker`, single-shot path):
    `{0, 0.10, 0.20, 0.30, 0.389, 0.43, 0.50, 0.60}` of heads per group. **0.389** marks the paper's
    KV 38.9% reduction; **0.43** marks our Stage-2 point. random group draws its fraction from the
    WHOLE head pool (count-matched control).
  - **int-2 SVD top-r retained-rank ratio** `r/cap ∈ {1.0, 0.75, 0.50, 0.25, 0.125, 0}` per group
    (segmented path). `ratio=1.0` == no-op origin; `ratio=0` == zero-state.
  - **int-3 spectrum-noise ladder** = `origin → top-r(r/cap=0.5) → spectrum-noise → zero` per group
    (segmented path).
- **Segmented intervention & segment_len** — the load-bearing design decision (verified 3-lens):
  GDN2 readout `o_t = S_t q_t` is INSIDE the fused/chunk kernel and the forward exposes ONLY the FINAL
  state (`output_final_state` → `cache["recurrent_state"]`, shape `[N,HV,K,V]`). Single-shot
  final-state surgery would perturb only the LAST token and cannot move PPL. **CHOSEN method =
  SEGMENTED**: split each sequence into segments of **segment_len = 256** (FROZEN), run the model per
  segment through the loader's SHARED-cache carry (`260722_exp/common.py` monkeypatch forces
  `use_cache=True` so the fla `Cache` threads recurrent_state AND conv_state across forwards), apply
  SVD-top-r / spectrum-noise / zero to the returned per-head `S_h` at each boundary, and let the next
  segment's readout flow through the surgered state (`state_surgery.SegmentedSurgeon`).
  - **S=1 CONTROL (hard):** `segment_len ≥ seq_len` ⇒ 1 segment, 0 boundaries, no surgery ⇒ MUST
    reproduce the single-shot origin PPL bit-for-bit (harness faithfulness). Asserted per seed and
    reported in the verdict; an aggregate-majority failure flags the G3 signal as suspect.
  - **Multi-segment origin drift (diagnostic):** the surgery-off forward at segment_len=256 vs
    single-shot is reported. GDN2's recurrence is associative and both recurrent+conv state carry, so
    drift ≈ 0. int-2/int-3 deltas are measured vs the **segmented same-segment_len origin** so any
    residual drift cancels (pure surgery isolation).
- **Spectrum-noise construction** (content-destroying null): `S_noise = U_rand diag(σ_orig) V_rand^T`
  with **Haar-random orthogonal** `U_rand (dk×dk), V_rand (dv×dv)` (QR of a Gaussian, sign-fixed R
  diagonal ⇒ orthogonal Haar measure). This PRESERVES the singular values exactly ⇒ identical
  `threshold_rank`, `nuclear_norm`, and Frobenius energy (asserted in the smoke test via
  `rank_metrics`), while RANDOMIZING the singular vectors ⇒ destroys the stored CONTENT.
- **Headroom gate:** origin NIAH < **0.30** ⇒ NIAH UNTESTABLE_FLOOR ⇒ PPL is the sole DV (PREREG).

## Decision gate (on the ≥3-seed aggregate, PIN-6)
Signals (deltas vs each group's OWN origin so a group-baseline shift can't confound):
- **int-2**: slope of macro PPL vs content-loss dose `(1 − r/cap)`; steep for HIGH, flat for LOW.
- **int-3**: spectrum-noise macro-PPL delta vs the group origin; large for HIGH (content matters
  despite preserved rank/energy), ≈0 for LOW/junk.
- **Contrast** = `delta(high) − delta(low)`, and both vs `delta(random)` (3-seed mean±std).

| Gate | Condition |
|---|---|
| **G3 = YES** | mean SVD-top-r slope(high) > slope(low) **AND** mean spectrum-noise delta(high) > delta(low) ⇒ high-rank USES its spectral capacity ⇒ NOT oversaturated ⇒ paper's rank→saturation leap REFUTED on pure GDN. |
| **G3 = NO/NULL** | high-rank not more content-sensitive than low on both surgeries ⇒ high-rank rank genuinely idle/redundant ⇒ record as generalization LIMITATION. |
| **valid** | data authenticity (`--require-real-data`) + S=1 control reproduces origin in the seed majority; else results flagged. |

## PREREG CAVEAT (echoed verbatim into the report)
The paper's **93.8 / 90.6 / 46.9 NIAH + KV 38.9%** are **Qwen3-Next (48-layer, POST-TRAINED)** TARGET
lines, **NOT pass standards** for 18-layer pure gdn2; a shortfall is a generalization limitation to
record, not a code failure. Interventions are **POST-HOC state surgery** (not retraining). S-NIAH here
is **RULER-multikey** (≠ the paper's single-needle gkamradt), so only the **within-run
high/low/random asymmetry** is the G3 signal — absolute values are not comparable. Hybrid `swa_gdn2`
comparison = future work (no matched checkpoint).

## Consistency check before running
Re-run `head_classifier` under the same seeds and confirm cross-domain agreement ≈ 0.97 and the SAME
high/low head sets as Stage 2, so the Stage 3 groups match the Stage 2 pruning-asymmetry result this
stage mechanistically explains.

---
_Post-hoc change log (dated) — none yet._
