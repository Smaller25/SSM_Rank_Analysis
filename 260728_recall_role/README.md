# Stage 4 — HIGH-rank head RECALL-ROLE test (gdn2-1.3B, 100B paper-matched)

Stage 2 showed HIGH-rank KV-state pruning catastrophically raises PPL (~32×: 12.6 → 854), but PPL is
a lumped metric that hides **which role** is lost. Stage 4 decomposes it with a standard mech-interp
**induction-head probe** on REAL text (Olsson et al. 2022; Elhage et al. 2021), adapted to the SSM
KV-state setting, to test whether the HIGH-rank heads are the **in-context recall (memory) units**.

This refutes — in the *role* dimension — the paper's (arXiv:2602.02195 §5) **oversaturation** reading
that treats high-rank heads as prunable junk: if HIGH-rank pruning collapses *recall* specifically,
high-rank heads are recall units, not junk.

## Probe (real-text 2× repetition, NO synthetic / distribution swap — PIN-5)
Each real passage `A` (length `L=1024`) is repeated once → `seq = [A][A]` (length `2L=2048`, within
`block_size` 4096). Per-token NLL (`analysis.token_nll_bits`, length `2L−1`, target=`ids[1:]`) is
split:
- `local_bits`  = mean(bits[0 : L−1])   — first copy, **no** in-context recall possible
- `recall_bits` = mean(bits[L : 2L−1])  — second copy, in-context recall **is** possible
- `induction_gain = local_bits − recall_bits`  (>0 ⇒ recall is working)

The seam token `bits[L−1]` (predicting the first token of the second copy) is dropped from both.

## Verdict (recall-role)
Baseline = `origin`. `Δrecall(c)=recall_bits(c)−recall_bits(origin)`, `Δlocal` analogous.
`H(recall-role)` supported iff, on aggregate across ≥3 seeds:
1. **Headroom** (pre-registered): `origin induction_gain > 0.30 bits` in the majority of seeds, else
   `UNTESTABLE_HEADROOM`.
2. **Recall-specific**: `Δrecall(high) > Δlocal(high)`  (damage concentrated on recall, not a uniform
   PPL hit).
3. **HIGH-specific**: `Δrecall(high) > Δrecall(low)` AND `> Δrecall(random)`.

NULL/reversal (recall-specificity or HIGH-specificity fails) = **LIMITATION to record, not a code
failure**.

## Four count-matched conditions (identical k across high/low/random — PIN-4)
`origin` (no mask) · `high` (zero k HIGH-rank heads' KV state) · `low` (zero k LOW-rank heads) ·
`random` (zero k random heads; seed governs the draw — the critical control).

## Files
- `induction_probe.py` — **NEW**: builds `[A][A]` probes from real passages (`data_stage1.load_all`,
  reused) and splits per-token NLL into local/recall segments (`analysis.token_nll_bits`, reused). No
  distribution replacement.
- `stage4_recall_role.py` — driver, mirrors `stage2_pruning.py`: classify → 4 conditions ×
  segment-split PPL → verdict; incremental per-condition JSON flush + `--resume`; provenance logging;
  `--smoke` CPU path.
- `aggregate_seeds.py` — mirrors Stage 2 aggregate: `Δrecall`/`Δlocal`/`induction_gain` mean±std,
  majority headroom gate, final recall-role verdict.
- `run_recall_role.sbatch` / `submit_recall_role_seeds.sh` / `run_aggregate.sbatch` — 1 seed/job +
  `afterok` aggregate (mirrors Stage 2).

## Reused unmodified (no re-implementation)
`head_classifier.classify` (k=min disjoint count-match, PIN-2) · `head_mask.HeadMasker` (KV-state
v-zeroing, PIN-4) · `analysis.token_nll_bits` (canonical NLL) · `data_stage1` (loader + real texts,
PIN-5) · `loader_gdn2.load` · `common` (PIN-1).

## PINs carried (CONSTANT, inherited from Stage 1/2/3)
- **[PIN-1]** gdn2-1.3B, 100B ckpt ONLY (10B HARD-REJECT), `Config.from_name("gdn2_1.3B")` +
  strict=False + bf16 + fused_recurrent. n_layer=18, mixer `num_heads==num_v_heads==16` → **288
  heads** (config `n_head=18` is the attention field, unrelated to the mixer). Head provenance
  (num_heads/num_v_heads/total) logged on the first real run.
- **[PIN-2]** threshold-rank eps=1e-4, θ_R=0.5, disjoint count-match, cross-domain agreement ~0.97
  sanity gate. Pure-rank (≠ JRNP Saturation Score, α undisclosed) DECLARED DEVIATION.
- **[PIN-4]** KV-state v-zeroing; k IDENTICAL across high/low/random.
- **[PIN-5]** 3 natural domains from `data_cache/` real texts + `--require-real-data` (fallback →
  INVALID). Induction is real-text 2× repetition, NOT a distribution swap.
- **[PIN-6]** seeds 0/1/2; torch/np/PYTHONHASHSEED/cuda pinned; 1 sbatch/seed + `afterok` aggregate
  mean±std.
- Headroom threshold **0.30 bits** is a pre-registered value (same headroom-gate pattern as Stage 2/3
  NIAH floor 0.30).

## Caveat (echo verbatim, PREREG)
Paper's 93.8/46.9/90.6/38.9% are Qwen3-Next (48-layer, **post-trained**) OBSERVATIONS adopted as
reproduction TARGET lines, **NOT** pass standards for 18-layer pure gdn2; falling short = a
generalization/role limitation to record, not a code failure. Stage 2 `origin` PPL (12.6±0.4) should
reproduce here (same loader/classifier/masking reused) — if not, suspect loader/data drift.

## Run
```bash
export GDN2_CKPT_PATH=/home/sohyung/models/gdn2_1.3B_100b.pth
export TRITON_CACHE_DIR=/home/sohyung/.triton_cache HF_HUB_DISABLE_XET=1
bash submit_recall_role_seeds.sh          # seeds 0/1/2 + afterok aggregate
```

## Smoke (CPU, no model, seconds)
```bash
python stage4_recall_role.py --smoke
```
Exercises: classifier count-match/disjoint + agreement, `[A][A]` build (length 2L) + segment split
(L−1 each), `induction_gain>0` on a planted recall-cheaper bundle, verdict recall-role signature, and
the headroom gate (origin gain < 0.30 → `UNTESTABLE_HEADROOM`).
