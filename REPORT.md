# Information Capacity of Fixed-Size Recurrent States — Interim Report

Study of what limits associative recall in fixed-size recurrent LMs (Mamba-2 SSD vs Gated DeltaNet),
which information-theoretic signal tracks it, and how information density sets a dynamic-chunking length.
All results run on an RTX PRO 6000 (Blackwell) via SLURM; code in `notebooks/`, shared helpers in
`notebooks/capacity_utils.py`, theory notes in `theory/`.

> **Scope.** This report covers the *diagnostic* sub-study (which signal = capacity, how much a state
> holds, how to trigger chunking). The end-to-end **snapshot→route→reuse** system (H3) is deliberately
> out of scope here and is pursued in the linear-memory-routing project.

## TL;DR
1. **eRank ≠ capacity.** Fixed-size states use only ~7–11 % of their max rank, and eRank is
   **anti-correlated** with recall under load (eRank↑ while recall↓). eRank is a state-spectrum /
   input-composition diagnostic, not a capacity meter.
2. **Capacity = model recall** (the state's own `C`-read; associative read-out is `value ≈ C·h`).
3. **The bottleneck is associative load (# key–value pairs), and length/load are entangled** — see the
   caveat in §3: our "distance-robust" horizon result is largely an artifact of low-information filler
   padding; with real content, longer context *means* more keys ⇒ more interference.
4. **Information density sets chunk length**: cumulative **epiplexity** predicts the density→chunk-length
   relation tightly (Spearman ρ≈0.94), eRank weakly (ρ≈0.73); the relation is a *saturation* effect.
5. **Update-rule compression ratio**: per unit state memory, **Mamba-2 SSD ≈ plain Gated-DeltaNet**
   (~10–11 recalled-bits / Mfloat); the **MoM mixture wrapper is much less efficient** (2.1). The
   headline "GDN recalls better" is not attributable to the gated-delta *update rule* here.

Models: `state-spaces/mamba2-370m` (SSD), `linear-moe-hub/Gated-Deltanet-340M` (plain gated delta,
loaded via a fused-MLP/`attn.D` weight adapter), `linear-moe-hub/MoM-Gated-Deltanet-340M` (MoM). All are
pretrained general LMs, **not** trained on the synthetic MQAR task (so recall is in-context).

---

## 1. Signal catalog & state utilization (`notebooks/information_capacity_signals.ipynb`)
Six signals — **S1** effective rank (eRank), **S2** predictive entropy / bits-per-token, **S3**
in-context epiplexity, **S4** ground-truth bits, **S5** Rényi-2 / participation-ratio rank, **S6** TwoNN
intrinsic dimension — over three datasets (MQAR, WikiText-2, A5 state-tracking) and two models.

**State utilization is low and roughly data-independent:** peak eRank on MQAR is 6.99 / 64 = **10.9 %**
(mamba2) and 27.52 / 256 = **10.7 %** (GDN-MoM). Both models leave ~90 % of their state rank unused.

Per-position trajectories (Δ = final − initial) and raw vs normalized signal curves:

![Δ signal matrix](notebooks/capacity_results/full_matrix_delta.png)
![signal trajectories (normalized)](notebooks/capacity_results/signal_trajectories.png)
![signal trajectories (raw)](notebooks/capacity_results/signal_trajectories_raw.png)

Worked example — S1 eRank vs MQAR load, both models (note the different state-matrix sizes):

![eRank vs MQAR load](notebooks/capacity_results/worked_example_S1_D1_both.png)

---

## 2. eRank ≠ capacity; capacity = recall (`notebooks/state_capacity_decodable.ipynb`)
Model recall (the state's own `C`-read) is the capacity signal. We disentangle **load** (# pairs `N`)
from **horizon** (context length `L`) with padded MQAR, and overlay eRank.

- **LOAD** (`L=512` fixed, vary `N`): recall **1.00 → 0.43** for N=2→200, while eRank **rises** 1.9 → 4.4.
  → **anti-correlated**: more eRank is an *overload* symptom, not headroom.
- **HORIZON** (`N=8` fixed, pad to `L=32→4096`): recall 0.92 → 0.82, eRank **falls** 3.4 → 1.8. → decoupled.

![load vs horizon](notebooks/state_capacity_results/load_vs_horizon.png)

**Caveat (important, corrects an earlier overstatement).** The horizon axis pads with a repeated
low-information filler token, which a selective SSM largely ignores — so the pairs survive to 4k. Real
long context is *diverse* content, i.e. more keys + interference. **Length and load are therefore
entangled**; the honest reading is that a bounded state has an associative-capacity limit that long
context stresses *because it carries more content*, consistent with the standard view that fixed-state
models are capacity-limited on long-context recall. The clean, filler-free result is the LOAD axis.

---

## 3. Dynamic chunking by information density (`notebooks/dynamic_chunking_by_density.ipynb`)
If we cut a chunk when the state's capacity signal plateaus, does chunk length depend on input
information density? Density is set by a controlled repetition knob and measured as bits/token.

- **Synthetic (10 seeds/level):** chunk length **grows with density** — epiplexity criterion Spearman
  ρ = **0.94** (95 % CI [0.89, 0.96]); eRank criterion ρ = **0.73** ([0.53, 0.86]). This is the
  *saturation* view (low-density, repetitive input saturates the state fast → short chunks).
- **Natural-passage cross-check (48 WikiText passages):** within the narrow natural density band the
  relation is weak/degenerate — eRank ρ ≈ **−0.01** ([−0.30, 0.28]), epiplexity degenerate (one chunk).
  → the strong synthetic effect is partly an artifact of the wide density range that repetition creates.

![chunk boundaries on eRank(t)](notebooks/chunking_results/worked_example_boundaries.png)
![chunk length vs density (synthetic, 10 seeds)](notebooks/chunking_results/chunk_by_density.png)
![chunk length vs density (natural passages)](notebooks/chunking_results/natural_passage_chunks.png)

Takeaway: **epiplexity is the better contents-based chunk-length signal**, but it is a loss/data-density
measure (not a state measure), and the density→length effect needs a wider-density validation.

---

## 4. Update-rule compression ratio (`notebooks/stored_vs_used_gap.ipynb`)
Per-state **capacity** `C = max # keys retrievable at recall ≥ 0.9`, normalized by state size (matched
memory) → a **compression ratio** that ranks *update rules* by how well the recurrence uses memory.

| update rule | state (Mfloat) | C_used@0.9 (raw keys) | **recalled bits / Mfloat** |
|---|---|---|---|
| Mamba-2 (SSD) | 12.58 | 23.3 | **11.13** |
| plain Gated-DeltaNet (gated delta) | 6.29 | 10.7 | **10.17** |
| MoM Gated-DeltaNet (mixture) | 31.46 | 11.0 | 2.11 |

![stored vs used, both models](notebooks/stored_vs_used_results/stored_vs_used.png)

- **SSD ≈ plain gated-delta** per unit memory (~10–11 bits/Mfloat). Mamba-2's higher *raw* capacity
  (23 vs 11) is mostly because its state is ~2× larger (12.6 vs 6.3 Mfloat); capacity scales ~linearly
  with state size, so the *update rule itself* does not change per-byte efficiency much in this setting.
- **MoM is memory-inefficient**: 5× the state for no raw-recall gain (idle memories inflate its size).
  MoM is a mixture *wrapper*, not an update rule, and is excluded from update-rule conclusions.

**Caveat.** The `stored` side (a bilinear probe meant to measure info present-but-unread in the raw
recurrent state) read at **chance** here and in an earlier variant, because an external probe with random
/ input-embedding key features cannot align with the model's internal `B`/`C` addressing. So we could
**not** measure a stored≥used gap; all capacity numbers above are the *realized* (recall) capacity, not a
theoretical maximum. A `B/C`-aware probe or a recurrent-state SAE is needed to size the gap.

---

## 5. Caveats & limitations (consolidated)
- **Length vs load are entangled**; the horizon "distance-robustness" is inflated by inert filler (§2).
- **`stored` probe invalid** (chance) — realized (recall) capacity only; no stored≥used gap yet (§4).
- **Pretrained, not MQAR-trained** — recall is in-context; trained models could differ.
- **eRank(spectral) ≠ algebraic rank** — random/​trained weights are a.s. full algebraic rank; low eRank
  is spectral concentration by the dynamics (see `theory/random_matrix_full_rank.md`).
- Coarse `N`/density grids; MoM compared at native (inflated) state size.

## 6. Conclusions & next
Established (diagnostics): capacity = recall; eRank is not capacity (anti-correlated under load);
epiplexity is the density/chunk-length signal; per-memory compression of SSD ≈ plain gated-delta.
**Not yet done here:** (a) update-rule comparison on the **long-context / horizon axis** with *real*
content (the axis that actually differentiates forgetting mechanisms); (b) a valid stored-capacity probe;
(c) the end-to-end **snapshot → route → reuse** system that would turn per-chunk capacity into extended
effective capacity (H3) — pursued in the linear-memory-routing project.

*Roadmap and hypotheses: `theory/research_goal.md`. Theory note: `theory/random_matrix_full_rank.md`.*
