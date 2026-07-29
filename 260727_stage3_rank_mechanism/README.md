# Stage 3 — rank = GENUINE CAPACITY (pure gdn2-1.3B, 100B): mechanism rebuttal (G3)

Refutes the paper's (arXiv:2602.02195 §5) chain **"full rank ⟹ oversaturated ⟹ prunable"** by
dissociating **(A)** linear-algebraic saturation (full threshold-rank, Eq.6) from **(B)** functional
saturation (capacity exceeded, information destroyed). Stage 2 already showed the REVERSAL (high-rank
KV-state pruning is catastrophic, low ≈ random). Stage 3 asks whether high-rank heads actually **use**
their rank capacity, via post-hoc **spectral-content surgery** on each head's recurrent state `S_h`,
comparing head groups {high, low, random} (reusing Stage 2's `k=min` disjoint count-matched
classifier). Details + frozen constants: **`PREREGISTRATION.md`**.

## The load-bearing design decision — SEGMENTED intervention (`state_surgery.py`)
GDN2 readout `o_t = S_t q_t` is INSIDE the fused/chunk kernel; the forward exposes ONLY the FINAL
state (`output_final_state` → `cache["recurrent_state"]`, `[N,HV,K,V]`). Single-shot final-state
surgery would perturb only the last token and cannot move PPL. So we run the model **per segment**
(`segment_len=256`, FROZEN), carry state through the loader's SHARED fla-`Cache` monkeypatch
(`legacy/legacy/260722_exp/common.py`, which threads recurrent+conv state and forces `use_cache=True`), and apply
surgery to each targeted head's `S_h` **at every segment boundary** before it becomes the next
segment's `initial_state`. The readout of every later token then flows through the surgered state ⇒
causal, PPL-visible. **S=1 control** (`segment_len ≥ seq_len` ⇒ 1 segment, no boundary, no surgery)
MUST reproduce the single-shot origin PPL — asserted per seed.

Three surgeries (all CPU float per (layer,head), cast back):
- **int-2 SVD top-r** — replace `S_h` by its rank-`r` truncation, `r = round(ratio·cap)`; sweep
  `r/cap ∈ {1,0.75,0.5,0.25,0.125,0}`. Full-rank head ⇒ sharp PPL rise as `r` shrinks; low-rank ⇒ flat.
- **int-3 spectrum-matched noise** — `S_noise = U_rand diag(σ) V_rand^T` with Haar-random orthogonal
  `U,V`: PRESERVES the singular values (rank / nuclear norm / energy identical) but RANDOMIZES the
  singular vectors (destroys content). Ladder `origin → top-r(0.5) → spectrum → zero`.
- **int-1 prune-fraction** — `head_mask` KV-state v-zeroing (== zero-state) of a fraction of each
  group's heads, `{0,…,0.389 (paper),0.43 (ours),…,0.60}`; single-shot path (no surgery needed).

**int-4 (optional, `probe_decodability.py`)** — linear probe decoding the domain label from flattened
`S_h`; HIGH ≫ LOW corroborates that high-rank state carries genuine task info. Gated behind `--probe`
/ standalone; SECONDARY.

## Files
| file | role |
|---|---|
| `state_surgery.py` | **CORE**: spectral surgeries + `SegmentedSurgeon` (segment carry + boundary surgery) + invariant asserts. `python state_surgery.py` runs the CPU self-test. |
| `stage3_mechanism.py` | driver: classify → groups {high,low,random} → S=1 control → int-1/2/3 → G3 verdict; incremental flush, `--resume`, provenance. |
| `probe_decodability.py` | optional int-4 linear probe. |
| `aggregate_seeds3.py` | ≥3-seed mean±std dose×group curves + `delta(high)−delta(low)`/random + final G3. |
| `run_stage3.sbatch` / `submit_stage3_seeds.sh` / `run_aggregate3.sbatch` | one-seed-per-job SLURM + dependency aggregate. |
| `PREREGISTRATION.md` | frozen grids, segment_len, spectrum-noise construction, headroom gate, caveats. |

Reuses Stage 1/2 assets IN-PLACE (no forks): `loader_gdn2`, `common`, `data_stage1`, `rank_metrics`,
`head_classifier`, `head_mask`, `ppl_eval`, `niah_retrieval`.

## Run (greenbeard SLURM ONLY — never direct CUDA on the login node)
```bash
bash submit_stage3_seeds.sh            # seeds 0/1/2 as independent 6h jobs + afterok aggregate
# or one seed:  sbatch run_stage3.sbatch 0
```
Outputs → `results/stage3_100b_seed{0,1,2}/stage3_report.json`, aggregate →
`results/aggregate_stage3.json`.

## Smoke test (CPU, no GPU / no model — seconds)
```bash
source /home/sohyung/sh_gdn2_venv/bin/activate
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
python state_surgery.py                      # surgery invariants: spectrum preserves rank+nuclear; top-r reduces full-rank only
python stage3_mechanism.py --smoke --out results/smoke   # classify→disjoint groups→S=1 control→int-1/2/3→G3 verdict on a fake bundle
python probe_decodability.py --smoke --out results/smoke # int-4 wiring (skips gracefully if sklearn absent)
```
The `--smoke` fake bundle's readout depends on the state's CONTENT (alignment with a fixed template),
so the smoke faithfully exercises the int-2 slope AND int-3 spectrum dissociation (HIGH ≫ LOW) and the
S=1 control (|diff|=0), not just plumbing.

## G3 verdict (aggregate, PIN-6)
**G3 = YES** iff mean SVD-top-r slope(high) > slope(low) **AND** mean spectrum-noise delta(high) >
delta(low): high-rank heads use their spectral capacity ⇒ **not** oversaturated ⇒ the paper's
rank→saturation leap is refuted on pure GDN. Reverse/flat ⇒ rank genuinely idle → LIMITATION (PREREG),
not a code failure. PPL is the primary DV (S-NIAH floored at origin 0.18 → UNTESTABLE).
