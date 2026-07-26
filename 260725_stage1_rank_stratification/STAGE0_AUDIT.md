# Stage 0 (G0) — repo audit & reusable-asset inventory (no code change)

Gate G0: inventory the repo, identify reusable assets for the gdn2-1.3B rank-stratification
reproduction, and confirm nothing must be re-implemented that already exists.

## Reusable assets (found)

| Asset | Path | Reused for |
|---|---|---|
| **Canonical gdn2-1.3B loader** | `260722_exp/common.py` (`load_model`, `Bundle`, `CONFIG_NAME="gdn2_1.3B"`) | [PIN-1] model load: lit_gpt `Config.from_name`, `strict=False`, bf16, `fused_recurrent`, `GatedDeltaNet2.forward` monkeypatch + shared `fla.models.utils.Cache` -> per-head `recurrent_state` `{layer:(heads,dk,dv)}`. `loader_gdn2.py` imports this directly; does NOT re-implement. |
| **entropy eRank / stable rank** | `notebooks/capacity_utils.py` (`effective_rank`, `stable_rank`) | [PIN-3] `rank_metrics.py` imports these verbatim (no re-implementation). |
| **prefix-sweep state trajectory** | `260722_exp/analysis.py` (`state_trajectory`) | [B] time-consistency: same prefix `[:t]` re-run pattern -> per-t per-head rank vectors. |
| **domain data-loader pattern** | `260722_exp/data.py` (`make_*(tok)->ids`) | [3] `data_stage1.py` follows the same `make_*` signature; adds 3 public domains + App.D attacks. |
| **model setup gotchas** | `MODEL_SETUP.md` §3 | gdn2 load: `layer_idx` manual set, monkeypatch cache, bf16, TinyLlama tokenizer, state shape `(1,16,128,128)` for 370m. |
| **Stage 2/3 skeletons (DO NOT TOUCH — G1a-gated [PIN-7])** | `notebooks/mqar_fromscratch.py`, `notebooks/pretrained_decay_mqar.py` (`make_mqar`), `260722_exp/niah_ruler.py` (S-NIAH GT) | planted-MQAR + S-NIAH for Stage 2/3 — not started. |

## Assets NOT present locally (VESSL /root only)
- `nb3_F6F7_decomposition_1p3B.py` — the "nb3" the spec extends. **Not in the repo locally.** Its
  loader/state-capture logic is fully captured by `260722_exp/common.py` (same monkeypatch pattern),
  which is what Stage 1 builds on. F6/F7 decomposition (entropy eRank + r̄) is re-derived here.
- `rebuild/F6_1p3B_data.npy` (per-head r̄ + entropy eRank) — VESSL-only cache. Per the plan, we
  **re-log rather than reuse** (avoids settings drift). No local cache dependency.
- `dscpkg`/`lit_gpt` (dsc training package) — VESSL `/root/dscpkg` (or local `/home/sohyung/long-gdn/dsc`).
  `common.py._find_lit()` resolves it; on this box it is absent, so **GPU logging runs on VESSL only**.

## Checkpoint note
- Spec [PIN-1] names `checkpoint-10B` = `/root/gdn2_1.3B_10B.pth`. Repo `common.py` defaults to a
  different HF slim weight (`gdn2-1.3b-weights.pth` / `model-95b.pth`). `loader_gdn2.load()` sets
  `GDN2_CKPT_PATH` to the 10B checkpoint so the canonical loader resolves it. **Pass `--ckpt` (or
  env `GDN2_CKPT_PATH`) to the exact checkpoint-10B path on VESSL.**

> **CORRECTION (2026-07-26) [PIN-1].** The 10B checkpoint is **superseded**. State rank is sensitive
> to training amount, so the reproduction uses the **100B paper-matched** `model-100b.pth`
> (`/home/sohyung/models/gdn2_1.3B_100b.pth`, 17.4 GB). The loader now **rejects any 10B checkpoint**
> and **accepts 95B** as a paper-matched near-equivalent. Run on greenbeard SLURM (not VESSL). This
> §checkpoint-note above is retained as the original Stage-0 record.

## G0 verdict
PASS — all Stage-1 primitives (loader, 2 of 3 rank metrics, prefix-sweep, data pattern) exist and are
reused; only the threshold-rank (Eq.6), time-consistency stats, 3-domain+attack data, cross-domain
agreement, and r̄->3-metric regression are new (implemented in Stage 1). No `experiment_protocol.md`
exists in this repo (confirmed via `find`); canonical conventions anchored to plan §6 + rebuild F6/F7
usage, per consistencyNotes.
