# Research goal & roadmap — capacity bottleneck & dynamic chunking

**North-star thesis.** A fixed-size recurrent state has an **associative capacity ceiling**. When load
exceeds it, retrieval fails — this **capacity (load) limit, not context length, breaks multi-key recall**.
Detecting saturation and **dynamically chunking** the stream (snapshot/reuse multiple states) raises the
*effective* capacity and should **restore retrieval, especially for multi-key**.

## Hypotheses
- **H1** — fixed-size state has an associative capacity ceiling; past it, values can't be retrieved
  (single- and multi-key), **worse for multi-key**.
- **H2** — the failure is **capacity/load-driven, not context-length/horizon-driven**.
- **H3** — chunking at saturation increases effective capacity → restores retrieval (esp. multi-key).

## Evidence status
| | status | evidence |
|---|---|---|
| **H2** | ✅ confirmed | nb3: recall collapses with load N (1.00→0.43 for N=2→200) but is **distance-robust** (0.92→0.82 over L=32→4096). The "~200-tok eRank saturation" is a horizon artifact, not a capacity limit. |
| **H1** | 🟡 partial | recall falls with N (multi-key worse). Open: is it *truly full* (info lost) or *read-limited* (info there, unread)? → **E2/nb4**. |
| **H3** | 🔴 untested | chunk-length rule built (nb2), but "chunking restores multi-key recall beyond the single-state ceiling" not yet tested → **E3**. |

## Signal roles (settled)
- **Capacity = model recall** (the state's own C-read; `value ≈ C·h`). This is the ground-truth signal.
- **eRank ≠ capacity** — it is anti-correlated with recall under load (eRank↑ as recall↓) and decoupled
  under horizon. Reusable as an **overload / when-to-chunk trigger**, not a capacity meter. See
  `theory/random_matrix_full_rank.md` (algebraic rank ≠ effective rank).
- **epiplexity** — input information-density / in-context structure; used to set **chunk length** (nb2).

## Experiment pipeline
- **E1 (done, nb3)** — capacity is load-limited, distance-robust; recall is the capacity signal.
- **E2 (nb4 `stored_vs_used_gap`)** — at saturation, *truly full* (stored≈used, info lost) vs
  *read-limited* (stored>used)? Bilinear probe with learnable key projection = "stored"; model recall =
  "used". Decides whether H3 (more state via chunking) is the needed remedy or a better read-out suffices.
- **E3 (payoff, H3)** — **chunked multi-state retrieval**: split the stream, snapshot states at
  saturation, combine/route at query time; does multi-key recall recover beyond the single-state ceiling
  (N~128–200)? The deferred **(3) SSC / LLM-memory-routing** is the state-combination mechanism here.

## Notebooks
- `information_capacity_signals.ipynb` — S1–S6 signal catalog × datasets × {mamba2-370m, MoM GDN-340m}.
- `dynamic_chunking_by_density.ipynb` — chunk length vs information density (epiplexity/eRank triggers).
- `state_capacity_decodable.ipynb` — recall vs eRank; load vs horizon (E1).
- `stored_vs_used_gap.ipynb` — bilinear-probe stored vs used (E2).
- `capacity_utils.py` — shared loaders / state extraction / signals.

## Pilot context (memory-routing / SSC) & the open questions
The bigger project: fix long-context recall by **freezing saturated states and routing over them**
(SSC / LLM-memory-routing). Pilot finding (fixed-length snapshots + routing):
- **single-query recall improves dramatically** (routing avoids whole-context saturation; one chunk, one read).
- **multi-query recall drops** — and the diagnosis is **"the right chunk is found, but retrieval *within*
  the chunk fails"** — i.e. the per-chunk state is itself saturated for multiple keys. Same count/capacity
  ceiling, now inside a chunk.

Open questions:
- **Q1 — which states to store?** (can't keep all). = an eviction/selection policy driven by a saturation
  signal.
- **Q2 — why does multi-query fail *inside* the found chunk?** → is the info still there but unread
  (`stored>used`) or genuinely gone (`stored≈used`)? → **nb4 (E2)**.
- **Q3 — dynamic (not fixed-length) chunking needs an unsupervised, contents-based saturation signal** →
  pick it by comparing **S1–S6 vs actual recall** (nb1/nb3).

## Why this matters — the compression-ratio test (the make-or-break criterion)
This methodology is worthwhile **only if one frozen chunk-state reliably serves several keys** (k ≫ 1):
- **k ≫ 1** → you store far fewer states than a full KV cache yet keep recall → you keep the SSM's
  small-memory / linear-cost advantage **and** extend capacity → **meaningful**.
- **k ≈ 1** → to save multi-query you'd chunk down to one item per state = storing everything =
  **you've just re-invented the attention KV cache** → the SSM advantage is gone → **not meaningful**
  (and Q1 collapses into the KV-eviction problem attention already has).

So the real research question is **not "does routing help"** but **"how many keys does one frozen
state compress (the per-chunk compression ratio) vs a KV cache?"** — which is exactly what nb3/nb4
measure. The pilot's "multi-query found-but-not-retrieved" is a warning that per-chunk effective
capacity may be ~1; **nb4 decides**: (a) `stored>used` → better read-out rescues it; (b) `stored≈used`
→ per-chunk capacity is genuinely small → compression ratio is poor → the method's value is at risk.
