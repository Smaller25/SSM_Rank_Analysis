"""Stage 2 head classifier — RE-DERIVE the low/high threshold-rank head sets from code.

The Stage 1 result JSON was lost with its worktree, so per PIN-2 / consistencyNotes we RE-COMPUTE
the per-(layer,head) classification here rather than trusting any stale JSON. This is the exact same
classifier Stage 1 used (stage1_repro.cross_domain_agreement): normalized threshold-rank vs
theta_R=0.5, aggregated across the 3 NATURAL domains (wikitext/github/arxiv) from data_cache/.

Method (mirrors Stage 1 [A] + [Dom]):
  1. loader_gdn2.load().states over the 3 natural domains (real texts, seed-dependent sampling).
  2. rank_metrics.threshold_rank per (layer,head) per sequence -> mean per head, per domain.
  3. normalize by cap d = min(dk,dv); label head HIGH(1) if (rank/cap) > theta_R=0.5 else LOW(0).
  4. cross-domain agreement (pairwise label consistency) — SANITY GATE (~0.97 expected).
  5. AGGREGATE label = per-head mean normalized rank across domains, split at theta_R=0.5.
     k = number of LOW-rank heads; we then take BOTTOM-k (low) and TOP-k (high) by aggregate
     normalized rank so the two masked groups are EXACTLY count-matched (k identical), which the
     asymmetry claim requires. If the theta_R split is already balanced, low==bottom-k, high==top-k.

Outputs a JSON dict: {low_heads, high_heads, k, per_head, cross_domain_agreement, provenance}.
Head identity is a (layer_idx, head_idx) pair; head_idx in [0, num_heads).

Reuse ONLY (no re-implementation): loader_gdn2, rank_metrics, data_stage1, common.
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
# reuse Stage 1 assets in-place (do not fork loaders/metrics/data)
_STAGE1 = os.path.abspath(os.path.join(_HERE, "..", "260725_stage1_rank_stratification"))
for _p in (_HERE, _STAGE1):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# vendored fla (mirror stage1_repro.py) so the gdn2 kernel + fla Cache resolve identically
_FLA_PATH = "/home/sohyung/linear-memory-routing"
if os.path.isdir(_FLA_PATH) and _FLA_PATH not in sys.path:
    sys.path.insert(0, _FLA_PATH)

from rank_metrics import threshold_rank, matrix_cap, EPS_THRESHOLD_RANK  # noqa: E402

NATURAL_DOMAINS = ["wikitext", "github", "arxiv"]   # [PIN-5] the SAME 3 natural domains
THETA_R = 0.5                                        # [PIN-2] theta_R split on normalized rank


def per_head_normrank(bundle, ids_list):
    """Mean per-(layer,head) NORMALIZED threshold-rank over a domain's sequences.

    Returns {(layer,head): norm_rank_mean}. norm_rank = threshold_rank / cap_d (both from the SAME
    per-head final state matrix, Eq.6 eps=1e-4). Averaging over >=16 seqs gives a stable per-head
    value (Stage 1 cross-domain agreement was 0.971)."""
    acc = {}
    for ids in ids_list:
        states = bundle.states(ids)                 # {layer: (heads, dk, dv)} CPU float
        for li in sorted(states.keys()):
            S = states[li]
            for h in range(S.shape[0]):
                M = S[h]
                cap = max(1, matrix_cap(M))
                nr = threshold_rank(M, eps=EPS_THRESHOLD_RANK) / cap
                acc.setdefault((li, h), []).append(nr)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def classify(bundle, tok, seq_len=2048, n_seq=16, seed=0, theta=THETA_R,
             domains=NATURAL_DOMAINS, require_real_data=True):
    """RE-DERIVE low/high head sets from the 3 natural domains. Returns a JSON-able dict.

    Aggregation across domains = mean normalized rank per head. LOW/HIGH by theta split; then
    count-matched to k = #low via bottom-k / top-k so |low|==|high|==k EXACTLY (asymmetry needs it).
    """
    import data_stage1
    data_stage1.set_data_seed(seed)                 # [PIN-6] seed-dependent sampling
    data = data_stage1.load_all(tok, seq_len=seq_len, n_seq=n_seq, which=list(domains))

    # [fix-4] fallback synthetic data may not ground the classification -> hard-fail if required
    fb = [d for d, (_, m) in data.items()
          if m.get("is_fallback") or m.get("source") == "fallback"]
    if fb:
        msg = f"[fix-4] natural domains fell back to SYNTHETIC data: {fb}."
        if require_real_data:
            raise RuntimeError(msg + " (--require-real-data set -> aborting classification).")
        print("  [WARN] " + msg + " Classification will be UNTRUSTWORTHY.", flush=True)

    per_domain = {}      # dom -> {(li,h): norm_rank}
    for dom, (ids_list, meta) in data.items():
        print(f"  [classify] domain {dom} ({meta['source']}) ...", flush=True)
        per_domain[dom] = per_head_normrank(bundle, ids_list)

    heads = sorted(set().union(*[set(d.keys()) for d in per_domain.values()]))

    # ---- cross-domain agreement (SANITY GATE ~0.97), same metric as Stage 1 cross_domain_agreement
    labels = {dom: {h: int(per_domain[dom].get(h, np.nan) > theta) for h in heads}
              for dom in per_domain}
    doms = list(labels.keys())
    pair_agree = {}
    for i in range(len(doms)):
        for j in range(i + 1, len(doms)):
            a, b = labels[doms[i]], labels[doms[j]]
            common = [h for h in heads if np.isfinite(per_domain[doms[i]].get(h, np.nan))
                      and np.isfinite(per_domain[doms[j]].get(h, np.nan))]
            agree = float(np.mean([a[h] == b[h] for h in common])) if common else float("nan")
            pair_agree[f"{doms[i]}|{doms[j]}"] = agree
    mean_agree = float(np.nanmean(list(pair_agree.values()))) if pair_agree else float("nan")

    # ---- aggregate normalized rank per head (mean across domains) -> label + count-matched sets
    agg = {h: float(np.nanmean([per_domain[d].get(h, np.nan) for d in doms])) for h in heads}
    low_by_theta = [h for h in heads if agg[h] <= theta]
    high_by_theta = [h for h in heads if agg[h] > theta]
    # count-match to k = #low (paper's groups are equal-count masks): bottom-k low, top-k high.
    k = len(low_by_theta)
    order = sorted(heads, key=lambda h: agg[h])       # ascending normalized rank
    low_heads = order[:k]                             # k lowest-rank heads
    high_heads = order[-k:] if k > 0 else []          # k highest-rank heads
    # guard: with theta=0.5 on a bimodal set, low_by_theta should equal bottom-k; log any mismatch.
    mismatch_low = sorted(set(low_heads) ^ set(low_by_theta))

    per_head = [{"layer": li, "head": h,
                 "agg_norm_rank": agg[(li, h)],
                 "per_domain": {d: per_domain[d].get((li, h)) for d in doms}}
                for (li, h) in heads]

    return {
        "theta_R": theta,
        "eps_threshold_rank": EPS_THRESHOLD_RANK,
        "n_heads_total": len(heads),
        "k": k,
        "low_heads": [list(h) for h in low_heads],
        "high_heads": [list(h) for h in high_heads],
        "low_by_theta": [list(h) for h in low_by_theta],
        "high_by_theta": [list(h) for h in high_by_theta],
        "theta_vs_bottomk_mismatch": [list(h) for h in mismatch_low],
        "cross_domain_agreement": {"pairwise": pair_agree, "mean": mean_agree,
                                   "domains": doms, "theta": theta},
        "per_head": per_head,
        "domains": list(data.keys()),
        "seed": seed, "seq_len": seq_len, "n_seq": n_seq,
    }
