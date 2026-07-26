"""Stage 1 (G1) — reproduce the paper's rank-stratification + temporal order-preservation on pure GDN.

SINGLE logging forward per (domain, sequence) produces ALL THREE Stage-1 analyses (efficiency, NOT
deferred judgement):
  [A] threshold-rank Rank_eff (Eq.6, eps=1e-4) stratification  -> bimodality of per-head rank
  [B] per-head rank-vector time-consistency Spearman rho(r_t1,r_t2) + norm consistency + cos sim
  [C] r_bar = exp(E_t log a_t) -> regression R^2 for EACH of the 3 rank metrics + theory-curve
      residual against min(d, e/(1-r_bar))
Plus the two internal-control diagnostics:
  [G1c] entropy eRank vs threshold-rank head-ranking Spearman (metric-artifact check)
  [Dom] cross-domain head-classification agreement (low/high rank label consistency across domains)

[PIN-2] every rank number is stored with metric name + implementation + cap d=min(dk,dv).
[PIN-4] pre-registered thresholds live in PREREGISTRATION.md and are echoed into the output JSON.
[PIN-7] Stage 2/3 (head masking/pruning, planted-MQAR SIR/oracle/MR, operator-composition C_t) are
        NOT started here — gated behind G1a. This file is Stage-1-only.

Run (VESSL A100):
  export GDN2_CKPT_PATH=/root/gdn2_1.3B_10B.pth TRITON_CACHE_DIR=/root/triton_cache HF_HUB_DISABLE_XET=1
  python stage1_repro.py --n-seq 16 --seq-len 2048 --out results/stage1

Smoke (CPU, no model — synthetic states):
  python stage1_repro.py --smoke
"""
import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from rank_metrics import all_ranks, EPS_THRESHOLD_RANK   # noqa: E402

# ------------------------------------------------------------------ pre-registered thresholds [PIN-4]
PREREG = {
    "spearman_time_consistency_min": 0.90,   # rho > 0.90 (paper: many layers)
    "norm_consistency_min": 0.98,            # paper App.
    "cos_sim_min": 0.97,                     # paper App.
    "theta_R_bimodality": 0.5,               # separation threshold for low/high classification
    "R2_pass": 0.7, "R2_weak": 0.3,          # G1b: r_bar -> rank regression
    "rank_corr_pass": 0.6, "rank_corr_strong": 0.8,   # G1c: eRank vs threshold-rank head ranking
    "MR_max": 0.2,                           # (Stage3 metric; pinned for consistency)
    "eps_threshold_rank": EPS_THRESHOLD_RANK,
    "config_name": "gdn2_1.3B",              # [PIN-1]
}


# ================================================================== analysis primitives
def _spearman(a, b):
    """Spearman rho without scipy (rank-transform + Pearson). Returns nan if degenerate."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def _r2(x, y):
    """R^2 of the OLS fit y ~ a*x + b."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 3 or np.all(x == x[0]):
        return float("nan"), (float("nan"), float("nan"))
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ coef
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return r2, (float(coef[0]), float(coef[1]))


def _bimodality_coefficient(x):
    """Sarle's bimodality coefficient BC = (g1^2 + 1) / (g2 + 3(n-1)^2/((n-2)(n-3))).

    g1 = sample skewness, g2 = sample EXCESS kurtosis (both with standard n-1 bias corrections).
    BC in (0,1]; uniform -> ~0.555; a perfect 2-point split -> ~1.0. Threshold 0.555 (Sarle/SAS).
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 4 or x.std(ddof=1) == 0:
        return float("nan")
    m = x.mean(); s = x.std(ddof=1)
    # bias-corrected sample skewness g1 and excess kurtosis g2 (Fisher-Pearson)
    m2 = ((x - m) ** 2).sum() / n
    m3 = ((x - m) ** 3).sum() / n
    m4 = ((x - m) ** 4).sum() / n
    g1 = m3 / (m2 ** 1.5 + 1e-12)
    g1 = g1 * np.sqrt(n * (n - 1)) / (n - 2)
    g2 = m4 / (m2 ** 2 + 1e-12) - 3.0
    g2 = ((n - 1) * ((n + 1) * g2 + 6.0)) / ((n - 2) * (n - 3))
    denom = g2 + (3.0 * (n - 1) ** 2) / ((n - 2) * (n - 3))
    return float((g1 ** 2 + 1.0) / denom) if denom > 0 else float("nan")


def _dip_gap(x, theta):
    """Cheap 2-mode separation score: fraction of mass beyond a normalized gap.

    Normalize threshold-rank to [0,1] by cap; a bimodal (low vs high) distribution has few heads
    in the middle band [theta-0.15, theta+0.15]. Returns (valley_fraction, low_frac, high_frac).
    Small valley_fraction + balanced low/high => stratified (bimodal)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    lo = float((x < theta - 0.15).mean())
    hi = float((x > theta + 0.15).mean())
    valley = float(((x >= theta - 0.15) & (x <= theta + 0.15)).mean())
    return valley, lo, hi


# ================================================================== per-domain Stage-1 pass
def analyze_domain(bundle, ids_list, seq_len, stride, layer_stride, meta):
    """Run [A],[B],[C],[G1c] for one domain. Returns a JSON-able dict.

    [A] final-state per-head threshold-rank across all (layer,head): stratification stats.
    [B] time consistency: for a few sequences, prefix-sweep t -> per-head rank vector; Spearman
        between the last two dense snapshots (+ norm/cos consistency).
    [C] r_bar per head (from decay probe) -> regression vs each of 3 rank metrics + theory residual.
    """
    n_layer = bundle.n_layer
    # ---- collect final-state ranks + r_bar over ALL sequences (variance via >=16 seqs) ----
    # rows: one per (seq, layer, head) with 3 ranks + cap + r_bar
    rows = []
    rbar_available = False
    for si, ids in enumerate(ids_list):
        states, rbar = bundle.states_and_rbar(ids)
        for li in sorted(states.keys()):
            S = states[li]                      # (heads, dk, dv), CPU float tensor
            H = S.shape[0]
            rb_layer = rbar.get(li)
            if rb_layer is not None:
                rbar_available = True
            for h in range(H):
                r = all_ranks(S[h])             # [PIN-2] metric name + cap included
                rb = float(rb_layer[h]) if (rb_layer is not None and h < len(rb_layer)) else float("nan")
                rows.append({"seq": si, "layer": li, "head": h,
                             "threshold_rank": r["threshold_rank"],
                             "entropy_erank": r["entropy_erank"],
                             "stable_rank": r["stable_rank"],
                             "cap_d": r["cap_d"], "r_bar": rb})

    tr = np.array([x["threshold_rank"] for x in rows], float)
    er = np.array([x["entropy_erank"] for x in rows], float)
    sr = np.array([x["stable_rank"] for x in rows], float)
    cap = np.array([x["cap_d"] for x in rows], float)
    rbar = np.array([x["r_bar"] for x in rows], float)
    tr_norm = tr / np.clip(cap, 1, None)        # normalized threshold-rank in [0,1] for bimodality

    # ---- [A] stratification / bimodality ----
    valley, low_frac, high_frac = _dip_gap(tr_norm, PREREG["theta_R_bimodality"])
    bc = _bimodality_coefficient(tr_norm)
    strat = {
        "metric": "threshold_rank (Eq.6, eps=%.0e)" % EPS_THRESHOLD_RANK,
        "cap_d_median": float(np.median(cap)),
        "threshold_rank_mean": float(np.nanmean(tr)),
        "threshold_rank_norm_mean": float(np.nanmean(tr_norm)),
        "bimodality_coefficient": bc,           # >0.555 => bimodal (Sarle)
        "valley_fraction": valley, "low_frac": low_frac, "high_frac": high_frac,
        "is_bimodal_BC": bool(np.isfinite(bc) and bc > 0.555),
        "n_heads_total": int(tr.size),
    }

    # ---- [G1c] head-ranking Spearman: entropy eRank vs threshold-rank (aggregated per head) ----
    # aggregate over sequences: mean rank per (layer,head)
    key = [(x["layer"], x["head"]) for x in rows]
    uniq = sorted(set(key))
    idx = {k: [] for k in uniq}
    for i, k in enumerate(key):
        idx[k].append(i)
    tr_head = np.array([tr[idx[k]].mean() for k in uniq])
    er_head = np.array([er[idx[k]].mean() for k in uniq])
    sr_head = np.array([sr[idx[k]].mean() for k in uniq])
    rbar_head = np.array([np.nanmean(rbar[idx[k]]) for k in uniq])
    g1c = {
        "spearman_erank_vs_threshold": _spearman(er_head, tr_head),
        "spearman_stable_vs_threshold": _spearman(sr_head, tr_head),
        "n_heads": len(uniq),
        "note": "eRank vs threshold-rank head ranking (metric-artifact check, G1c)",
    }

    # ---- [C] r_bar -> rank regressions + theory curve residual ----
    reg = {"r_bar_available": bool(rbar_available)}
    if rbar_available and np.isfinite(rbar_head).sum() >= 3:
        for name, arr in [("threshold_rank", tr_head), ("entropy_erank", er_head),
                          ("stable_rank", sr_head)]:
            r2, coef = _r2(rbar_head, arr)
            reg[name] = {"R2": r2, "slope": coef[0], "intercept": coef[1]}
        # theory curve: rank ~ min(d, e/(1-r_bar)); residual vs threshold-rank (per head)
        d = float(np.median(cap))
        rb = np.clip(rbar_head, 0, 0.999999)
        theory = np.minimum(d, np.e / (1.0 - rb))
        resid = tr_head - theory
        m = np.isfinite(resid)
        reg["theory_curve"] = {
            "form": "min(d, e/(1-r_bar))", "d": d,
            "residual_rmse": float(np.sqrt(np.nanmean(resid[m] ** 2))) if m.any() else float("nan"),
            "residual_mean": float(np.nanmean(resid[m])) if m.any() else float("nan"),
            "spearman_theory_vs_threshold": _spearman(theory, tr_head),
        }
    else:
        reg["note"] = "r_bar unavailable on this build -> regression skipped (flagged, not fabricated)"

    # ---- [B] time consistency (prefix sweep) on a subset of sequences ----
    n_tc = min(4, len(ids_list))
    tc_layers = {}
    for si in range(n_tc):
        grid, per_t = _rank_trajectory(bundle, ids_list[si], seq_len, stride, layer_stride)
        # compare last two dense snapshots t1 (earlier) vs t2 (final) per layer
        if len(grid) < 2:
            continue
        for li in per_t:
            vecs = per_t[li]              # dict t -> (heads,) threshold-rank vector
            ts = sorted(vecs.keys())
            v1, v2 = vecs[ts[-2]], vecs[ts[-1]]
            rho = _spearman(v1, v2)
            nc = _norm_consistency(v1, v2)
            cs = _cos_sim(v1, v2)
            tc_layers.setdefault(li, {"rho": [], "norm": [], "cos": []})
            tc_layers[li]["rho"].append(rho)
            tc_layers[li]["norm"].append(nc)
            tc_layers[li]["cos"].append(cs)
    tc_summary = {}
    rhos_all = []
    for li, d in tc_layers.items():
        rho_m = float(np.nanmean(d["rho"]))
        rhos_all.append(rho_m)
        tc_summary[str(li)] = {"spearman_rho": rho_m,
                               "norm_consistency": float(np.nanmean(d["norm"])),
                               "cos_sim": float(np.nanmean(d["cos"]))}
    time_consistency = {
        "per_layer": tc_summary,
        "n_layers": len(tc_summary),
        "frac_layers_rho_gt_0.90": float(np.mean([r > PREREG["spearman_time_consistency_min"]
                                                  for r in rhos_all])) if rhos_all else float("nan"),
        "median_rho": float(np.nanmedian(rhos_all)) if rhos_all else float("nan"),
        "note": "rho(r_t1,r_t2) on prefix-sweep snapshots; metric=threshold_rank per head",
    }

    return {
        "meta": meta,
        "stratification": strat,          # [A]
        "time_consistency": time_consistency,  # [B]
        "rbar_regression": reg,           # [C]
        "g1c_metric_agreement": g1c,      # [G1c]
        "_head_table": {"layer_head": [list(k) for k in uniq],
                        "threshold_rank": tr_head.tolist(),
                        "entropy_erank": er_head.tolist(),
                        "stable_rank": sr_head.tolist(),
                        "r_bar": rbar_head.tolist()},   # for cross-domain agreement + plots
    }


def _rank_trajectory(bundle, ids, seq_len, stride, layer_stride):
    """Prefix-sweep t -> per-layer per-head threshold-rank vector (mirrors analysis.state_trajectory)."""
    T = ids.shape[1]
    grid = list(range(max(stride, 8), T + 1, stride))
    if not grid or grid[-1] != T:
        grid.append(T)
    per_t = {}   # layer -> {t: (heads,) threshold-rank}
    for t in grid:
        states = bundle.states(ids[:, :t])
        layers = sorted(states.keys())[::layer_stride]
        for li in layers:
            S = states[li]
            vec = np.array([all_ranks(S[h])["threshold_rank"] for h in range(S.shape[0])], float)
            per_t.setdefault(li, {})[t] = vec
    return grid, per_t


def _norm_consistency(v1, v2):
    v1 = np.asarray(v1, float); v2 = np.asarray(v2, float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return float("nan")
    return float(min(n1, n2) / max(n1, n2))


def _cos_sim(v1, v2):
    v1 = np.asarray(v1, float); v2 = np.asarray(v2, float)
    d = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(np.dot(v1, v2) / d) if d > 0 else float("nan")


# ================================================================== cross-domain agreement
def cross_domain_agreement(per_domain, natural_domains, theta=0.5):
    """[Dom] head-classification (low/high threshold-rank) agreement across natural domains.

    For each domain, label each head low(0)/high(1) by normalized threshold-rank vs theta. Report
    pairwise label-agreement fraction across natural domains (App.D attacks excluded from this metric
    since they are adversarial, not natural-domain generalization)."""
    labels = {}
    caps = {}
    for dom in natural_domains:
        ht = per_domain[dom]["_head_table"]
        lh = [tuple(x) for x in ht["layer_head"]]
        tr = np.array(ht["threshold_rank"], float)
        # cap per domain from stratification median
        cap = per_domain[dom]["stratification"]["cap_d_median"]
        labels[dom] = {k: int((tr[i] / cap) > theta) for i, k in enumerate(lh)}
        caps[dom] = cap
    doms = list(labels.keys())
    pairs = {}
    for i in range(len(doms)):
        for j in range(i + 1, len(doms)):
            a, b = labels[doms[i]], labels[doms[j]]
            common = set(a) & set(b)
            agree = np.mean([a[k] == b[k] for k in common]) if common else float("nan")
            pairs[f"{doms[i]}|{doms[j]}"] = float(agree)
    return {"pairwise_agreement": pairs,
            "mean_agreement": float(np.nanmean(list(pairs.values()))) if pairs else float("nan"),
            "theta": theta, "domains": doms}


# ================================================================== verdict
def verdict(report):
    """Apply pre-registered thresholds [PIN-4] -> G1a/G1b/G1c booleans + overall gate."""
    doms = report["domains"]
    natural = [d for d in doms if d in ("wikitext", "github", "arxiv")]
    # G1a: bimodal stratification + time consistency
    bimodal = np.mean([report["per_domain"][d]["stratification"]["is_bimodal_BC"] for d in natural])
    rho_frac = np.nanmean([report["per_domain"][d]["time_consistency"]["frac_layers_rho_gt_0.90"]
                           for d in natural])
    g1a = bool(bimodal >= 0.5 and rho_frac >= 0.5)
    # G1b: r_bar regression R^2
    r2s = []
    for d in natural:
        reg = report["per_domain"][d]["rbar_regression"]
        if reg.get("r_bar_available") and "threshold_rank" in reg:
            r2s.append(reg["threshold_rank"]["R2"])
    r2_med = float(np.nanmedian(r2s)) if r2s else float("nan")
    g1b = "pass" if (np.isfinite(r2_med) and r2_med >= PREREG["R2_pass"]) else \
          ("weak" if (np.isfinite(r2_med) and r2_med >= PREREG["R2_weak"]) else
           ("fail" if np.isfinite(r2_med) else "unavailable"))
    # G1c: eRank vs threshold-rank head ranking
    corrs = [report["per_domain"][d]["g1c_metric_agreement"]["spearman_erank_vs_threshold"]
             for d in natural]
    corr_med = float(np.nanmedian(corrs)) if corrs else float("nan")
    g1c = "strong" if (np.isfinite(corr_med) and corr_med >= PREREG["rank_corr_strong"]) else \
          ("pass" if (np.isfinite(corr_med) and corr_med >= PREREG["rank_corr_pass"]) else "fail")
    return {
        "G1a_stratification_and_time": g1a,
        "G1a_detail": {"frac_domains_bimodal": float(bimodal), "mean_frac_layers_rho>0.90": float(rho_frac)},
        "G1b_rbar_regression": g1b, "G1b_R2_median": r2_med,
        "G1c_metric_agreement": g1c, "G1c_spearman_median": corr_med,
        "cross_domain_head_agreement": report.get("cross_domain", {}).get("mean_agreement"),
        "next": ("G1a=YES -> Stage 2 may start; paper access permitted."
                 if g1a else
                 "G1a=NO (unimodal) -> record limitation (hybrid/scale/post-training dependence); "
                 "do NOT train hybrid; Stage 2 stays gated [PIN-7]."),
    }


# ================================================================== main
def run(bundle, tok, args):
    import data_stage1
    which = args.domains.split(",") if args.domains else list(data_stage1.ALL.keys())
    data = data_stage1.load_all(tok, seq_len=args.seq_len, n_seq=args.n_seq, which=which)
    report = {"config": PREREG, "domains": list(data.keys()), "per_domain": {},
              "args": vars(args), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    for dom, (ids_list, meta) in data.items():
        t0 = time.time()
        print(f"\n=== domain {dom} ({meta['source']}) ===", flush=True)
        report["per_domain"][dom] = analyze_domain(
            bundle, ids_list, args.seq_len, args.stride, args.layer_stride, meta)
        report["per_domain"][dom]["minutes"] = round((time.time() - t0) / 60, 2)
        s = report["per_domain"][dom]["stratification"]
        print(f"  [A] threshold-rank norm mean={s['threshold_rank_norm_mean']:.3f} "
              f"BC={s['bimodality_coefficient']:.3f} bimodal={s['is_bimodal_BC']}", flush=True)
    natural = [d for d in report["domains"] if d in ("wikitext", "github", "arxiv")]
    if len(natural) >= 2:
        report["cross_domain"] = cross_domain_agreement(report["per_domain"], natural,
                                                        theta=PREREG["theta_R_bimodality"])
    report["verdict"] = verdict(report)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(_HERE, "results", "stage1"))
    ap.add_argument("--ckpt", default=os.environ.get("GDN2_CKPT_PATH", "/root/gdn2_1.3B_10B.pth"),
                    help="[PIN-1] checkpoint-10B path")
    ap.add_argument("--n-seq", type=int, default=16, dest="n_seq")   # >=16 [PIN-5]
    ap.add_argument("--seq-len", type=int, default=2048, dest="seq_len")
    ap.add_argument("--stride", type=int, default=128, help="prefix-sweep step for time consistency")
    ap.add_argument("--layer-stride", type=int, default=1, dest="layer_stride",
                    help="1 = ALL gdn2 layers (recommended for repro judgement)")
    ap.add_argument("--domains", default=None, help="comma list; default all 5")
    ap.add_argument("--smoke", action="store_true", help="CPU synthetic-state smoke (no model)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.smoke:
        return _smoke(args)

    import loader_gdn2
    print(f"[load] gdn2-1.3B checkpoint={args.ckpt} config={loader_gdn2.CONFIG_NAME} [PIN-1]", flush=True)
    bundle = loader_gdn2.load(checkpoint_path=args.ckpt)
    from transformers import AutoTokenizer
    import common as gdn2_common
    tok = AutoTokenizer.from_pretrained(gdn2_common.TOKENIZER)

    report = run(bundle, tok, args)
    report["ckpt_provenance"] = getattr(bundle, "ckpt_provenance", None)   # [PIN-1] resolved path/size/hash
    out = os.path.join(args.out, "stage1_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[verdict] {json.dumps(report['verdict'], indent=2)}")
    print(f"[written] {out}")
    sys.stdout.flush(); sys.stderr.flush()
    # HF `datasets` streaming leaves a prefetch thread that can trigger a benign GIL-teardown crash
    # (non-zero exit) during interpreter finalization in a torch process. All results are already
    # written above, so hard-exit to guarantee a clean 0 exit code for the orchestrator.
    os._exit(0)


# ------------------------------------------------------------------ CPU smoke (no GPU, no model)
class _FakeBundle:
    """Synthetic bundle: planted bimodal per-head rank + decaying prefix ranks + r_bar, so every
    analysis path executes on CPU in seconds. Confirms wiring/thresholds/JSON, NOT the model."""
    def __init__(self, n_layer=6, heads=8, d=64, seed=0):
        self.n_layer = n_layer
        self.heads = heads; self.d = d
        self.rng = np.random.default_rng(seed)
        # planted per-(layer,head) target rank: half low (~3), half high (~d)
        self.target = {}
        for li in range(n_layer):
            for h in range(heads):
                self.target[(li, h)] = 3 if (h % 2 == 0) else d
        self.rbar = {(li, h): (0.2 if (h % 2 == 0) else 0.98) for li in range(n_layer) for h in range(heads)}

    def _state_at(self, li, h, t):
        import torch
        r = min(self.target[(li, h)], max(1, t // 4))   # rank grows with prefix length -> order-preserving
        U = self.rng.standard_normal((self.d, r)); V = self.rng.standard_normal((r, self.d))
        return torch.from_numpy((U @ V).astype("float32"))

    def states(self, ids):
        import torch
        t = ids.shape[1]
        return {li: torch.stack([self._state_at(li, h, t) for h in range(self.heads)])
                for li in range(self.n_layer)}

    def states_and_rbar(self, ids):
        st = self.states(ids)
        rb = {li: np.array([self.rbar[(li, h)] for h in range(self.heads)]) for li in range(self.n_layer)}
        return st, rb


def _smoke(args):
    import torch
    bundle = _FakeBundle()
    ids_list = [torch.zeros(1, 256, dtype=torch.long) for _ in range(4)]
    data = {d: (ids_list, {"domain": d, "source": "smoke", "seq_len": 256, "n_seq": 4})
            for d in ["wikitext", "github", "arxiv", "repeat_rarechar"]}
    report = {"config": PREREG, "domains": list(data.keys()), "per_domain": {}, "args": vars(args),
              "timestamp": "smoke"}
    for dom, (ids, meta) in data.items():
        report["per_domain"][dom] = analyze_domain(bundle, ids, 256, stride=64, layer_stride=1, meta=meta)
    natural = ["wikitext", "github", "arxiv"]
    report["cross_domain"] = cross_domain_agreement(report["per_domain"], natural,
                                                    theta=PREREG["theta_R_bimodality"])
    report["verdict"] = verdict(report)
    out = os.path.join(args.out, "stage1_report_smoke.json")
    os.makedirs(args.out, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print("[smoke] verdict:", json.dumps(report["verdict"], indent=2))
    print("[smoke] wrote", out)
    # sanity: planted data should read as bimodal
    assert report["per_domain"]["wikitext"]["stratification"]["is_bimodal_BC"] or \
        report["per_domain"]["wikitext"]["stratification"]["low_frac"] > 0.3, "smoke: expected stratification"
    print("[smoke] OK")


if __name__ == "__main__":
    main()
