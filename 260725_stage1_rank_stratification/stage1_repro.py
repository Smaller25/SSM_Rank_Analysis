"""Stage 1 (G1) — reproduce the paper's rank-stratification + temporal order-preservation on pure GDN.

SINGLE logging forward per (domain, sequence) produces ALL THREE Stage-1 analyses (efficiency, NOT
deferred judgement):
  [A] threshold-rank Rank_eff (Eq.6, eps=1e-4) stratification  -> bimodality of per-head rank
  [B] per-head time-consistency measured in the GROWTH regime (prefix pairs with t < d = min(dk,dv),
      where rank is still rising, so temporal order-preservation is testable) — NOT the saturated
      tail. TWO SEPARATE axes: (i) Spearman rho on the per-head RANK vector; (ii) norm-cosine on the
      per-head NUCLEAR-NORM vector (magnitude order-preservation, paper Thm 4.4 / Eq.13). Saturated
      prefixes (all-heads-equal -> Spearman NaN) are COUNTED and reported as a separate regime, never
      silently averaged away.
  [C] r_bar = exp(E_t log a_t) -> regression R^2 for EACH of the 3 rank metrics + theory-curve
      residual against the PAPER upper bound min(t, d) (Thm 3.1, rank(S(t)) <= min(t, d)). The old
      min(d, e/(1-r_bar)) curve was a local nb3 decay heuristic that is NOT in arXiv:2602.02195; it
      is retained ONLY as an explicitly-labelled auxiliary diagnostic.
Plus the two internal-control diagnostics:
  [G1c] entropy eRank vs threshold-rank head-ranking Spearman (metric-artifact check)
  [Dom] cross-domain head-classification agreement (low/high rank label consistency across domains)

[PIN-2] every rank number is stored with metric name + implementation + cap d=min(dk,dv).
[PIN-4] pre-registered thresholds live in PREREGISTRATION.md and are echoed into the output JSON.
        These thresholds (rho>0.90, norm-cos>0.98, cos>0.97, ...) are Qwen3-Next(48-layer, large
        post-trained) OBSERVED values adopted as reproduction TARGET lines for pure GDN (18 layers,
        100B-base) — a target, not a pass standard (see PREREGISTRATION.md).
[PIN-6] seed>=3 fixed (torch / numpy / PYTHONHASHSEED) via set_seed() before any data is drawn.
[PIN-7] Stage 2/3 (head masking/pruning, planted-MQAR SIR/oracle/MR, operator-composition C_t) are
        NOT started here — gated behind G1a. This file is Stage-1-only.

Data authenticity gate (credibility-fix 4): if any NATURAL domain (wikitext/github/arxiv) fell back
to the synthetic corpus (meta.source=='fallback'), G1a is forced INVALID (None) and routing halts;
--require-real-data turns that into an immediate hard failure.

Run (greenbeard SLURM, 100B checkpoint):
  export GDN2_CKPT_PATH=/home/sohyung/models/gdn2_1.3B_100b.pth   # [PIN-1] 100B paper-matched
  export TRITON_CACHE_DIR=/home/sohyung/.triton_cache HF_HUB_DISABLE_XET=1
  python stage1_repro.py --n-seq 16 --seq-len 2048 --layer-stride 1 --stride 16 --seed 0 \
         --require-real-data --out results/stage1

Smoke (CPU, no model — synthetic states):
  python stage1_repro.py --smoke
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
# [infra] vendored fla (flash-linear-attention 0.5.2) — mirror rank_decomp_100b.py so the gdn2 kernel
# + fla Cache resolve identically here (loader only wires common._find_lit; fla itself lives here).
_FLA_PATH = "/home/sohyung/linear-memory-routing"
if os.path.isdir(_FLA_PATH) and _FLA_PATH not in sys.path:
    sys.path.insert(0, _FLA_PATH)

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
    "threshold_provenance": ("Qwen3-Next 48-layer large post-trained OBSERVED values, adopted as "
                             "reproduction TARGET lines for pure GDN (18 layers, 100B-base) — "
                             "targets, not pass standards [PIN-4]."),
}


def set_seed(seed):
    """[PIN-6] pin torch / numpy / python-hash / cuda RNG so the whole pipeline is reproducible."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


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
    [B] time consistency in the GROWTH regime (t < d): Spearman rho on the per-head RANK vector +
        norm-cosine on the per-head NUCLEAR-NORM vector, over adjacent growth prefix pairs; saturated
        pairs counted separately (fix-1, fix-3).
    [C] r_bar per head (from decay probe) -> regression vs each of 3 rank metrics + PAPER bound
        min(t,d) residual (Thm 3.1); local decay heuristic retained as labelled aux (fix-2).
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

    # ---- [C] r_bar -> rank regressions + PAPER theory-bound residual (fix-2) ----
    reg = {"r_bar_available": bool(rbar_available)}
    d = float(np.median(cap))
    seqlen = int(meta.get("seq_len", seq_len))
    # Paper bound Thm 3.1: rank(S(t)) <= min(t, d). At the FINAL state t = seq_len >> d the binding
    # term is d, so the paper-grounded final-state residual is (threshold_rank - d): how far the
    # observed rank sits below its algebraic ceiling. (The min(t,d) SLOPE vs t is measured in [B]'s
    # growth trajectory, where t < d makes it non-trivial.)
    paper_bound_final = float(min(seqlen, d))
    tr_resid_bound = tr_head - paper_bound_final
    mb = np.isfinite(tr_resid_bound)
    reg["paper_theory_bound"] = {
        "form": "min(t, d)  (Thm 3.1, rank(S(t)) <= min(t,d))",
        "source": "arXiv:2602.02195 Thm 3.1",
        "t_final": seqlen, "d": d, "bound_final": paper_bound_final,
        "residual_below_bound_mean": float(np.nanmean(tr_resid_bound[mb])) if mb.any() else float("nan"),
        "residual_below_bound_rmse": float(np.sqrt(np.nanmean(tr_resid_bound[mb] ** 2))) if mb.any() else float("nan"),
        "frac_heads_at_or_below_bound": float(np.mean(tr_head[mb] <= paper_bound_final + 1e-9)) if mb.any() else float("nan"),
    }
    if rbar_available and np.isfinite(rbar_head).sum() >= 3:
        for name, arr in [("threshold_rank", tr_head), ("entropy_erank", er_head),
                          ("stable_rank", sr_head)]:
            r2, coef = _r2(rbar_head, arr)
            reg[name] = {"R2": r2, "slope": coef[0], "intercept": coef[1]}
        # AUX (NOT from the paper): local nb3 decay heuristic rank ~ min(d, e/(1-r_bar)). Retained,
        # explicitly labelled, so the r_bar->rank monotone relation is still visible; it is NOT used
        # for any gate (G1b uses the R^2 above + the paper bound residual).
        rb = np.clip(rbar_head, 0, 0.999999)
        heur = np.minimum(d, np.e / (1.0 - rb))
        resid = tr_head - heur
        m = np.isfinite(resid)
        reg["aux_decay_heuristic"] = {
            "form": "min(d, e/(1-r_bar))",
            "label": "nb3_local_decay_heuristic (NOT from arXiv:2602.02195)",
            "d": d,
            "residual_rmse": float(np.sqrt(np.nanmean(resid[m] ** 2))) if m.any() else float("nan"),
            "residual_mean": float(np.nanmean(resid[m])) if m.any() else float("nan"),
            "spearman_heuristic_vs_threshold": _spearman(heur, tr_head),
        }
    else:
        reg["note"] = "r_bar unavailable on this build -> regression skipped (flagged, not fabricated)"

    # ---- [B] time consistency in the GROWTH regime (t < d), two separate axes (fix-1, fix-3) ----
    # For each layer we sweep prefix lengths t with a DENSE grid inside the growth regime t < d
    # (rank still rising -> order-preservation is testable). We evaluate two axes on adjacent prefix
    # pairs (t_a < t_b), and classify each pair as GROWTH (rank vector non-degenerate -> Spearman
    # finite) or SATURATED (all-heads-equal or a repeat of the previous vector -> Spearman NaN). The
    # saturated pairs are COUNTED (not silently averaged away).
    #   axis (i)  RANK           -> Spearman rho(rank_ta, rank_tb)          (order-preservation)
    #   axis (ii) NUCLEAR-NORM   -> norm-cosine( nuc_ta, nuc_tb )           (magnitude order-pres.)
    n_tc = min(4, len(ids_list))
    d_cap = int(np.median(cap)) if cap.size else 128
    tc_layers = {}    # li -> adjacent-growth (SECONDARY, diagnostic) accumulators
    sep_layers = {}   # li -> separated-pair (PRIMARY, paper §3.2) accumulators
    for si in range(n_tc):
        per_t = _rank_trajectory(bundle, ids_list[si], seq_len, stride, layer_stride, d_cap)
        for li, vecs in per_t.items():        # vecs: {t: {"rank":(H,), "nuc":(H,)}}
            ts = sorted(vecs.keys())
            if len(ts) < 2:
                continue
            # PRIMARY [fix-1, paper §3.2]: separated pairs = early anchor (largest t<=d, rank just
            # established) vs each late point (t>d in {256,512,1024,seq_len}). This IS the paper's
            # t=128-vs-t=2048 order-preservation; it spans growth->saturation and is NOT trivially high.
            growth_ts = [t for t in ts if t <= d_cap]
            anchor = growth_ts[-1] if growth_ts else ts[0]
            late_ts = [t for t in ts if t > d_cap]
            sl = sep_layers.setdefault(li, {"rho": [], "normcos": [], "cos": [], "n": 0, "anchor": anchor})
            for tb in late_ts:
                ra, rb = vecs[anchor]["rank"], vecs[tb]["rank"]
                na, nb = vecs[anchor]["nuc"], vecs[tb]["nuc"]
                rho = _spearman(ra, rb)
                if not np.isfinite(rho):
                    continue
                sl["rho"].append(rho); sl["normcos"].append(_norm_cosine(na, nb))
                sl["cos"].append(_cos_sim(ra, rb)); sl["n"] += 1
            # SECONDARY: adjacent growth pairs (diagnostic; degenerate/saturated pairs counted).
            slot = tc_layers.setdefault(li, {"rho_growth": [], "normcos_growth": [], "cos_growth": [],
                                             "n_growth": 0, "n_sat": 0})
            for ta, tb in zip(ts[:-1], ts[1:]):
                rk_a, rk_b = vecs[ta]["rank"], vecs[tb]["rank"]
                rho = _spearman(rk_a, rk_b)
                if not np.isfinite(rho):
                    slot["n_sat"] += 1
                    continue
                slot["n_growth"] += 1
                slot["rho_growth"].append(rho)
                slot["normcos_growth"].append(_norm_cosine(vecs[ta]["nuc"], vecs[tb]["nuc"]))
                slot["cos_growth"].append(_cos_sim(rk_a, rk_b))
    tc_summary = {}
    rho_sep_all, normcos_sep_all = [], []    # PRIMARY separated
    rhos_all, normcos_all = [], []           # SECONDARY growth
    tot_growth = tot_sat = tot_sep = 0
    for li in sorted(set(list(sep_layers.keys()) + list(tc_layers.keys()))):
        sp = sep_layers.get(li, {"rho": [], "normcos": [], "cos": [], "n": 0, "anchor": None})
        gr = tc_layers.get(li, {"rho_growth": [], "normcos_growth": [], "cos_growth": [], "n_growth": 0, "n_sat": 0})
        tot_sep += sp["n"]; tot_growth += gr["n_growth"]; tot_sat += gr["n_sat"]
        rho_sep_m = float(np.nanmedian(sp["rho"])) if sp["rho"] else float("nan")
        ncos_sep_m = float(np.nanmedian(sp["normcos"])) if sp["normcos"] else float("nan")
        if np.isfinite(rho_sep_m):
            rho_sep_all.append(rho_sep_m); normcos_sep_all.append(ncos_sep_m)
        rho_g_m = float(np.nanmedian(gr["rho_growth"])) if gr["rho_growth"] else float("nan")
        ncos_g_m = float(np.nanmedian(gr["normcos_growth"])) if gr["normcos_growth"] else float("nan")
        if np.isfinite(rho_g_m):
            rhos_all.append(rho_g_m); normcos_all.append(ncos_g_m)
        tc_summary[str(li)] = {
            # PRIMARY separated (paper): anchor(t<=d) vs late saturated points
            "spearman_rho_separated": rho_sep_m,
            "spearman_rho_separated_min": float(np.nanmin(sp["rho"])) if sp["rho"] else float("nan"),
            "norm_cosine_separated": ncos_sep_m,
            "anchor_t": sp["anchor"], "n_separated_pairs": sp["n"],
            # SECONDARY growth-adjacent (diagnostic)
            "spearman_rho_growth": rho_g_m, "norm_cosine_growth": ncos_g_m,
            "cos_sim_growth": float(np.nanmedian(gr["cos_growth"])) if gr["cos_growth"] else float("nan"),
            "n_growth_pairs": gr["n_growth"], "n_saturated_pairs": gr["n_sat"],
        }
    time_consistency = {
        "regime": "PRIMARY=separated pairs anchor(t<=d) vs late(256/512/1024/seq_len) per paper §3.2; "
                  "SECONDARY=adjacent growth pairs (diagnostic)",
        "d_cap": d_cap,
        "per_layer": tc_summary,
        "n_layers": len(tc_summary),
        # PRIMARY (paper axis) — the verdict gates on this
        "n_layers_with_separated": len(rho_sep_all),
        "frac_layers_rho_sep_gt_0.90": float(np.mean([r > PREREG["spearman_time_consistency_min"]
                                                      for r in rho_sep_all])) if rho_sep_all else float("nan"),
        "median_rho_separated": float(np.nanmedian(rho_sep_all)) if rho_sep_all else float("nan"),
        "min_rho_separated": float(np.nanmin(rho_sep_all)) if rho_sep_all else float("nan"),
        "median_norm_cosine_separated": float(np.nanmedian(normcos_sep_all)) if normcos_sep_all else float("nan"),
        "frac_layers_norm_cos_sep_gt_0.98": float(np.mean([n > PREREG["norm_consistency_min"]
                                                           for n in normcos_sep_all])) if normcos_sep_all else float("nan"),
        "n_separated_pairs_total": int(tot_sep),
        # SECONDARY (growth diagnostic) — reported, NOT gated
        "n_layers_with_growth": len(rhos_all),
        "frac_layers_rho_gt_0.90": float(np.mean([r > PREREG["spearman_time_consistency_min"]
                                                  for r in rhos_all])) if rhos_all else float("nan"),
        "median_rho_growth": float(np.nanmedian(rhos_all)) if rhos_all else float("nan"),
        "median_norm_cosine_growth": float(np.nanmedian(normcos_all)) if normcos_all else float("nan"),
        "n_growth_pairs_total": int(tot_growth), "n_saturated_pairs_total": int(tot_sat),
        "saturation_fraction": (float(tot_sat / (tot_growth + tot_sat)) if (tot_growth + tot_sat) else float("nan")),
        "note": ("PRIMARY: Spearman rho(rank_anchor, rank_late) on separated pairs (paper t=128 vs "
                 "t=2048, order-preservation across growth->saturation) + nuclear-norm cosine (Thm4.4). "
                 "SECONDARY: adjacent growth-pair rho (diagnostic; degenerate/saturated pairs counted)."),
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


def _growth_grid(T, stride, d_cap):
    """Dense prefix grid inside the GROWTH regime t < d (fix-1).

    The rank of a GDN head state can only rise while t < d = min(dk,dv); past that it saturates and
    consecutive rank vectors become identical (Spearman NaN). So temporal order-preservation must be
    read off pairs with t < d. We build a dense grid from a small start up to min(T, d) with the
    requested `stride` (default 16 covers t<128 densely), and append one saturated snapshot (T) so
    the driver can still COUNT the saturation transition."""
    top = min(T, max(d_cap, 8))
    step = max(1, min(stride, 16))
    grid = list(range(step, top + 1, step))
    if not grid or grid[-1] != top:
        grid.append(top)
    # fix-1 (paper §3.2): the paper compares an early anchor (t≈d) with WIDELY-SEPARATED late points
    # (t=256/512/1024/2048) spanning growth->saturation. Add those late snapshots so the driver can
    # form the paper's separated pairs (PRIMARY time-consistency), not just adjacent growth pairs.
    for late in (256, 512, 1024, T):
        if d_cap < late <= T:
            grid.append(late)
    return sorted(set(grid))


def _rank_trajectory(bundle, ids, seq_len, stride, layer_stride, d_cap):
    """Prefix-sweep t (GROWTH regime) -> per-layer {t: {"rank":(H,), "nuc":(H,)}}.

    Two per-head vectors per snapshot: threshold-rank (RANK axis) and nuclear norm (NORM axis).
    Only prefixes t <= min(T,d) are dense; one saturated snapshot at T is added for counting."""
    T = ids.shape[1]
    grid = _growth_grid(T, stride, d_cap)
    per_t = {}   # layer -> {t: {"rank":(H,), "nuc":(H,)}}
    for t in grid:
        states = bundle.states(ids[:, :t])
        layers = sorted(states.keys())[::layer_stride]
        for li in layers:
            S = states[li]
            H = S.shape[0]
            rank_vec = np.empty(H, float); nuc_vec = np.empty(H, float)
            for h in range(H):
                r = all_ranks(S[h])
                rank_vec[h] = r["threshold_rank"]
                nuc_vec[h] = r["nuclear_norm"]
            per_t.setdefault(li, {})[t] = {"rank": rank_vec, "nuc": nuc_vec}
    return per_t


def _norm_cosine(v1, v2):
    """Norm-cosine on per-head NUCLEAR-NORM vectors (fix-3): cos(nuc_t1, nuc_t2).

    This is the paper's magnitude order-preservation axis (Thm 4.4 / Eq.13), computed on the vector
    of per-head nuclear norms, NOT on the rank vector. Returns nan if either vector is all-zero."""
    v1 = np.asarray(v1, float); v2 = np.asarray(v2, float)
    d = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(np.dot(v1, v2) / d) if d > 0 else float("nan")


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
def _fallback_natural_domains(report, natural):
    """fix-4 data-authenticity gate: natural domains that fell back to the synthetic corpus."""
    bad = []
    for d in natural:
        meta = report["per_domain"][d].get("meta", {})
        if meta.get("is_fallback") or meta.get("source") == "fallback":
            bad.append(d)
    return bad


def verdict(report):
    """Apply pre-registered thresholds [PIN-4] -> G1a/G1b/G1c booleans + overall gate.

    fix-4: if ANY natural domain (wikitext/github/arxiv) used the synthetic fallback corpus, G1a is
    forced INVALID (None) and routing halts — a fallback corpus is never allowed to drive the verdict.
    """
    doms = report["domains"]
    natural = [d for d in doms if d in ("wikitext", "github", "arxiv")]
    fb = _fallback_natural_domains(report, natural)
    # G1a: bimodal stratification + time consistency. PRIMARY axis [fix-1] = SEPARATED pairs
    # (paper §3.2: early anchor t≈d vs late t=256/512/1024/2048 order-preservation). The growth-adjacent
    # rho is a SECONDARY diagnostic and is reported but NOT gated.
    bimodal = np.mean([report["per_domain"][d]["stratification"]["is_bimodal_BC"] for d in natural])
    rho_frac = np.nanmean([report["per_domain"][d]["time_consistency"]
                           .get("frac_layers_rho_sep_gt_0.90", float("nan")) for d in natural])   # PRIMARY
    rho_frac_growth = np.nanmean([report["per_domain"][d]["time_consistency"]
                                  .get("frac_layers_rho_gt_0.90", float("nan")) for d in natural])  # secondary
    normcos_frac = np.nanmean([report["per_domain"][d]["time_consistency"]
                               .get("frac_layers_norm_cos_sep_gt_0.98", float("nan")) for d in natural])
    sat_frac = np.nanmean([report["per_domain"][d]["time_consistency"]
                           .get("saturation_fraction", float("nan")) for d in natural])
    if fb:
        g1a = None   # INVALID — fallback synthetic data cannot ground a reproduction verdict
    else:
        g1a = bool(bimodal >= 0.5 and np.isfinite(rho_frac) and rho_frac >= 0.5)
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
    if g1a is None:
        nxt = (f"G1a=INVALID -> natural domains used the synthetic fallback corpus ({fb}); verdict "
               "cannot be trusted. Re-run with real data (--require-real-data) [fix-4]. Stage 2 stays "
               "gated [PIN-7].")
    elif g1a:
        nxt = "G1a=YES -> Stage 2 may start; paper access permitted."
    else:
        nxt = ("G1a=NO (unimodal) -> record limitation (hybrid/scale/post-training dependence); "
               "do NOT train hybrid; Stage 2 stays gated [PIN-7].")
    return {
        "G1a_stratification_and_time": g1a,   # True / False / None(=INVALID fallback)
        "G1a_valid": g1a is not None,
        "G1a_fallback_natural_domains": fb,
        "G1a_detail": {
            "frac_domains_bimodal": float(bimodal),
            "mean_frac_layers_rho_sep>0.90 (PRIMARY: separated pairs, rank)": float(rho_frac),
            "mean_frac_layers_rho>0.90 (SECONDARY: growth-adjacent, rank)": float(rho_frac_growth),
            "mean_frac_layers_norm_cos_sep>0.98 (separated, nuclear-norm)": float(normcos_frac),
            "mean_saturation_fraction": float(sat_frac),
        },
        "G1b_rbar_regression": g1b, "G1b_R2_median": r2_med,
        "G1c_metric_agreement": g1c, "G1c_spearman_median": corr_med,
        "cross_domain_head_agreement": report.get("cross_domain", {}).get("mean_agreement"),
        "next": nxt,
    }


# ================================================================== main
def run(bundle, tok, args):
    import data_stage1
    data_stage1.set_data_seed(args.seed)   # [PIN-6] seed-dependent sampling (real multi-seed variance)
    which = args.domains.split(",") if args.domains else list(data_stage1.ALL.keys())
    data = data_stage1.load_all(tok, seq_len=args.seq_len, n_seq=args.n_seq, which=which)

    # fix-4: if a NATURAL domain fell back to synthetic data, either hard-fail (--require-real-data)
    # or continue but let verdict() force G1a=INVALID. Attacks (appD_reconstruct) are never fallback.
    fb = [d for d, (_, m) in data.items()
          if d in ("wikitext", "github", "arxiv") and (m.get("is_fallback") or m.get("source") == "fallback")]
    if fb:
        msg = (f"[fix-4] natural domains fell back to SYNTHETIC data: {fb}. Real HF corpora are "
               f"required for a reproduction verdict.")
        if getattr(args, "require_real_data", False):
            raise RuntimeError(msg + " (--require-real-data set -> aborting).")
        print("  [WARN] " + msg + " Verdict will be forced G1a=INVALID.", flush=True)

    report = {"config": PREREG, "domains": list(data.keys()), "per_domain": {},
              "args": vars(args), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    partial_dir = getattr(args, "out", None)
    for dom, (ids_list, meta) in data.items():
        # resume: skip a domain whose partial_<dom>.json already exists (incremental save)
        pj = os.path.join(partial_dir, f"partial_{dom}.json") if partial_dir else None
        if pj and getattr(args, "resume", False) and os.path.isfile(pj):
            try:
                with open(pj) as f:
                    report["per_domain"][dom] = json.load(f)
                print(f"\n=== domain {dom}: RESUMED from {pj} ===", flush=True)
                continue
            except Exception as e:
                print(f"  [resume] {pj} unreadable ({type(e).__name__}) -> recompute", flush=True)
        t0 = time.time()
        print(f"\n=== domain {dom} ({meta['source']}) ===", flush=True)
        res = analyze_domain(bundle, ids_list, args.seq_len, args.stride, args.layer_stride, meta)
        res["minutes"] = round((time.time() - t0) / 60, 2)
        report["per_domain"][dom] = res
        s = res["stratification"]
        print(f"  [A] threshold-rank norm mean={s['threshold_rank_norm_mean']:.3f} "
              f"BC={s['bimodality_coefficient']:.3f} bimodal={s['is_bimodal_BC']}", flush=True)
        tc = res["time_consistency"]
        print(f"  [B] median rho(growth,rank)={tc['median_rho_growth']} "
              f"median norm-cos(growth,nuc)={tc['median_norm_cosine_growth']} "
              f"sat_frac={tc['saturation_fraction']}", flush=True)
        if pj:                                   # incremental flush per completed domain
            with open(pj, "w") as f:
                json.dump(res, f, indent=2)
            print(f"  [partial] wrote {pj}", flush=True)
    natural = [d for d in report["domains"] if d in ("wikitext", "github", "arxiv")]
    if len(natural) >= 2:
        report["cross_domain"] = cross_domain_agreement(report["per_domain"], natural,
                                                        theta=PREREG["theta_R_bimodality"])
    report["verdict"] = verdict(report)
    return report


def _git_head():
    try:
        import subprocess
        return subprocess.check_output(["git", "-C", _HERE, "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"


def _runtime_versions():
    v = {"python": sys.version.split()[0]}
    for mod in ("torch", "numpy", "transformers", "datasets", "fla"):
        try:
            v[mod] = __import__(mod).__version__
        except Exception:
            v[mod] = "n/a"
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(_HERE, "results", "stage1"))
    ap.add_argument("--ckpt",
                    default=os.environ.get("GDN2_CKPT_PATH", "/home/sohyung/models/gdn2_1.3B_100b.pth"),
                    help="[PIN-1] paper-matched 100B checkpoint (95B accepted; 10B REJECTED by loader)")
    ap.add_argument("--n-seq", type=int, default=16, dest="n_seq")   # >=16 [PIN-5]
    ap.add_argument("--seq-len", type=int, default=2048, dest="seq_len")
    ap.add_argument("--stride", type=int, default=16,
                    help="growth-regime prefix step (<=16 gives a dense t<d=128 grid for [B]) (fix-1)")
    ap.add_argument("--layer-stride", type=int, default=1, dest="layer_stride",
                    help="1 = ALL gdn2 layers (recommended for repro judgement)")
    ap.add_argument("--domains", default=None, help="comma list; default all 5")
    ap.add_argument("--seed", type=int, default=0, help="[PIN-6] torch/np/PYTHONHASHSEED seed")
    ap.add_argument("--require-real-data", action="store_true", dest="require_real_data",
                    help="[fix-4] hard-fail if any NATURAL domain falls back to synthetic data")
    ap.add_argument("--resume", action="store_true",
                    help="skip domains whose partial_<dom>.json already exists in --out")
    ap.add_argument("--smoke", action="store_true", help="CPU synthetic-state smoke (no model)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    set_seed(args.seed)      # [PIN-6] pin RNG before any data is drawn

    if args.smoke:
        return _smoke(args)

    import loader_gdn2
    print(f"[load] gdn2-1.3B checkpoint={args.ckpt} config={loader_gdn2.CONFIG_NAME} seed={args.seed} [PIN-1/6]", flush=True)
    bundle = loader_gdn2.load(checkpoint_path=args.ckpt)
    from transformers import AutoTokenizer
    import common as gdn2_common
    tok = AutoTokenizer.from_pretrained(gdn2_common.TOKENIZER)

    report = run(bundle, tok, args)
    report["ckpt_provenance"] = getattr(bundle, "ckpt_provenance", None)   # [PIN-1] resolved path/size/hash
    report["at_capture_mode"] = getattr(bundle, "capture_mode", None)      # kernel|reconstruct|none
    report["git_head"] = _git_head()
    report["runtime_versions"] = _runtime_versions()
    report["seed"] = args.seed
    prov = report.get("ckpt_provenance") or {}
    tag = "%s_%s_%s" % (time.strftime("%y%m%d"), prov.get("token_tag", "ckpt"), report["git_head"])
    out = os.path.join(args.out, f"stage1_report_{tag}.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    # stable alias for downstream tooling
    with open(os.path.join(args.out, "stage1_report.json"), "w") as f:
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
    data = {d: (ids_list, {"domain": d, "source": "smoke", "is_fallback": False,
                           "seq_len": 256, "n_seq": 4})
            for d in ["wikitext", "github", "arxiv", "repeat_rarechar"]}
    report = {"config": PREREG, "domains": list(data.keys()), "per_domain": {}, "args": vars(args),
              "timestamp": "smoke"}
    for dom, (ids, meta) in data.items():
        # small stride -> dense growth grid inside t<d (fix-1). d=64 here so t<=64 is the growth band.
        report["per_domain"][dom] = analyze_domain(bundle, ids, 256, stride=8, layer_stride=1, meta=meta)
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
    # sanity 1: planted data should read as bimodal
    assert report["per_domain"]["wikitext"]["stratification"]["is_bimodal_BC"] or \
        report["per_domain"]["wikitext"]["stratification"]["low_frac"] > 0.3, "smoke: expected stratification"
    # sanity 2 (fix-1): growth-regime time consistency produced finite rho + a NORM axis, and counted
    # both regimes. Planted ranks are prefix-growing/order-preserving so median rho should be high.
    tc = report["per_domain"]["wikitext"]["time_consistency"]
    assert np.isfinite(tc["median_rho_growth"]), "smoke: expected finite growth-regime rho"
    assert "median_norm_cosine_growth" in tc, "smoke: expected nuclear-norm axis (fix-3)"
    assert (tc["n_growth_pairs_total"] + tc["n_saturated_pairs_total"]) > 0, "smoke: pair count"
    assert tc["median_rho_growth"] > 0.5, "smoke: planted order-preserving ranks -> rho>0.5"
    # sanity 3 (fix-2): the PAPER bound min(t,d) residual block exists and no un-labelled theory curve.
    reg = report["per_domain"]["wikitext"]["rbar_regression"]
    assert "paper_theory_bound" in reg and "min(t, d)" in reg["paper_theory_bound"]["form"], "smoke: fix-2"
    if "aux_decay_heuristic" in reg:
        assert "NOT from arXiv" in reg["aux_decay_heuristic"]["label"], "smoke: heuristic must be labelled"
    # sanity 4 (fix-4): a fabricated fallback natural domain forces G1a=INVALID (None).
    fb_report = json.loads(json.dumps(report))       # deep copy
    fb_report["per_domain"]["arxiv"]["meta"]["is_fallback"] = True
    fb_report["per_domain"]["arxiv"]["meta"]["source"] = "fallback"
    assert verdict(fb_report)["G1a_stratification_and_time"] is None, "smoke: fallback must force INVALID"
    print("[smoke] OK (fix-1 growth+sat, fix-2 paper-bound, fix-3 norm-cosine, fix-4 fallback-INVALID)")


if __name__ == "__main__":
    main()
