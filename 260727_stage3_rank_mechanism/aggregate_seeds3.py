"""Aggregate Stage 3 across >=3 seeds -> final G3 (the gate is defined on the seed aggregate, PIN-6).

Reads results/<base>_seed{0,1,2}/stage3_report.json and aggregates, per head GROUP {high,low,random}:
  - int-1 prune-fraction PPL curve (macro PPL vs fraction; marks 0.389 paper / 0.43 ours)
  - int-2 SVD top-r dose-response (macro PPL vs retained-rank ratio) + PPL-vs-contentloss SLOPE
  - int-3 spectrum-noise ladder (origin -> top-r@0.5 -> spectrum -> zero) + spectrum-noise DELTA
all as mean+-std across seeds, plus the CONTRAST metrics delta(high)-delta(low) and both vs random.

G3 (aggregate) fires iff HIGH is more content-sensitive than LOW on BOTH:
  (int-2) mean SVD-top-r slope(high) > slope(low)      [full rank is USED]
  (int-3) mean spectrum-noise delta(high) > delta(low) [content matters despite rank/energy preserved]
i.e. high-rank heads use their spectral capacity (definition-A rank IS genuine) => NOT oversaturated
=> the paper's "high-rank => oversaturated => prunable" leap is REFUTED on pure GDN. Reverse/flat =
rank genuinely idle -> LIMITATION (PREREG), not a code failure. PPL is the primary DV (NIAH floored).

Usage: python aggregate_seeds3.py --results-root results --base stage3_100b --seeds 0,1,2 \
         --out results/aggregate_stage3.json
"""
import argparse
import glob
import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
import sys
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from stage3_mechanism import PRUNE_FRACTIONS, TOPR_RATIOS, NOISE_LADDER  # frozen grids


def _load_reports(root, base, seeds):
    reps = []
    for s in seeds:
        d = os.path.join(root, f"{base}_seed{s}")
        cand = [f for f in glob.glob(os.path.join(d, "stage3_report*.json")) if "smoke" not in f]
        pref = os.path.join(d, "stage3_report.json")
        f = pref if os.path.isfile(pref) else (sorted(cand)[-1] if cand else None)
        if f:
            reps.append((s, json.load(open(f))))
        else:
            print(f"  [agg] WARNING: no report for seed {s} in {d}")
    return reps


def _ms(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and np.isfinite(x)]
    return {"mean": float(np.mean(xs)) if xs else float("nan"),
            "std": float(np.std(xs)) if xs else float("nan"), "n": len(xs)}


def _curve(reps, section, group, keys, field="macro_ppl"):
    """mean+-std of `field` at each key across seeds, for one group/section."""
    out = {}
    for key in keys:
        vals = []
        for _s, r in reps:
            try:
                vals.append(r[section][group][key][field])
            except Exception:
                pass
        out[str(key)] = _ms(vals)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--base", default="stage3_100b")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    reps = _load_reports(a.results_root, a.base, seeds)
    if len(reps) < 2:
        print(f"[agg] only {len(reps)} seed(s) found -> cannot aggregate G3 (need >=2, prefer 3)."); return

    groups = ("high", "low", "random")
    frac_keys = [f"{f:.3f}" for f in PRUNE_FRACTIONS]
    ratio_keys = [f"{r:.3f}" for r in TOPR_RATIOS]

    # ---- per-group mean+-std curves ----
    int1 = {g: _curve(reps, "int1_prune_fraction", g, frac_keys) for g in groups}
    int2 = {g: _curve(reps, "int2_topr", g, ratio_keys) for g in groups}
    int3 = {g: _curve(reps, "int3_noise_ladder", g, NOISE_LADDER) for g in groups}

    # ---- per-seed verdict signals -> aggregate ----
    slope = {g: [] for g in groups}
    spec = {g: [] for g in groups}
    g3_seed = []
    controls = []
    origin_niah = []
    for _s, r in reps:
        v = r.get("verdict", {})
        sl = v.get("int2_ppl_vs_contentloss_slope", {})
        sd = v.get("int3_spectrum_noise_delta_vs_origin", {})
        for g in groups:
            slope[g].append(sl.get(g, float("nan")))
            spec[g].append(sd.get(g, float("nan")))
        g3_seed.append(bool(v.get("G3_genuine_capacity_seed", False)))
        controls.append(bool(v.get("segment_control_reproduces_origin", False)))
        if v.get("origin_niah") is not None:
            origin_niah.append(v["origin_niah"])

    slope_ms = {g: _ms(slope[g]) for g in groups}
    spec_ms = {g: _ms(spec[g]) for g in groups}

    # contrasts (high - low) and (high - random) on the two content-sensitivity signals
    def contrast(ms, a_, b_):
        return ms[a_]["mean"] - ms[b_]["mean"] if (np.isfinite(ms[a_]["mean"])
                                                   and np.isfinite(ms[b_]["mean"])) else float("nan")
    topr_high_minus_low = contrast(slope_ms, "high", "low")
    topr_high_minus_rand = contrast(slope_ms, "high", "random")
    spec_high_minus_low = contrast(spec_ms, "high", "low")
    spec_high_minus_rand = contrast(spec_ms, "high", "random")

    topr_dissoc = bool(np.isfinite(topr_high_minus_low) and topr_high_minus_low > 0)
    spec_dissoc = bool(np.isfinite(spec_high_minus_low) and spec_high_minus_low > 0)
    g3 = bool(topr_dissoc and spec_dissoc)
    controls_ok = bool(sum(controls) > len(controls) / 2)

    verdict = (
        "G3=YES: high-rank heads USE their spectral capacity (SVD top-r truncation AND spectrum-noise "
        "degrade LM more than on low-rank/random heads) => full rank = GENUINE capacity, not "
        "oversaturated => the paper's 'high-rank => oversaturated => prunable' leap is REFUTED on pure "
        "GDN. Combined with Stage 2 (high-rank pruning catastrophic), definition-A saturation (full "
        "rank) != definition-B saturation (interference)."
        if g3 else
        "G3=NO/NULL: high-rank content-sensitivity does NOT exceed low-rank on both surgeries -> the "
        "high-rank rank may be genuinely idle/redundant on this pure 18L gdn2. Record as a "
        "generalization LIMITATION (PREREG), not a code failure.")

    agg = {
        "n_seeds": len(reps), "seeds": [s for s, _ in reps],
        "primary_dv": "ppl",
        "niah_untestable_floor": bool(len(origin_niah) == 0 or np.mean(origin_niah) < 0.30),
        "origin_niah_mean": (float(np.mean(origin_niah)) if origin_niah else None),
        "segment_control_reproduces_origin_majority": controls_ok,
        "int1_prune_fraction_ppl": int1,
        "int2_topr_ppl": int2,
        "int3_noise_ladder_ppl": int3,
        "int2_topr_slope(mean+-std)": slope_ms,
        "int3_spectrum_delta(mean+-std)": spec_ms,
        "contrast_topr_slope": {"high-low": topr_high_minus_low, "high-random": topr_high_minus_rand},
        "contrast_spectrum_delta": {"high-low": spec_high_minus_low, "high-random": spec_high_minus_rand},
        "topr_dissociation(high>low)": topr_dissoc,
        "spectrum_dissociation(high>low)": spec_dissoc,
        "G3_per_seed": g3_seed,
        "G3_FINAL": g3,
        "verdict": verdict,
        "prereg_caveat": ("paper 93.8/90.6/46.9/38.9% = Qwen3-Next(48L, post-trained) TARGET lines, "
                          "not pass standards for 18L gdn2 [PIN-4]. Interventions = POST-HOC state "
                          "surgery. S-NIAH = RULER-multikey (!= paper single-needle); only within-run "
                          "high/low/random asymmetry is the signal."),
    }
    if not controls_ok:
        agg["WARNING"] = ("S=1 harness-faithfulness control FAILED in majority of seeds -> segmented "
                          "surgery-off path does not reproduce single-shot origin; G3 signal suspect.")
    out = a.out or os.path.join(a.results_root, "aggregate_stage3.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    json.dump(agg, open(out, "w"), indent=2)
    print(json.dumps({k: agg[k] for k in
                      ("n_seeds", "int2_topr_slope(mean+-std)", "int3_spectrum_delta(mean+-std)",
                       "contrast_topr_slope", "contrast_spectrum_delta", "G3_FINAL", "verdict")},
                     indent=2))
    print(f"[agg] wrote {out}")


if __name__ == "__main__":
    main()
