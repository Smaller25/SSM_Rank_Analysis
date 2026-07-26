"""Aggregate Stage 2 across >=3 seeds -> final G2 (the gate is defined on the seed aggregate, PIN-6).

Reads results/<base>_seed{0,1,2}/stage2_report.json, aggregates per-condition NIAH acc / macro PPL
(mean+-std) and the low/high/random degradation deltas, then decides G2 on the AGGREGATE:
  G2 (NIAH) = mean(delta_low) > mean(delta_high) AND mean(delta_low) > mean(delta_random)
If origin NIAH is at floor (untestable) in the majority of seeds, the primary DV falls back to PPL.

Usage: python aggregate_seeds.py --results-root results --base stage2_100b --out results/aggregate_stage2.json
"""
import argparse
import glob
import json
import os

import numpy as np


def _load_reports(root, base, seeds):
    reps = []
    for s in seeds:
        d = os.path.join(root, f"{base}_seed{s}")
        cand = [f for f in glob.glob(os.path.join(d, "stage2_report*.json")) if "smoke" not in f]
        pref = os.path.join(d, "stage2_report.json")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--base", default="stage2_100b")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    reps = _load_reports(a.results_root, a.base, seeds)
    if len(reps) < 2:
        print(f"[agg] only {len(reps)} seed(s) found -> cannot aggregate G2 (need >=2, prefer 3)."); return

    conds = ("origin", "high", "low", "random")
    per_cond = {c: {"niah": [], "ppl": []} for c in conds}
    dlow_n, dhigh_n, drand_n = [], [], []      # NIAH accuracy drop vs origin
    dlow_p, dhigh_p, drand_p = [], [], []      # PPL rise vs origin
    untest = []
    kv = {}
    for s, r in reps:
        cc = r["conditions"]
        for c in conds:
            per_cond[c]["niah"].append(cc[c]["niah_retrieval_accuracy"])
            per_cond[c]["ppl"].append(cc[c]["macro_ppl"])
        v = r["verdict"]
        dlow_n.append(v["delta_niah_drop"]["low"]); dhigh_n.append(v["delta_niah_drop"]["high"]); drand_n.append(v["delta_niah_drop"]["random"])
        dlow_p.append(v["delta_ppl_rise"]["low"]);  dhigh_p.append(v["delta_ppl_rise"]["high"]);  drand_p.append(v["delta_ppl_rise"]["random"])
        untest.append(bool(v.get("niah_untestable_floor", False)))
        # kv reduction (same per condition across seeds; take from conditions_full if present)
        cf = r.get("conditions_full", {})
        for c in ("high", "low", "random"):
            if c in cf and "kv_reduction" in cf[c]:
                kv[c] = cf[c]["kv_reduction"]

    niah_floored = sum(untest) > len(untest) / 2
    dl_n, dh_n, dr_n = _ms(dlow_n), _ms(dhigh_n), _ms(drand_n)
    dl_p, dh_p, dr_p = _ms(dlow_p), _ms(dhigh_p), _ms(drand_p)
    g2_niah = bool(dl_n["mean"] > dh_n["mean"] and dl_n["mean"] > dr_n["mean"])
    g2_ppl = bool(dl_p["mean"] > dh_p["mean"] and dl_p["mean"] > dr_p["mean"])
    primary = "ppl" if niah_floored else "niah"
    g2 = g2_ppl if niah_floored else g2_niah

    agg = {
        "n_seeds": len(reps), "seeds": [s for s, _ in reps],
        "primary_dv": primary, "niah_floored_majority": niah_floored,
        "per_condition": {c: {"niah_acc": _ms(per_cond[c]["niah"]),
                              "macro_ppl": _ms(per_cond[c]["ppl"])} for c in conds},
        "delta_niah_drop_vs_origin": {"low": dl_n, "high": dh_n, "random": dr_n},
        "delta_ppl_rise_vs_origin": {"low": dl_p, "high": dh_p, "random": dr_p},
        "kv_reduction": kv,
        "G2_niah(low>high & low>random, aggregate)": g2_niah,
        "G2_ppl(low>high & low>random, aggregate)": g2_ppl,
        "G2_FINAL": g2,
        "verdict": ("G2=YES: pruning-asymmetry reproduced in pure GDN (low-rank pruning hurts most, "
                    "beyond random control)" if g2 else
                    "G2=NO: asymmetry not reproduced on aggregate -> record as generalization limit "
                    "(hybrid/post-training specific?), reframe before Stage 3"),
        "prereg_caveat": ("paper 93.8/46.9/90.6/38.9% = Qwen3-Next(48L, post-trained) TARGET lines, "
                          "not pass standards for 18L gdn2 [PIN-4]"),
    }
    out = a.out or os.path.join(a.results_root, "aggregate_stage2.json")
    json.dump(agg, open(out, "w"), indent=2)
    print(json.dumps({k: agg[k] for k in ("n_seeds", "primary_dv", "delta_niah_drop_vs_origin",
                                          "delta_ppl_rise_vs_origin", "kv_reduction", "G2_FINAL",
                                          "verdict")}, indent=2))
    print(f"[agg] wrote {out}")


if __name__ == "__main__":
    main()
