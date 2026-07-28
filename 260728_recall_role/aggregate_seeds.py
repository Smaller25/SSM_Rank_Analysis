"""Aggregate Stage 4 across >=3 seeds -> final recall-role verdict (defined on the seed aggregate, PIN-6).

Reads results/<base>_seed{0,1,2}/stage4_report.json, aggregates per-condition local/recall/gain bits
(mean+-std) and the Delta_recall / Delta_local degradation deltas, then decides the recall-role
verdict on the AGGREGATE:
  RECALL_ROLE = mean(Delta_recall_high) > mean(Delta_local_high)                     (recall-specific)
            AND mean(Delta_recall_high) > mean(Delta_recall_low)
            AND mean(Delta_recall_high) > mean(Delta_recall_random)                  (HIGH-specific)
Headroom gate (majority): if origin induction_gain <= 0.30 bits in the MAJORITY of seeds the probe is
UNTESTABLE_HEADROOM and the verdict is withheld (same majority-gate pattern as Stage 2's NIAH floor).
NULL/reversal = LIMITATION to record (not a code failure).

Usage: python aggregate_seeds.py --results-root results --base recall_role_100b \
           --out results/aggregate_stage4.json
"""
import argparse
import glob
import json
import os

import numpy as np

HEADROOM_GAIN_BITS = 0.30   # [PREREG] must match stage4_recall_role.HEADROOM_GAIN_BITS


def _load_reports(root, base, seeds):
    reps = []
    for s in seeds:
        d = os.path.join(root, f"{base}_seed{s}")
        cand = [f for f in glob.glob(os.path.join(d, "stage4_report*.json")) if "smoke" not in f]
        pref = os.path.join(d, "stage4_report.json")
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
    ap.add_argument("--base", default="recall_role_100b")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    reps = _load_reports(a.results_root, a.base, seeds)
    if len(reps) < 2:
        print(f"[agg] only {len(reps)} seed(s) found -> cannot aggregate verdict (need >=2, prefer 3)."); return

    conds = ("origin", "high", "low", "random")
    per_cond = {c: {"local": [], "recall": [], "gain": []} for c in conds}
    dr_high, dr_low, dr_rand = [], [], []      # Delta_recall vs origin
    dl_high, dl_low, dl_rand = [], [], []      # Delta_local  vs origin
    origin_gain = []
    headroom_flags = []
    kv = {}
    for s, r in reps:
        cc = r["conditions"]
        for c in conds:
            per_cond[c]["local"].append(cc[c]["local_bits"])
            per_cond[c]["recall"].append(cc[c]["recall_bits"])
            per_cond[c]["gain"].append(cc[c]["induction_gain"])
        v = r["verdict"]
        dr_high.append(v["delta_recall_vs_origin"]["high"]); dr_low.append(v["delta_recall_vs_origin"]["low"]); dr_rand.append(v["delta_recall_vs_origin"]["random"])
        dl_high.append(v["delta_local_vs_origin"]["high"]);  dl_low.append(v["delta_local_vs_origin"]["low"]);  dl_rand.append(v["delta_local_vs_origin"]["random"])
        origin_gain.append(v["origin_induction_gain"])
        headroom_flags.append(bool(v.get("headroom_ok(origin_gain>thr)", False)))
        cf = r.get("conditions_full", {})
        for c in ("high", "low", "random"):
            if c in cf and "kv_reduction" in cf[c]:
                kv[c] = cf[c]["kv_reduction"]

    # majority headroom gate: need origin gain > threshold in > half the seeds
    headroom_majority = sum(headroom_flags) > len(headroom_flags) / 2
    drH, drL, drR = _ms(dr_high), _ms(dr_low), _ms(dr_rand)
    dlH, dlL, dlR = _ms(dl_high), _ms(dl_low), _ms(dl_rand)

    recall_specific = bool(drH["mean"] > dlH["mean"])                 # Delta_recall(high) > Delta_local(high)
    high_vs_low = bool(drH["mean"] > drL["mean"])
    high_vs_random = bool(drH["mean"] > drR["mean"])
    high_specific = bool(high_vs_low and high_vs_random)
    recall_role = bool(headroom_majority and recall_specific and high_specific)

    if not headroom_majority:
        status = "UNTESTABLE_HEADROOM"
    elif recall_role:
        status = "RECALL_ROLE_SUPPORTED"
    else:
        status = "NULL_OR_REVERSAL"

    agg = {
        "n_seeds": len(reps), "seeds": [s for s, _ in reps],
        "headroom_threshold_bits": HEADROOM_GAIN_BITS,
        "origin_induction_gain": _ms(origin_gain),
        "headroom_majority(origin_gain>thr)": headroom_majority,
        "per_condition": {c: {"local_bits": _ms(per_cond[c]["local"]),
                              "recall_bits": _ms(per_cond[c]["recall"]),
                              "induction_gain": _ms(per_cond[c]["gain"])} for c in conds},
        "delta_recall_vs_origin": {"high": drH, "low": drL, "random": drR},
        "delta_local_vs_origin": {"high": dlH, "low": dlL, "random": dlR},
        "kv_reduction": kv,
        "recall_specific_high(dRecall>dLocal, aggregate)": recall_specific,
        "high_vs_low(dRecall_high>dRecall_low)": high_vs_low,
        "high_vs_random(dRecall_high>dRecall_random)": high_vs_random,
        "high_recall_specific(vs low & random)": high_specific,
        "RECALL_ROLE_FINAL": recall_role,
        "STATUS_FINAL": status,
        "verdict": (
            "RECALL-ROLE=YES: HIGH-rank pruning collapses induction_gain specifically on the RECALL "
            "segment (Delta_recall(high) >> Delta_local(high) AND >> low/random) -> HIGH-rank heads "
            "are in-context recall units; refutes the paper's oversaturation (high-rank=prunable junk) "
            "reading in the role dimension." if status == "RECALL_ROLE_SUPPORTED" else
            "UNTESTABLE: origin induction_gain <= %.2f bits in the majority of seeds -> the probe has "
            "no recall headroom to lose; record as untestable, not a null." % HEADROOM_GAIN_BITS
            if status == "UNTESTABLE_HEADROOM" else
            "RECALL-ROLE=NO/REVERSAL: HIGH-rank pruning is NOT recall-specific (Delta_recall(high) ~ "
            "Delta_local(high) or ~ low/random) -> HIGH-rank is general load-bearing, not a recall "
            "unit; record as a LIMITATION (not a code failure)."),
        "prereg_caveat": ("paper 93.8/46.9/90.6/38.9% = Qwen3-Next(48L, post-trained) TARGET lines, "
                          "not pass standards for 18L gdn2"),
    }
    out = a.out or os.path.join(a.results_root, "aggregate_stage4.json")
    json.dump(agg, open(out, "w"), indent=2)
    print(json.dumps({k: agg[k] for k in ("n_seeds", "origin_induction_gain",
                                          "headroom_majority(origin_gain>thr)",
                                          "delta_recall_vs_origin", "delta_local_vs_origin",
                                          "STATUS_FINAL", "verdict")}, indent=2))
    print(f"[agg] wrote {out}")


if __name__ == "__main__":
    main()
