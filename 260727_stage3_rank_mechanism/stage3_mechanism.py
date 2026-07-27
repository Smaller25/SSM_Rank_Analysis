"""Stage 3 driver (G3) — rank = GENUINE CAPACITY vs oversaturated junk, on pure gdn2-1.3B (100B).

Refutes the paper's (arXiv:2602.02195) chain "full rank => oversaturated => prunable" by dissociating
two definitions of saturation on the SAME heads Stage 2 used:
  (A) linear-algebraic saturation = full threshold-rank
  (B) functional saturation        = capacity exceeded, information destroyed

Stage 2 already showed the REVERSAL (high-rank KV-state pruning is catastrophic, low ~ random). Stage 3
asks whether high-rank heads USE that rank capacity (genuine) or carry saturated junk, via post-hoc
SPECTRAL-CONTENT surgery on each head's recurrent state S_h, comparing head GROUPS high / low / random
(reusing Stage 2's k=min disjoint count-matched classifier).

Three interventions (grids FROZEN in PREREGISTRATION.md before any logging run):
  (int-1) PRUNE-FRACTION sweep  : head_mask KV-state v-zeroing of a fraction of each group's heads.
                                  single-shot path (no surgery needed; zero-state == KV prune). Marks
                                  the paper's 0.389 (KV 38.9%) and our Stage-2 0.43 points.
  (int-2) SVD TOP-r dose        : replace S_h by its rank-r truncation, r/cap in a grid. SEGMENTED
                                  path. High-rank genuine => sharp PPL rise as r shrinks; low-rank
                                  already low-dim => flat. Tests "is the full rank actually used".
  (int-3) SPECTRUM-MATCHED noise: replace S_h by U_rand diag(sigma) V_rand^T (rank/energy preserved,
                                  singular VECTORS randomized). SEGMENTED path. Ladder
                                  origin -> top-r(0.5) -> spectrum-noise -> zero. High-rank genuine =>
                                  spectrum-noise catastrophic (content matters); junk => ~origin.

DV primary   = macro PPL delta vs origin (2^mean-bits/token over wikitext/github/arxiv real text).
DV secondary = S-NIAH retrieval accuracy — headroom-gated: reported only if origin NIAH >= 0.30, else
               UNTESTABLE_FLOOR and PPL is the sole signal (Stage 2 origin NIAH was 0.18). NIAH is
               de-scoped by default (--niah) to save GPU: PPL is the primary DV.

G3 signal (WITHIN-run asymmetry, the paper has none of these controls):
  int-2 slope of PPL vs decreasing r  : steep for HIGH, flat for LOW.
  int-3 spectrum-noise PPL >> origin  : for HIGH (content matters despite preserved rank/energy),
                                        ~ origin for junk.
  contrast = delta(high) - delta(low), and both vs delta(random), 3-seed mean+-std.
=> high-rank sensitive to content-destroying surgery (top-r, spectrum-noise) AND low-rank insensitive
   => high-rank = genuine capacity != oversaturated => the rank->saturation leap is REFUTED on pure GDN.
NULL/reverse (high-rank flat under top-r/spectrum-noise) = rank genuinely idle/redundant, recorded as
a LIMITATION not a code failure (PREREG).

CONSTANTS (do not tune per session):
  [PIN-1] gdn2-1.3B, Config.from_name("gdn2_1.3B"), strict=False, bf16, fused_recurrent; 100B ckpt
          ONLY (95B accepted, 10B REJECTED by resolve_and_assert_ckpt); 18L x 16H = 288 heads, no GVA.
  [PIN-2] threshold_rank eps=1e-4 (Eq.6), cap=min(dk,dv), theta_R=0.5; groups via head_classifier
          k=min(#low,#high) DISJOINT count-matched; cross-domain agreement gate ~0.97.
  [PIN-5] SAME 3 natural domains from data_cache/, >=16 seqs, up to 2048 tok, --require-real-data.
  [PIN-6] seeds 0/1/2 fully pinned, one seed per sbatch + dependency aggregate.
  Segmented intervention: segment_len FROZEN (default 256); S=1 (segment_len>=seq_len) reproduces
  single-shot origin PPL (harness-faithfulness control). See state_surgery + PREREGISTRATION.md.

PREREG CAVEAT (echoed verbatim into the report): the paper's 93.8/90.6/46.9/38.9% are Qwen3-Next
(48-layer, POST-TRAINED) TARGET lines, NOT pass standards for 18-layer pure gdn2; shortfall = a
generalization limitation to record, not a code failure. Interventions are POST-HOC state surgery
(not retraining). S-NIAH is RULER-multikey (!= paper single-needle) -> only within-run asymmetry is
the signal.

Run (greenbeard SLURM, 100B checkpoint), ONE seed per job:
  python stage3_mechanism.py --seed 0 --require-real-data --resume --out results/stage3_100b_seed0
Smoke (CPU, no model — synthetic bundle; exercises classify + all 3 interventions + verdict):
  python stage3_mechanism.py --smoke
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_STAGE1 = os.path.abspath(os.path.join(_HERE, "..", "260725_stage1_rank_stratification"))
_STAGE2 = os.path.abspath(os.path.join(_HERE, "..", "260726_stage2_pruning_asymmetry"))
_260722 = os.path.abspath(os.path.join(_HERE, "..", "260722_exp"))
for _p in (_HERE, _STAGE1, _STAGE2, _260722):
    if _p not in sys.path:
        sys.path.insert(0, _p)
_FLA_PATH = "/home/sohyung/linear-memory-routing"
if os.path.isdir(_FLA_PATH) and _FLA_PATH not in sys.path:
    sys.path.insert(0, _FLA_PATH)

import head_classifier          # noqa: E402  (reuse Stage 2 classifier, disjoint k=min groups)
import head_mask                # noqa: E402  (reuse Stage 2 KV-state v-zeroing for int-1)
import ppl_eval                 # noqa: E402  (reuse for single-shot int-1 PPL)
import state_surgery            # noqa: E402  (Stage 3 core: segmented spectral surgery)

CROSS_DOMAIN_AGREEMENT_MIN = 0.90    # sanity gate (Stage 1 was 0.971)
NIAH_FLOOR = 0.30                    # headroom gate; below this NIAH is UNTESTABLE_FLOOR -> PPL primary

# ------------------------------------------------------------- PREREGISTERED grids (FROZEN, PIN-4)
# int-1 prune fractions of heads per group (KV-state v-zeroing). Includes paper's 0.389 (KV 38.9%)
# and our Stage-2 0.43 point so both external markers are on the curve.
PRUNE_FRACTIONS = [0.0, 0.10, 0.20, 0.30, 0.389, 0.43, 0.50, 0.60]
# int-2 SVD retained-rank ratios r/cap per group (1.0 == origin/no-op; 0 == zero-state).
TOPR_RATIOS = [1.0, 0.75, 0.50, 0.25, 0.125, 0.0]
# int-3 spectrum-noise ladder step (origin -> top-r(0.5) -> spectrum-noise -> zero) per group.
NOISE_LADDER = ["origin", "topr@0.5", "spectrum", "zero"]
SEGMENT_LEN_DEFAULT = 256            # FROZEN segment length for the segmented intervention (int-2/3)

PREREG_CAVEAT = ("paper 93.8/90.6/46.9 NIAH + KV 38.9% are Qwen3-Next (48-layer, POST-TRAINED) "
                 "OBSERVATIONS adopted as reproduction TARGET lines, NOT pass standards for 18-layer "
                 "gdn2; falling short = generalization limitation to record, not code failure [PIN-4]. "
                 "Interventions are POST-HOC state surgery (not retraining). S-NIAH is RULER-multikey "
                 "(!= paper single-needle) -> only WITHIN-run high/low/random asymmetry is the signal.")
DECLARED_DEVIATION = ("Pure-rank head classification is a DECLARED deviation from JRNP Eq.14 "
                      "S_h=alpha*(rbar/d)+(1-alpha)*(nbar/max nbar): alpha is UNPUBLISHED, so we "
                      "classify on normalized threshold-rank alone (theta_R=0.5) [PIN-2].")


def set_seed(seed):
    """[PIN-6] pin torch / numpy / python-hash / cuda RNG."""
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


def reset_shared_cache(bundle):
    """[fix blocker-1/2] purge stale recurrent-state cache before any single-shot logits eval, so the
    patched GatedDeltaNet2.forward takes the stateless branch (see stage2_pruning.reset_shared_cache).
    The SegmentedSurgeon path installs+clears its OWN cache and does not rely on this."""
    base = getattr(bundle, "base", bundle)
    shared = getattr(base, "shared", None)
    if isinstance(shared, dict) and shared.get("cache") is not None:
        shared["cache"] = None


def draw_random_heads(all_heads, k, seed, salt=0):
    """k randomly-chosen heads (the count-matched control), reproducible per seed (PIN-6)."""
    rng = np.random.default_rng(int(seed) * 97 + salt)
    idx = rng.choice(len(all_heads), size=k, replace=False)
    return [tuple(all_heads[i]) for i in sorted(idx.tolist())]


def frac_subset(heads, frac, seed, salt=0):
    """A reproducible fraction (round(frac*len)) of `heads` for the int-1 prune sweep (PIN-6). Uses a
    fixed ordering (heads is already the classifier's rank-sorted list) then a seeded permutation so
    the same fraction is a superset-stable, seed-reproducible draw."""
    n = int(round(frac * len(heads)))
    if n <= 0:
        return []
    if n >= len(heads):
        return [tuple(h) for h in heads]
    rng = np.random.default_rng(int(seed) * 131 + salt)
    perm = rng.permutation(len(heads))
    return [tuple(heads[i]) for i in sorted(perm[:n].tolist())]


# --------------------------------------------------------------------------------- INT-1 (single-shot)
def run_int1_prune_fraction(bundle, tok, masker, ranked_heads, domain_ids, args):
    """int-1: COUNT-MATCHED prune-fraction sweep (fix major int-1). At fraction f, mask exactly
    n=round(f*total) heads in EACH arm so the KV fraction (=n/total) is identical across arms and
    directly comparable to the paper's 38.9%: high = top-n by rank, low = bottom-n, random = seeded
    random-n from all heads. `ranked_heads` = all (layer,head) sorted by ascending agg_norm_rank.
    Single-shot PPL (v-zero == zero-state, no surgery)."""
    out = {"high": {}, "low": {}, "random": {}}
    total = len(ranked_heads)                       # = n_layers * num_heads
    for frac in PRUNE_FRACTIONS:
        n = int(round(frac * total))
        key = f"{frac:.3f}"
        low_sub = [tuple(h) for h in ranked_heads[:n]]                    # lowest-rank n
        high_sub = [tuple(h) for h in ranked_heads[total - n:]] if n > 0 else []  # highest-rank n
        rng = np.random.default_rng(int(args.seed) * 131 + int(frac * 1000))
        ridx = sorted(rng.permutation(total)[:n].tolist()) if n > 0 else []
        rand_sub = [tuple(ranked_heads[i]) for i in ridx]
        for g, sub in (("high", high_sub), ("low", low_sub), ("random", rand_sub)):
            reset_shared_cache(bundle)
            masker.set_mask(set(sub))
            kv = masker.kv_reduction(set(sub))
            ppl = ppl_eval.run_ppl(bundle, domain_ids)
            masker.clear()
            out[g][key] = {"frac": frac, "n_masked": len(sub),
                           "kv_reduction_pct": kv["kv_reduction_pct"],
                           "macro_ppl": ppl["_macro"]["mean_ppl"],
                           "macro_bits": ppl["_macro"]["mean_bits_per_token"],
                           "per_domain_ppl": {d: ppl[d]["ppl"] for d in ppl if d != "_macro"}}
            print(f"    [int1 {g} frac={frac:.3f}] n={len(sub)} kv={kv['kv_reduction_pct']}% "
                  f"macro_ppl={out[g][key]['macro_ppl']:.3f}", flush=True)
    return out


# --------------------------------------------------------------------------------- INT-2 (segmented)
def run_int2_topr(surgeon, groups, domain_ids, args):
    """int-2: SVD top-r dose-response via SEGMENTED surgery. Returns {group:{ratio:{macro_ppl,...}}}.
    ratio=1.0 is the no-op origin (target_heads present but surgery is identity -> must ~ origin)."""
    out = {g: {} for g in groups}
    for g, heads in groups.items():
        for ratio in TOPR_RATIOS:
            key = f"{ratio:.3f}"
            surgery = None if ratio >= 1.0 else "topr"
            ppl = state_surgery.run_segmented_ppl(
                surgeon, domain_ids, target_heads=heads,
                surgery=surgery, ratio=ratio, seed=args.seed)
            out[g][key] = {"ratio": ratio, "n_heads": len(heads),
                           "macro_ppl": ppl["_macro"]["mean_ppl"],
                           "macro_bits": ppl["_macro"]["mean_bits_per_token"],
                           "per_domain_ppl": {d: ppl[d]["ppl"] for d in ppl if d != "_macro"}}
            print(f"    [int2 {g} r/cap={ratio:.3f}] macro_ppl={out[g][key]['macro_ppl']:.3f}",
                  flush=True)
    return out


# --------------------------------------------------------------------------------- INT-3 (segmented)
def _ladder_call(surgeon, domain_ids, heads, step, seed):
    if step == "origin":
        return surgeon, dict(target_heads=heads, surgery=None, ratio=1.0, seed=seed)
    if step == "topr@0.5":
        return surgeon, dict(target_heads=heads, surgery="topr", ratio=0.5, seed=seed)
    if step == "spectrum":
        return surgeon, dict(target_heads=heads, surgery="spectrum", ratio=1.0, seed=seed)
    if step == "zero":
        return surgeon, dict(target_heads=heads, surgery="zero", ratio=1.0, seed=seed)
    raise ValueError(step)


def run_int3_noise_ladder(surgeon, groups, domain_ids, args):
    """int-3: ladder origin -> top-r(0.5) -> spectrum-noise -> zero via SEGMENTED surgery.
    Returns {group:{step:{macro_ppl,...}}}."""
    out = {g: {} for g in groups}
    for g, heads in groups.items():
        for step in NOISE_LADDER:
            _, kw = _ladder_call(surgeon, domain_ids, heads, step, args.seed)
            ppl = state_surgery.run_segmented_ppl(surgeon, domain_ids, **kw)
            out[g][step] = {"step": step, "n_heads": len(heads),
                            "macro_ppl": ppl["_macro"]["mean_ppl"],
                            "macro_bits": ppl["_macro"]["mean_bits_per_token"],
                            "per_domain_ppl": {d: ppl[d]["ppl"] for d in ppl if d != "_macro"}}
            print(f"    [int3 {g} {step}] macro_ppl={out[g][step]['macro_ppl']:.3f}", flush=True)
    return out


# --------------------------------------------------------------- harness-faithfulness control (S=1)
def run_segment_control(surgeon, bundle, domain_ids, args):
    """PREREG controls for harness faithfulness. Two checks:

    (1) S=1 CONTROL (hard): segmented forward with segment_len >= seq_len (1 segment, 0 boundaries,
        surgery off) MUST reproduce the single-shot origin PPL bit-for-bit. This certifies the
        SegmentedSurgeon carry+readout is a faithful control (the surgery-off path is the model).
    (2) MULTI-SEGMENT ORIGIN DRIFT (diagnostic): the surgery-off forward at the CONFIGURED
        segment_len vs single-shot. GDN2's recurrence is associative and the fla Cache carries both
        recurrent_state AND conv_state, so exact carry => ~0 drift. Any drift is a segmentation
        artifact and is REPORTED so the int-2/int-3 deltas (measured vs the segmented same-segment_len
        origin, which cancels this drift) are interpreted correctly.

    Returns both comparisons; verdict gates on (1)."""
    reset_shared_cache(bundle)
    single = ppl_eval.run_ppl(bundle, domain_ids)["_macro"]["mean_ppl"]
    big = state_surgery.SegmentedSurgeon(surgeon.bundle, segment_len=10 ** 9)
    seg1 = state_surgery.run_segmented_ppl(big, domain_ids, target_heads=[], surgery=None,
                                           ratio=1.0, seed=args.seed)["_macro"]["mean_ppl"]
    segN = state_surgery.run_segmented_ppl(surgeon, domain_ids, target_heads=[], surgery=None,
                                           ratio=1.0, seed=args.seed)["_macro"]["mean_ppl"]
    diff = abs(single - seg1) if (np.isfinite(single) and np.isfinite(seg1)) else float("nan")
    drift = abs(single - segN) if (np.isfinite(single) and np.isfinite(segN)) else float("nan")
    ok = bool(np.isfinite(diff) and diff <= 1e-2 * max(1.0, single))
    print(f"  [control S=1] single_shot={single:.4f} segmented_1seg={seg1:.4f} |diff|={diff:.4g} "
          f"reproduces_origin={ok}", flush=True)
    print(f"  [control seg_len={surgeon.segment_len}] surgery_off_multiseg={segN:.4f} "
          f"drift_vs_single={drift:.4g} (segmentation artifact diagnostic)", flush=True)
    return {"single_shot_macro_ppl": single, "segmented_1seg_macro_ppl": seg1,
            "segmented_multiseg_origin_macro_ppl": segN,
            "abs_diff": diff, "reproduces_origin": ok,
            "multiseg_origin_drift_vs_single": drift, "segment_len": surgeon.segment_len,
            "note": ("S=1 segmented path must reproduce single-shot origin PPL (hard). Multi-segment "
                     "surgery-off drift vs single-shot is a reported diagnostic; int-2/int-3 deltas "
                     "are measured vs the segmented same-segment_len origin so this drift cancels.")}


# ----------------------------------------------------------------------------------------- VERDICT
def _slope(xs, ys):
    """Least-squares slope of ys vs xs (finite pairs only). Used for int-2 PPL-vs-(decreasing r)."""
    pts = [(x, y) for x, y in zip(xs, ys) if np.isfinite(x) and np.isfinite(y)]
    if len(pts) < 2:
        return float("nan")
    x = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
    return float(np.polyfit(x, y, 1)[0])


def compute_verdict(int1, int2, int3, control, origin_niah=None):
    """G3 gate from ONE seed's interventions (PPL is the primary DV; NIAH headroom-gated).

    Signals (all deltas vs the group's OWN origin so a group baseline shift can't confound):
      int-2 : PPL rise as r/cap shrinks 1->0 (retained-rank ratio). Slope of macro_ppl vs (1-ratio)
              [content-loss dose]. HIGH genuine => steep positive; LOW low-dim => ~flat.
      int-3 : spectrum-noise macro_ppl delta vs the group's origin. HIGH => large (content matters
              despite rank/energy preserved); LOW/junk => ~0.
    Contrast = high vs low, and both vs random. G3_seed fires if HIGH is more content-sensitive than
    LOW on BOTH int-2 slope and int-3 spectrum delta (the dissociation), for the PPL DV."""
    def macro(d, g, key):
        return d[g][key]["macro_ppl"] if (g in d and key in d[g]) else float("nan")

    # ---- int-2 slope of PPL vs content-loss dose (1 - ratio), per group ----
    doses = [1.0 - r for r in TOPR_RATIOS]                     # 0 (origin) .. 1 (zero)
    slopes = {}
    for g in ("high", "low", "random"):
        ppls = [macro(int2, g, f"{r:.3f}") for r in TOPR_RATIOS]
        slopes[g] = _slope(doses, ppls)

    # ---- int-3 spectrum-noise delta vs group origin, per group ----
    spec_delta = {}
    for g in ("high", "low", "random"):
        o = macro(int3, g, "origin"); s = macro(int3, g, "spectrum")
        spec_delta[g] = float(s - o) if (np.isfinite(o) and np.isfinite(s)) else float("nan")

    # ---- int-1 prune-fraction PPL rise at the paper (0.389) & our (0.43) marks, delta vs frac 0 ----
    def d1(g, frac):
        o = macro(int1, g, "0.000"); c = macro(int1, g, f"{frac:.3f}")
        return float(c - o) if (np.isfinite(o) and np.isfinite(c)) else float("nan")
    int1_marks = {g: {"0.389": d1(g, 0.389), "0.430": d1(g, 0.43)} for g in ("high", "low", "random")}

    topr_dissoc = bool(np.isfinite(slopes["high"]) and np.isfinite(slopes["low"])
                       and slopes["high"] > slopes["low"])
    spec_dissoc = bool(np.isfinite(spec_delta["high"]) and np.isfinite(spec_delta["low"])
                       and spec_delta["high"] > spec_delta["low"])
    high_vs_random_topr = bool(np.isfinite(slopes["high"]) and np.isfinite(slopes["random"])
                               and slopes["high"] > slopes["random"])
    high_vs_random_spec = bool(np.isfinite(spec_delta["high"]) and np.isfinite(spec_delta["random"])
                               and spec_delta["high"] > spec_delta["random"])

    niah_untestable = bool(origin_niah is None or not (origin_niah >= NIAH_FLOOR))
    g3_seed = bool(topr_dissoc and spec_dissoc)          # PPL DV (primary); NIAH is untestable at floor

    return {
        "int2_ppl_vs_contentloss_slope": slopes,
        "int3_spectrum_noise_delta_vs_origin": spec_delta,
        "int1_prune_ppl_rise_at_marks": int1_marks,
        "topr_dissociation(high>low slope)": topr_dissoc,
        "spectrum_dissociation(high>low delta)": spec_dissoc,
        "high_gt_random_topr": high_vs_random_topr,
        "high_gt_random_spectrum": high_vs_random_spec,
        "primary_dv": "ppl",
        "niah_untestable_floor": niah_untestable, "niah_floor_threshold": NIAH_FLOOR,
        "origin_niah": origin_niah,
        "segment_control_reproduces_origin": bool(control.get("reproduces_origin")),
        "G3_genuine_capacity_seed": g3_seed,
        "note": ("G3 confirmed only on the >=3-seed aggregate. G3_seed = HIGH more content-sensitive "
                 "than LOW on BOTH SVD top-r slope AND spectrum-noise delta (content-destroying "
                 "surgery), i.e. high-rank uses its rank capacity => not oversaturated. NULL/reverse "
                 "= rank genuinely idle (limitation, PREREG). PPL is primary; NIAH floored (Stage 2 "
                 "origin 0.18)."),
    }


# ----------------------------------------------------------------------------------------- one seed
def run_seed(bundle, tok, masker, args, out_dir):
    import data_stage1
    set_seed(args.seed)

    # ---- (1) RE-DERIVE head classification (PIN-2); gate on cross-domain agreement ~0.97 ----
    cls_path = os.path.join(out_dir, "head_classification.json")
    if args.resume and os.path.isfile(cls_path):
        cls = json.load(open(cls_path))
        print(f"  [classify] RESUMED from {cls_path}", flush=True)
    else:
        cls = head_classifier.classify(bundle, tok, seq_len=args.seq_len, n_seq=args.n_seq,
                                       seed=args.seed, require_real_data=args.require_real_data)
        json.dump(cls, open(cls_path, "w"), indent=2)
    agree = cls["cross_domain_agreement"]["mean"]
    print(f"  [classify] k={cls['k']} n_heads={cls['n_heads_total']} "
          f"cross_domain_agreement={agree:.3f} (gate >= {CROSS_DOMAIN_AGREEMENT_MIN})", flush=True)
    agreement_ok = bool(np.isfinite(agree) and agree >= CROSS_DOMAIN_AGREEMENT_MIN)
    if not agreement_ok:
        print(f"  [WARN] cross-domain agreement {agree:.3f} < {CROSS_DOMAIN_AGREEMENT_MIN} -> "
              f"head sets UNTRUSTED; results flagged.", flush=True)

    all_heads = [(h["layer"], h["head"]) for h in cls["per_head"]]
    # rank-ordered (ascending agg_norm_rank) full head list for the count-matched int-1 sweep (fix int-1)
    ranked_heads = [(h["layer"], h["head"]) for h in
                    sorted(cls["per_head"], key=lambda x: x.get("agg_norm_rank", 0.0))]
    # [fix major] runtime head-count provenance: config n_head can differ from the mixer's num_heads.
    total_heads_rt = masker.n_layers * masker.num_heads
    print(f"  [heads] runtime num_heads={masker.num_heads} x n_layers={masker.n_layers} "
          f"= {total_heads_rt} total (Stage1/2 used 288); n_ranked={len(ranked_heads)}", flush=True)
    assert total_heads_rt == len(all_heads), (
        f"head-count mismatch: masker {total_heads_rt} vs classifier {len(all_heads)}")
    low = [tuple(x) for x in cls["low_heads"]]
    high = [tuple(x) for x in cls["high_heads"]]
    k = cls["k"]
    rand = draw_random_heads(all_heads, k, seed=args.seed)       # PIN-6 seed governs the draw
    assert len(low) == len(high) == len(rand) == k, (
        f"count-match violated: |low|={len(low)} |high|={len(high)} |rand|={len(rand)} k={k}")
    assert set(low).isdisjoint(set(high)), (
        f"low/high overlap ({len(set(low) & set(high))} shared heads) -> contrast axis contaminated.")
    groups = {"high": high, "low": low, "random": rand}
    print(f"  [groups] k={k} high/low DISJOINT, random count-matched ({k} heads each)", flush=True)

    # ---- shared eval data drawn ONCE per seed (same across all conditions; only surgery changes) ----
    data_stage1.set_data_seed(args.seed)
    data = data_stage1.load_all(tok, seq_len=args.seq_len, n_seq=args.ppl_n_seq,
                                which=head_classifier.NATURAL_DOMAINS)
    domain_ids = {d: ids for d, (ids, _m) in data.items()}
    data_meta = {d: m for d, (_ids, m) in data.items()}

    surgeon = state_surgery.SegmentedSurgeon(bundle, segment_len=args.segment_len)

    # ---- (0) harness-faithfulness control: S=1 segmented == single-shot origin (PREREG) ----
    ctrl_path = os.path.join(out_dir, "segment_control.json")
    if args.resume and os.path.isfile(ctrl_path):
        control = json.load(open(ctrl_path))
        print("  [control S=1] RESUMED", flush=True)
    else:
        control = run_segment_control(surgeon, bundle, domain_ids, args)
        json.dump(control, open(ctrl_path, "w"), indent=2)

    # ---- optional NIAH origin (headroom gate); de-scoped by default (PPL is primary) ----
    origin_niah = None
    if args.niah:
        import niah_retrieval
        reset_shared_cache(bundle)
        niah = niah_retrieval.run_niah(bundle, tok, n_samples=args.niah_samples,
                                       max_seq_length=args.seq_len,
                                       tokens_to_generate=args.gen_tokens, seed=args.seed)
        origin_niah = niah["niah_retrieval_accuracy"]
        json.dump(niah, open(os.path.join(out_dir, "origin_niah.json"), "w"), indent=2)
        print(f"  [niah origin] acc={origin_niah:.3f} (floor gate {NIAH_FLOOR})", flush=True)

    # ---- interventions (incremental per-intervention flush; --resume) ----
    def stage(name, fn):
        p = os.path.join(out_dir, f"{name}.json")
        if args.resume and os.path.isfile(p):
            print(f"  [{name}] RESUMED from {p}", flush=True)
            return json.load(open(p))
        t0 = time.time()
        print(f"\n  === {name} ===", flush=True)
        res = fn()
        res_wrapped = {"data": res, "minutes": round((time.time() - t0) / 60, 2)}
        json.dump(res_wrapped, open(p, "w"), indent=2)
        print(f"  [{name}] done ({res_wrapped['minutes']} min) [flushed]", flush=True)
        return res_wrapped

    int1 = stage("int1_prune_fraction",
                 lambda: run_int1_prune_fraction(bundle, tok, masker, ranked_heads,
                                                 domain_ids, args))["data"]
    int2 = stage("int2_topr",
                 lambda: run_int2_topr(surgeon, groups, domain_ids, args))["data"]
    int3 = stage("int3_noise_ladder",
                 lambda: run_int3_noise_ladder(surgeon, groups, domain_ids, args))["data"]

    verdict = compute_verdict(int1, int2, int3, control, origin_niah=origin_niah)
    return {
        "seed": args.seed,
        "classification": {k2: cls[k2] for k2 in
                           ("theta_R", "eps_threshold_rank", "n_heads_total", "k",
                            "low_heads", "high_heads", "cross_domain_agreement",
                            "theta_vs_bottomk_mismatch")},
        "cross_domain_agreement_ok": agreement_ok,
        "groups": {g: [list(h) for h in sorted(hs)] for g, hs in groups.items()},
        "segment_control": control,
        "origin_niah": origin_niah,
        "int1_prune_fraction": int1,
        "int2_topr": int2,
        "int3_noise_ladder": int3,
        "grids": {"prune_fractions": PRUNE_FRACTIONS, "topr_ratios": TOPR_RATIOS,
                  "noise_ladder": NOISE_LADDER, "segment_len": args.segment_len},
        "data_meta": data_meta,
        "verdict": verdict,
    }


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
    ap.add_argument("--out", default=os.path.join(_HERE, "results", "stage3"))
    ap.add_argument("--ckpt",
                    default=os.environ.get("GDN2_CKPT_PATH", "/home/sohyung/models/gdn2_1.3B_100b.pth"),
                    help="[PIN-1] paper-matched 100B checkpoint (95B accepted; 10B REJECTED by loader)")
    ap.add_argument("--n-seq", type=int, default=16, dest="n_seq",
                    help="[PIN-5] seqs per domain for head classification (>=16)")
    ap.add_argument("--ppl-n-seq", type=int, default=16, dest="ppl_n_seq",
                    help="seqs per domain for PPL eval")
    ap.add_argument("--seq-len", type=int, default=2048, dest="seq_len")
    ap.add_argument("--segment-len", type=int, default=SEGMENT_LEN_DEFAULT, dest="segment_len",
                    help="FROZEN segment length for the segmented intervention (int-2/3). "
                         ">=seq_len => 1 segment => single-shot control.")
    ap.add_argument("--niah", action="store_true",
                    help="also run origin S-NIAH (headroom gate). Off by default: PPL is primary and "
                         "NIAH is floored (Stage 2 origin 0.18).")
    ap.add_argument("--niah-samples", type=int, default=20, dest="niah_samples")
    ap.add_argument("--gen-tokens", type=int, default=48, dest="gen_tokens")
    ap.add_argument("--seed", type=int, default=0, help="[PIN-6] governs random-group draw + sampling")
    ap.add_argument("--require-real-data", action="store_true", dest="require_real_data",
                    help="[PIN-5] hard-fail if any natural domain falls back to synthetic data")
    ap.add_argument("--resume", action="store_true",
                    help="skip classification/interventions whose JSON already exists in --out")
    ap.add_argument("--smoke", action="store_true", help="CPU synthetic-bundle smoke (no model)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    set_seed(args.seed)

    if args.smoke:
        return _smoke(args)

    import loader_gdn2
    import common as gdn2_common
    from transformers import AutoTokenizer
    print(f"[load] gdn2-1.3B ckpt={args.ckpt} config={loader_gdn2.CONFIG_NAME} seed={args.seed} "
          f"segment_len={args.segment_len} [PIN-1/6]", flush=True)
    bundle = loader_gdn2.load(checkpoint_path=args.ckpt)     # resolve_and_assert_ckpt inside
    tok = AutoTokenizer.from_pretrained(gdn2_common.TOKENIZER)
    masker = head_mask.HeadMasker(bundle)
    print(f"[mask] KV-state v-zeroing on {masker.n_layers} layers x {masker.num_heads} heads "
          f"(int-1 prune fraction; zero-state == KV prune)", flush=True)

    report = run_seed(bundle, tok, masker, args, args.out)
    report["prereg_caveat"] = PREREG_CAVEAT
    report["declared_deviation"] = DECLARED_DEVIATION
    report["paper_target_lines"] = {"model": "Qwen3-Next (48-layer, post-trained)",
                                    "origin_niah": 93.8, "prune_low_rank_niah": 46.9,
                                    "prune_high_rank_niah": 90.6, "kv_down_pct": 38.9,
                                    "note": "TARGET lines, not pass standards for 18-layer gdn2 [PIN-4]"}
    report["ckpt_provenance"] = getattr(bundle, "ckpt_provenance", None)
    report["at_capture_mode"] = getattr(bundle, "capture_mode", None)
    report["git_head"] = _git_head()
    report["runtime_versions"] = _runtime_versions()
    report["args"] = vars(args)
    report["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    prov = report.get("ckpt_provenance") or {}
    tag = "%s_%s_seed%d_%s" % (time.strftime("%y%m%d"), prov.get("token_tag", "ckpt"),
                              args.seed, report["git_head"])
    out = os.path.join(args.out, f"stage3_report_{tag}.json")
    json.dump(report, open(out, "w"), indent=2)
    json.dump(report, open(os.path.join(args.out, "stage3_report.json"), "w"), indent=2)
    print(f"\n[verdict] {json.dumps(report['verdict'], indent=2)}")
    print(f"[written] {out}")
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)   # clean exit (datasets prefetch thread can crash at interpreter teardown)


# ------------------------------------------------------------------ CPU smoke (no GPU, no model)
class _FakeBundle:
    """Synthetic bundle exercising classify + segmented surgery + int-1/2/3 wiring on CPU.

    .states  : bimodal per-head rank (even heads low ~3, odd heads high ~d) so head_classifier splits.
    .model   : a callable that maps ids -> logits DEPENDING on a live SHARED cache's recurrent_state,
               so (a) the segmented carry path is exercised and (b) surgery visibly changes PPL. It
               fakes the fla-Cache read/write the SegmentedSurgeon relies on. NOT the real model."""
    class _Model:
        def __init__(self, outer):
            self.outer = outer
            self._params = [__import__("torch").zeros(1)]
            self.lm_head = type("L", (), {"weight": self._params[0]})()

        def parameters(self):
            return iter(self._params)

        def __call__(self, ids):
            import torch
            outer = self.outer
            B, T = ids.shape
            vocab = outer.vocab
            # read the carry state from the SHARED cache (fla-Cache-like) and compute a CONTENT signal:
            # the correlation of the HIGH heads' state with a FIXED readout template. This depends on
            # the singular VECTORS, so spectrum-noise (which randomizes vectors but keeps sigma) and
            # top-r truncation both degrade it, while energy alone does NOT recover it -> the fake
            # bundle faithfully exercises the int-2/int-3 content dissociation, not just magnitude.
            cache = outer.shared.get("cache")
            content = 0.0
            if cache is not None and len(cache) > 0:
                try:
                    rs = cache[0]["recurrent_state"]
                    if rs is not None:
                        hi = rs[0, 1::2].float()                    # HIGH (odd) heads (nh_hi, d, d)
                        tpl = outer.readout_template                 # fixed (d, d), unit-norm
                        # normalized alignment with the template: cos-like, in [0, ~1]
                        num = float((hi * tpl).sum().item())
                        den = float(hi.pow(2).sum().sqrt().item()) + 1e-6
                        content = max(0.0, num / den)
                except Exception:
                    content = 0.0
            out = torch.zeros(B, T, vocab)
            for i in range(T):
                tid = int(ids[0, i].item()) % vocab
                # correct next-token logit is boosted by the CONTENT alignment; destroying content
                # (top-r, spectrum-noise, zero) lowers it -> PPL rises. Low heads carry ~no content.
                boost = 6.0 + 8.0 * min(1.0, content)
                out[0, i, (tid + 1) % vocab] = boost
                out[0, i, (tid + 2) % vocab] = 2.0
            # write/update a fake recurrent_state into the cache so the NEXT segment can be surgeried.
            # HIGH (odd) heads carry a full-rank state ALIGNED with the readout template (genuine
            # capacity); LOW (even) heads carry a low-rank state ORTHOGONAL to it (idle).
            if cache is not None:
                heads = outer.heads
                rs = torch.zeros(1, heads, outer.d, outer.d)
                for h in range(heads):
                    if h % 2 == 1:                                   # HIGH: full-rank, aligned
                        rs[0, h] = outer.readout_template * 3.0
                    else:                                            # LOW: rank-3, orthogonal noise
                        g = np.random.default_rng(1000 + h + int(ids.sum().item()))
                        U = g.standard_normal((outer.d, 3)); V = g.standard_normal((3, outer.d))
                        rs[0, h] = torch.from_numpy((U @ V).astype("float32")) * 0.1
                cache.update(recurrent_state=rs, layer_idx=0, offset=T)
            return out

    def __init__(self, n_layer=1, heads=8, d=32, vocab=64, seed=0):
        import torch
        self.n_layer = n_layer; self.heads = heads; self.d = d; self.vocab = vocab
        self.rng = np.random.default_rng(seed); self.torch = torch
        self.shared = {"cache": None}
        # fixed FULL-RANK unit-norm readout template: HIGH heads store content aligned with it, so
        # top-r truncation (loses trailing singular directions) AND spectrum-noise (randomizes the
        # singular vectors) both reduce the alignment -> the content dissociation is exercised.
        g = np.random.default_rng(4242)
        tpl = g.standard_normal((d, d))
        tpl = tpl / np.linalg.norm(tpl)
        self.readout_template = torch.from_numpy(tpl.astype("float32"))
        self.model = _FakeBundle._Model(self)
        self.base = self

    def _state(self, li, h):
        r = 3 if (h % 2 == 0) else self.d
        U = self.rng.standard_normal((self.d, r)); V = self.rng.standard_normal((r, self.d))
        return self.torch.from_numpy((U @ V).astype("float32"))

    def states(self, ids):
        return {li: self.torch.stack([self._state(li, h) for h in range(self.heads)])
                for li in range(self.n_layer)}

    def logits(self, ids):
        # single-shot origin path: route through the SAME .model() the segmented path uses, with a
        # fresh 1-forward cache, so a 1-segment segmented forward reproduces it EXACTLY (mirrors the
        # real common.Bundle.logits, which also just calls self.model(ids)). This is what makes the
        # S=1 harness-faithfulness control meaningful in the smoke test.
        from fla.models.utils import Cache
        prev = self.shared.get("cache")
        self.shared["cache"] = Cache()
        try:
            return self.model(ids)
        finally:
            self.shared["cache"] = prev


def _smoke(args):
    import torch
    state_surgery._selftest()                 # spectral surgery invariants (rank/nuclear preserved)

    bundle = _FakeBundle()

    class _Tok:
        eos_token_id = None
        def __call__(self, s, add_special_tokens=False):
            class R: pass
            r = R(); r.input_ids = [(ord(c) % 60) + 2 for c in s][:256]; return r
        def decode(self, ids):
            if isinstance(ids, int): ids = [ids]
            return "".join(chr(65 + (int(i) % 26)) for i in ids)
    tok = _Tok()

    # --- classifier: bimodal planted ranks -> balanced low/high split ---
    ids_list = [torch.arange(32, dtype=torch.long).reshape(1, 32) for _ in range(3)]
    import data_stage1
    orig = data_stage1.load_all
    data_stage1.load_all = lambda t, seq_len, n_seq, which: {
        d: (ids_list, {"domain": d, "source": "smoke", "is_fallback": False,
                       "seq_len": 32, "n_seq": 3}) for d in which}
    try:
        cls = head_classifier.classify(bundle, tok, seq_len=32, n_seq=3, seed=0,
                                       require_real_data=False)
    finally:
        data_stage1.load_all = orig
    print("[smoke] classify k=%d low=%d high=%d agree=%.3f" % (
        cls["k"], len(cls["low_heads"]), len(cls["high_heads"]),
        cls["cross_domain_agreement"]["mean"]))
    assert cls["k"] > 0 and len(cls["low_heads"]) == len(cls["high_heads"]) == cls["k"]

    low = [tuple(x) for x in cls["low_heads"]]; high = [tuple(x) for x in cls["high_heads"]]
    all_heads = [(h["layer"], h["head"]) for h in cls["per_head"]]
    rand = draw_random_heads(all_heads, cls["k"], seed=0)
    groups = {"high": high, "low": low, "random": rand}
    assert set(low).isdisjoint(set(high)), "smoke: disjoint groups"
    domain_ids = {d: ids_list for d in head_classifier.NATURAL_DOMAINS}

    surgeon = state_surgery.SegmentedSurgeon(bundle, segment_len=8)   # 32 tok / 8 -> 4 segments

    # --- S=1 control: segmented (1 seg) reproduces single-shot origin ---
    control = run_segment_control(surgeon, bundle, domain_ids, args)

    # --- interventions run end to end on the fake bundle ---
    masker = type("M", (), {"n_layers": 1, "num_heads": 8,
                            "set_mask": lambda self, s: None, "clear": lambda self: None,
                            "kv_reduction": lambda self, s: {"kv_reduction_pct": 0.0}})()
    # int-1 needs ppl_eval.run_ppl on the fake bundle (single-shot); patch masker minimally
    ranked_heads = [(h["layer"], h["head"]) for h in
                    sorted(cls["per_head"], key=lambda x: x.get("agg_norm_rank", 0.0))]
    int1 = run_int1_prune_fraction(bundle, tok, masker, ranked_heads, domain_ids, args)
    int2 = run_int2_topr(surgeon, groups, domain_ids, args)
    int3 = run_int3_noise_ladder(surgeon, groups, domain_ids, args)
    verdict = compute_verdict(int1, int2, int3, control, origin_niah=None)
    print("[smoke] int2 slopes:", verdict["int2_ppl_vs_contentloss_slope"])
    print("[smoke] int3 spectrum delta:", verdict["int3_spectrum_noise_delta_vs_origin"])
    print("[smoke] G3_seed:", verdict["G3_genuine_capacity_seed"])

    out = os.path.join(args.out, "stage3_report_smoke.json")
    os.makedirs(args.out, exist_ok=True)
    json.dump({"classification": cls, "segment_control": control, "verdict": verdict,
               "int2_topr": int2, "int3_noise_ladder": int3}, open(out, "w"), indent=2)
    print("[smoke] wrote", out)
    print("[smoke] OK (surgery invariants, classify disjoint groups, S=1 control, int-1/2/3, verdict)")


if __name__ == "__main__":
    main()
