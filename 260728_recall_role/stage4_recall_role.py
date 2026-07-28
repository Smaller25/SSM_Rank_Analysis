"""Stage 4 driver — GDN HIGH-rank head RECALL-ROLE test (mech-interp induction probe).

Stage 2 showed HIGH-rank KV-state pruning catastrophically raises PPL (~32x), but PPL is a lumped
metric that hides WHICH role is lost. Stage 4 decomposes it with a standard induction-head probe on
REAL text (Olsson et al. 2022): each real passage A (length L) is repeated once -> seq=[A][A]; the
per-token NLL is split into local_bits (first copy, no recall) vs recall_bits (second copy, recall
possible). induction_gain = local_bits - recall_bits (>0 => in-context recall is working).

H(recall-role): HIGH-rank heads are in-context RECALL units, so pruning them collapses
induction_gain SPECIFICALLY on the recall segment, while LOW/random pruning does not. Signature:
    Delta_recall(high) >> Delta_local(high)   (recall-specific, not a uniform PPL hit)     AND
    Delta_recall(high) >> Delta_recall(low/random)                                          .
NULL/reversal (Delta_recall(high) ~ Delta_local(high), or ~ low/random) = HIGH-rank is NOT
recall-specific -> record as a LIMITATION, not a code failure.

Four count-matched conditions (identical k across high/low/random, PIN-4):
  origin  : no masking (baseline for all deltas)
  high    : zero the k HIGH-rank heads' KV state
  low     : zero the k LOW-rank heads' KV state
  random  : zero k randomly-chosen heads (seed governs the draw, PIN-6)

Headroom gate: origin induction_gain must exceed the pre-registered 0.3 bits threshold, else the
probe has no signal to lose -> UNTESTABLE_HEADROOM (same headroom-gate pattern as Stage 2/3 NIAH
floor 0.30).

[PIN-1] gdn2-1.3B, 100B paper-matched ckpt ONLY (10B rejected by resolve_and_assert_ckpt), bf16,
        fused_recurrent, Config.from_name + strict=False. n_layer=18, mixer num_heads==num_v_heads==16
        -> 288 heads (config n_head=18 is the attention field, unrelated to the mixer; HeadMasker
        reads mixer.num_heads=16 at runtime). Provenance logs num_heads/num_v_heads/total on the
        first real run (Stage3 major-2 resolution). [PIN-2] threshold-rank eps=1e-4, theta_R=0.5
        (head_classifier, re-derived). [PIN-4] KV-state v-zeroing (head_mask.HeadMasker). [PIN-5]
        3 natural domains from data_cache/ real texts, 2x repeated (--require-real-data -> fallback
        hard-fails). [PIN-6] seeds 0/1/2 govern passage sampling + random-mask draw.

Caveat (echo verbatim, PREREG): paper's 93.8/46.9/90.6/38.9% are Qwen3-Next (48-layer, POST-TRAINED)
OBSERVATIONS adopted as reproduction TARGET lines, NOT pass standards for 18-layer gdn2; falling
short = generalization/role limitation to record, not code failure.

Run (greenbeard SLURM, 100B checkpoint):
  export GDN2_CKPT_PATH=/home/sohyung/models/gdn2_1.3B_100b.pth
  export TRITON_CACHE_DIR=/home/sohyung/.triton_cache HF_HUB_DISABLE_XET=1
  python stage4_recall_role.py --seed 0 --require-real-data --out results/recall_role_100b_seed0

Smoke (CPU, no model — synthetic bundle; exercises probe build + segment split + verdict wiring):
  python stage4_recall_role.py --smoke
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

import head_classifier          # noqa: E402  (Stage 2, reused unmodified)
import head_mask                # noqa: E402  (Stage 2, reused unmodified)
import induction_probe          # noqa: E402  (Stage 4 new: probe build + segment-split PPL)

CROSS_DOMAIN_AGREEMENT_MIN = 0.90   # sanity gate (Stage 1 was 0.971); below -> classification untrusted
HEADROOM_GAIN_BITS = 0.30           # [PREREG] origin induction_gain must exceed this or UNTESTABLE
RECALL_SPECIFIC_MARGIN = 1e-9       # ">>": we require strict inequality (aggregate applies the real margin)
PREREG_CAVEAT = ("paper 93.8/46.9/90.6 NIAH + KV 38.9% are Qwen3-Next (48-layer, POST-TRAINED) "
                 "OBSERVATIONS adopted as reproduction TARGET lines, NOT pass standards for 18-layer "
                 "gdn2; falling short = generalization/role limitation to record, not code failure.")


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


def draw_random_heads(all_heads, k, seed):
    """k randomly-chosen heads (the control), reproducible per seed (governs the draw, PIN-6)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(all_heads), size=k, replace=False)
    return [all_heads[i] for i in sorted(idx.tolist())]


def reset_shared_cache(bundle):
    """Purge the recurrent-state cache left behind by any prior bundle.states() call BEFORE any
    logits-based eval (identical fix to Stage 2). head_classifier.classify() calls bundle.states()
    which sets shared["cache"] = Cache() and never resets it; the patched GatedDeltaNet2.forward then
    feeds that stale Cache as the initial recurrent_state for EVERY subsequent forward. Resetting to
    None makes the patched forward take the clean stateless branch, so every token_nll_bits() call is
    deterministic and uncontaminated across conditions. We reach the canonical common.Bundle via
    bundle.base.shared (Stage1Bundle wraps it)."""
    base = getattr(bundle, "base", bundle)
    shared = getattr(base, "shared", None)
    if isinstance(shared, dict) and shared.get("cache") is not None:
        shared["cache"] = None


def evaluate_condition(bundle, masker, mask_set, domain_seqs, passage_len):
    """Set the mask, run the segment-split induction probe, return the metrics dict. Mask reset by
    the caller between conditions."""
    reset_shared_cache(bundle)   # purge stale recurrent state from classification / previous cond
    masker.set_mask(mask_set)
    kv = masker.kv_reduction(mask_set)   # TRUE KV-state pruning: memory footprint freed
    ind = induction_probe.run_induction(bundle, domain_seqs, passage_len)
    masker.clear()
    macro = ind["_macro"]
    return {"n_masked": len(mask_set), "kv_reduction": kv,
            "induction": ind,
            "local_bits": macro["local_bits"],
            "recall_bits": macro["recall_bits"],
            "induction_gain": macro["induction_gain"]}


def compute_verdict(conds):
    """recall-role verdict from the four conditions of ONE seed.

    Baseline = origin. For each pruned condition:
      Delta_recall = recall_bits(cond) - recall_bits(origin)   (rise in recall-segment NLL)
      Delta_local  = local_bits(cond)  - local_bits(origin)    (rise in local-segment NLL)
    HIGH-rank = recall unit requires the recall-specific signature:
      (a) Delta_recall(high) > Delta_local(high)                  (damage concentrated on RECALL, not
                                                                    a uniform PPL hit)
      (b) Delta_recall(high) > Delta_recall(low) AND > Delta_recall(random)   (HIGH-specific vs
                                                                    the low/random controls)
    Headroom gate: if origin induction_gain <= 0.30 bits the probe cannot show recall loss ->
    UNTESTABLE_HEADROOM (verdict withheld). NULL/reversal (either (a) or (b) fails) = HIGH-rank NOT
    recall-specific -> LIMITATION, not a code failure.
    """
    o_local = conds["origin"]["local_bits"]
    o_recall = conds["origin"]["recall_bits"]
    o_gain = conds["origin"]["induction_gain"]

    def d_recall(c): return conds[c]["recall_bits"] - o_recall
    def d_local(c):  return conds[c]["local_bits"] - o_local

    dr = {c: d_recall(c) for c in ("high", "low", "random")}
    dl = {c: d_local(c) for c in ("high", "low", "random")}

    # headroom gate (pre-registered): need origin recall headroom to detect a collapse.
    headroom_ok = bool(np.isfinite(o_gain) and o_gain > HEADROOM_GAIN_BITS)

    m = RECALL_SPECIFIC_MARGIN
    recall_specific_high = bool(dr["high"] > dl["high"] + m)                 # (a)
    high_vs_low = bool(dr["high"] > dr["low"] + m)
    high_vs_random = bool(dr["high"] > dr["random"] + m)
    high_specific = bool(high_vs_low and high_vs_random)                     # (b)
    recall_role_seed = bool(headroom_ok and recall_specific_high and high_specific)

    if not headroom_ok:
        status = "UNTESTABLE_HEADROOM"
    elif recall_role_seed:
        status = "RECALL_ROLE_SUPPORTED"
    else:
        status = "NULL_OR_REVERSAL"   # LIMITATION (not a code failure)

    return {
        "origin_local_bits": o_local, "origin_recall_bits": o_recall,
        "origin_induction_gain": o_gain,
        "headroom_threshold_bits": HEADROOM_GAIN_BITS,
        "headroom_ok(origin_gain>thr)": headroom_ok,
        "delta_recall_vs_origin": dr, "delta_local_vs_origin": dl,
        "induction_gain_by_cond": {c: conds[c]["induction_gain"] for c in
                                   ("origin", "high", "low", "random")},
        "recall_specific_high(dRecall>dLocal)": recall_specific_high,
        "high_vs_low(dRecall_high>dRecall_low)": high_vs_low,
        "high_vs_random(dRecall_high>dRecall_random)": high_vs_random,
        "high_recall_specific(vs low & random)": high_specific,
        "recall_role_seed": recall_role_seed,
        "status": status,
        "note": ("recall-role confirmed only on aggregate across >=3 seeds. Requires origin "
                 "induction_gain > %.2f bits (headroom), Delta_recall(high) > Delta_local(high) "
                 "(recall-specific), and Delta_recall(high) > Delta_recall(low/random) (HIGH-specific)."
                 % HEADROOM_GAIN_BITS),
    }


def run_seed(bundle, tok, masker, args, out_dir):
    """Full Stage 4 for one seed: classify -> 4 conditions x segment-split probe -> verdict.
    Incremental per-condition flush + --resume."""
    import data_stage1
    set_seed(args.seed)

    # ---- (1) RE-DERIVE head classification (PIN-2); gate on cross-domain agreement (Stage 2 reuse) ----
    cls_path = os.path.join(out_dir, "head_classification.json")
    if args.resume and os.path.isfile(cls_path):
        with open(cls_path) as f:
            cls = json.load(f)
        print(f"  [classify] RESUMED from {cls_path}", flush=True)
    else:
        cls = head_classifier.classify(
            bundle, tok, seq_len=args.cls_seq_len, n_seq=args.n_seq, seed=args.seed,
            require_real_data=args.require_real_data)
        with open(cls_path, "w") as f:
            json.dump(cls, f, indent=2)
    agree = cls["cross_domain_agreement"]["mean"]
    print(f"  [classify] k={cls['k']} n_heads={cls['n_heads_total']} "
          f"cross_domain_agreement={agree:.3f} (gate >= {CROSS_DOMAIN_AGREEMENT_MIN})", flush=True)
    agreement_ok = bool(np.isfinite(agree) and agree >= CROSS_DOMAIN_AGREEMENT_MIN)
    if not agreement_ok:
        print(f"  [WARN] cross-domain agreement {agree:.3f} < {CROSS_DOMAIN_AGREEMENT_MIN} -> "
              f"head sets UNTRUSTED; results flagged.", flush=True)

    all_heads = [(h["layer"], h["head"]) for h in cls["per_head"]]
    low = [tuple(x) for x in cls["low_heads"]]
    high = [tuple(x) for x in cls["high_heads"]]
    k = cls["k"]
    rand = draw_random_heads(all_heads, k, seed=args.seed)   # PIN-6: seed governs the draw
    assert len(low) == len(high) == len(rand) == k, (
        f"count-match violated: |low|={len(low)} |high|={len(high)} |rand|={len(rand)} k={k}")
    # low and high head sets MUST be disjoint or the low-vs-high contrast axis is contaminated (PIN-2).
    assert set(low).isdisjoint(set(high)), (
        f"low/high overlap ({len(set(low) & set(high))} shared heads) -> contrast axis contaminated.")
    mask_sets = {"origin": set(), "high": set(high), "low": set(low), "random": set(rand)}

    # ---- shared induction probes built ONCE per seed (same across conditions; only the mask changes) ----
    probes = induction_probe.build_induction_seqs(
        tok, passage_len=args.passage_len, n_passages=args.n_passages,
        domains=induction_probe.NATURAL_DOMAINS, seed=args.seed,
        require_real_data=args.require_real_data)
    domain_seqs = {d: seqs for d, (seqs, _m) in probes.items()}
    probe_meta = {d: m for d, (_seqs, m) in probes.items()}

    # ---- (2)+(3) evaluate each condition; incremental flush per condition ----
    conds = {}
    for cname in ("origin", "high", "low", "random"):
        cpath = os.path.join(out_dir, f"cond_{cname}.json")
        if args.resume and os.path.isfile(cpath):
            with open(cpath) as f:
                conds[cname] = json.load(f)
            print(f"  [cond {cname}] RESUMED from {cpath}", flush=True)
            continue
        t0 = time.time()
        print(f"\n  === condition {cname} (mask {len(mask_sets[cname])} heads) ===", flush=True)
        res = evaluate_condition(bundle, masker, mask_sets[cname], domain_seqs, args.passage_len)
        res["minutes"] = round((time.time() - t0) / 60, 2)
        res["masked_heads"] = [list(h) for h in sorted(mask_sets[cname])]
        conds[cname] = res
        with open(cpath, "w") as f:
            json.dump(res, f, indent=2)
        print(f"  [cond {cname}] local={res['local_bits']:.3f} recall={res['recall_bits']:.3f} "
              f"gain={res['induction_gain']:.3f} bits  ({res['minutes']} min) [flushed]", flush=True)

    verdict = compute_verdict(conds)
    return {
        "seed": args.seed,
        "classification": {k2: cls[k2] for k2 in
                           ("theta_R", "eps_threshold_rank", "n_heads_total", "k",
                            "low_heads", "high_heads", "cross_domain_agreement",
                            "theta_vs_bottomk_mismatch")},
        "cross_domain_agreement_ok": agreement_ok,
        "conditions": {c: {kk: conds[c][kk] for kk in
                           ("n_masked", "local_bits", "recall_bits", "induction_gain",
                            "masked_heads")}
                       for c in conds},
        "conditions_full": conds,
        "probe_meta": probe_meta,
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
    ap.add_argument("--out", default=os.path.join(_HERE, "results", "recall_role"))
    ap.add_argument("--ckpt",
                    default=os.environ.get("GDN2_CKPT_PATH", "/home/sohyung/models/gdn2_1.3B_100b.pth"),
                    help="[PIN-1] paper-matched 100B checkpoint (95B accepted; 10B REJECTED by loader)")
    ap.add_argument("--n-seq", type=int, default=16, dest="n_seq",
                    help="[PIN-5] seqs per domain for head classification (>=16)")
    ap.add_argument("--cls-seq-len", type=int, default=2048, dest="cls_seq_len",
                    help="seq len for head classification (matches Stage 2)")
    ap.add_argument("--passage-len", type=int, default=1024, dest="passage_len",
                    help="[PIN-5] real passage A length L; probe seq = [A][A] has length 2L "
                         "(1024 -> 2048, within block_size 4096) FROZEN")
    ap.add_argument("--n-passages", type=int, default=16, dest="n_passages",
                    help="[PIN-5] induction passages per domain (>=16)")
    ap.add_argument("--seed", type=int, default=0, help="[PIN-6] governs random-mask draw + sampling")
    ap.add_argument("--require-real-data", action="store_true", dest="require_real_data",
                    help="[PIN-5] hard-fail if any natural domain falls back to synthetic data")
    ap.add_argument("--resume", action="store_true",
                    help="skip classification/conditions whose JSON already exists in --out")
    ap.add_argument("--smoke", action="store_true", help="CPU synthetic-bundle smoke (no model)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    set_seed(args.seed)

    if args.smoke:
        return _smoke(args)

    import loader_gdn2
    import common as gdn2_common
    from transformers import AutoTokenizer
    print(f"[load] gdn2-1.3B ckpt={args.ckpt} config={loader_gdn2.CONFIG_NAME} seed={args.seed} [PIN-1/6]",
          flush=True)
    bundle = loader_gdn2.load(checkpoint_path=args.ckpt)      # resolve_and_assert_ckpt inside
    tok = AutoTokenizer.from_pretrained(gdn2_common.TOKENIZER)
    masker = head_mask.HeadMasker(bundle)
    # [PIN-1] provenance: log num_heads/num_v_heads/total (288 re-confirm; Stage3 major-2 resolution)
    head_prov = {"n_layers": masker.n_layers, "num_heads": masker.num_heads,
                 "num_v_heads": masker.num_v_heads,
                 "total_heads": masker.n_layers * masker.num_heads}
    print(f"[mask] TRUE KV-state pruning: v_conv1d hooks on {masker.n_layers} layers x "
          f"{masker.num_heads} heads (num_v_heads={masker.num_v_heads}) = "
          f"{head_prov['total_heads']} heads (zero head value -> S_h=0, frees KV state) [PIN-1]",
          flush=True)

    report = run_seed(bundle, tok, masker, args, args.out)
    report["prereg_caveat"] = PREREG_CAVEAT
    report["head_provenance"] = head_prov
    report["paper_target_lines"] = {"model": "Qwen3-Next (48-layer, post-trained)",
                                    "origin_niah": 93.8, "prune_low_rank_niah": 46.9,
                                    "prune_high_rank_niah": 90.6, "kv_down_pct": 38.9,
                                    "note": "TARGET lines, not pass standards for 18-layer gdn2"}
    report["ckpt_provenance"] = getattr(bundle, "ckpt_provenance", None)
    report["at_capture_mode"] = getattr(bundle, "capture_mode", None)
    report["git_head"] = _git_head()
    report["runtime_versions"] = _runtime_versions()
    report["args"] = vars(args)
    report["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    prov = report.get("ckpt_provenance") or {}
    tag = "%s_%s_seed%d_%s" % (time.strftime("%y%m%d"), prov.get("token_tag", "ckpt"),
                              args.seed, report["git_head"])
    out = os.path.join(args.out, f"stage4_report_{tag}.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(args.out, "stage4_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[verdict] {json.dumps(report['verdict'], indent=2)}")
    print(f"[written] {out}")
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)   # clean exit (datasets prefetch thread can crash at interpreter teardown)


# ------------------------------------------------------------------ CPU smoke (no GPU, no model)
class _FakeMixerBundle:
    """Synthetic bundle exercising head_classifier + segment-split probe wiring on CPU.
      - .states: bimodal per-head rank (half low ~3, half high ~d) so head_classifier splits cleanly
      - .logits: a tiny deterministic distribution that makes the RECALL segment cheaper than the
        LOCAL segment (favors the repeated copy), so induction_gain > 0 and the plumbing is checked.
        NOT the real model — only checks the segment split + verdict wiring, not effect size."""
    def __init__(self, n_layer=4, heads=8, d=32, vocab=64, seed=0):
        import torch
        self.n_layer = n_layer; self.heads = heads; self.d = d; self.vocab = vocab
        self.rng = np.random.default_rng(seed)
        self.torch = torch

    def _state(self, li, h, t):
        r = 3 if (h % 2 == 0) else self.d
        U = self.rng.standard_normal((self.d, r)); V = self.rng.standard_normal((r, self.d))
        return self.torch.from_numpy((U @ V).astype("float32"))

    def states(self, ids):
        t = ids.shape[1]
        return {li: self.torch.stack([self._state(li, h, t) for h in range(self.heads)])
                for li in range(self.n_layer)}

    def logits(self, ids):
        # For probe seq=[A][A] of length 2L: in the SECOND half (recall segment) put a very confident
        # correct-next-token logit; in the FIRST half (local) a weaker one. => recall_bits < local_bits
        # => induction_gain > 0, exercising the segment split + headroom gate + verdict.
        T = ids.shape[1]
        out = self.torch.zeros(1, T, self.vocab)
        for i in range(T):
            nxt = int(ids[0, (i + 1) % T].item()) % self.vocab  # "correct" next token
            conf = 8.0 if i >= T // 2 else 1.0                  # recall half far more confident
            out[0, i, nxt] = conf
        return out


def _smoke(args):
    import torch
    bundle = _FakeMixerBundle()

    class _Tok:
        eos_token_id = None
        def __call__(self, s, add_special_tokens=False):
            class R: pass
            r = R(); r.input_ids = [(ord(c) % 60) + 2 for c in s][:256]; return r
        def decode(self, ids):
            if isinstance(ids, int): ids = [ids]
            return "".join(chr(65 + (int(i) % 26)) for i in ids)
    tok = _Tok()

    # --- classifier: bimodal planted ranks -> balanced low/high split, agreement ~1.0 (Stage 2 reuse) ---
    import data_stage1
    orig_load_all = data_stage1.load_all
    # fake loader RESPECTS seq_len/n_seq (real data_stage1 does) so the probe builder gets length-L
    # passages and can form length-2L probes.
    def _fake_load_all(t, seq_len, n_seq, which):
        ids_list = [torch.randint(2, 60, (1, seq_len), dtype=torch.long) for _ in range(n_seq)]
        return {d: (ids_list, {"domain": d, "source": "smoke", "is_fallback": False,
                               "seq_len": seq_len, "n_seq": n_seq}) for d in which}
    data_stage1.load_all = _fake_load_all
    try:
        cls = head_classifier.classify(bundle, tok, seq_len=64, n_seq=3, seed=0,
                                       require_real_data=False)
        # --- induction probe build: real-text 2x repetition -> seq length 2L ---
        probes = induction_probe.build_induction_seqs(
            tok, passage_len=32, n_passages=3, seed=0, require_real_data=False)
    finally:
        data_stage1.load_all = orig_load_all

    print("[smoke] classify k=%d agreement=%.3f low=%d high=%d" % (
        cls["k"], cls["cross_domain_agreement"]["mean"], len(cls["low_heads"]), len(cls["high_heads"])))
    assert cls["k"] > 0, "smoke: expected non-empty low set"
    assert len(cls["low_heads"]) == len(cls["high_heads"]) == cls["k"], "smoke: count-match"
    assert set(tuple(h) for h in cls["low_heads"]).isdisjoint(
        set(tuple(h) for h in cls["high_heads"])), "smoke: low/high disjoint"
    assert cls["cross_domain_agreement"]["mean"] > 0.9, "smoke: planted bimodal -> high agreement"

    # --- probe shape: seq = [A][A] length 2L; segment split lengths ---
    seqs = probes["wikitext"][0]
    assert seqs[0].shape[1] == 2 * 32, "smoke: probe length must be 2L"
    loc, rec = induction_probe.segment_bits(bundle, seqs[0], passage_len=32)
    assert loc.shape[0] == 32 - 1 and rec.shape[0] == 32 - 1, "smoke: segment lengths L-1 each"
    ind = induction_probe.run_induction(bundle, {"wikitext": seqs}, passage_len=32)
    assert np.isfinite(ind["_macro"]["induction_gain"]), "smoke: finite induction_gain"
    assert ind["_macro"]["induction_gain"] > 0, "smoke: planted recall-cheaper -> gain > 0"

    # --- verdict logic: planted case where HIGH-rank pruning collapses recall specifically ---
    def cond(local, recall):
        return {"local_bits": local, "recall_bits": recall, "induction_gain": local - recall}
    fake = {"origin": cond(6.0, 5.0),          # gain 1.0 bits > 0.30 headroom
            "high":   cond(6.1, 8.5),          # recall collapses (dRecall=3.5), local ~ (dLocal=0.1)
            "low":    cond(6.2, 5.3),          # recall barely moves (dRecall=0.3)
            "random": cond(6.1, 5.4)}          # recall barely moves (dRecall=0.4)
    v = compute_verdict(fake)
    assert v["headroom_ok(origin_gain>thr)"], "smoke: origin gain 1.0 > 0.30 headroom"
    assert v["recall_specific_high(dRecall>dLocal)"], "smoke: dRecall(high) >> dLocal(high)"
    assert v["high_recall_specific(vs low & random)"], "smoke: dRecall(high) >> low/random"
    assert v["status"] == "RECALL_ROLE_SUPPORTED", "smoke: planted signature must be SUPPORTED"

    # --- headroom gate: origin below threshold -> UNTESTABLE ---
    flat = {"origin": cond(5.0, 4.9), "high": cond(5.0, 6.0),
            "low": cond(5.0, 4.95), "random": cond(5.0, 4.95)}
    vf = compute_verdict(flat)
    assert vf["status"] == "UNTESTABLE_HEADROOM", "smoke: origin gain 0.1 < 0.30 -> UNTESTABLE"

    out = os.path.join(args.out, "stage4_report_smoke.json")
    os.makedirs(args.out, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"classification": cls, "induction": ind, "verdict": v,
                   "headroom_gate": vf}, f, indent=2)
    print("[smoke] wrote", out)
    print("[smoke] OK (classifier count-match/disjoint + agreement, probe 2L split, gain>0, "
          "verdict recall-role + headroom gate)")


if __name__ == "__main__":
    main()
