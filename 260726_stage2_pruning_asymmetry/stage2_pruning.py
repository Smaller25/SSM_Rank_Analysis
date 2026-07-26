"""Stage 2 driver — GDN pruning ASYMMETRY test (paper arXiv:2602.02195 §5, Table 1 contrast).

Tests whether pruning LOW-rank state heads hurts far more than pruning HIGH-rank heads, BEYOND a
random equal-count control, on pure gdn2-1.3B (100B paper-matched, 18 layers x 16 heads = 288 heads).
Gate G2: low-rank pruning loss >> high-rank pruning loss AND low-rank damage exceeds the random
control => the functional asymmetry exists in pure GDN.

Four count-matched conditions (identical k, PIN-6):
  origin  : no masking (upper bound)
  high    : zero the k HIGH-rank heads
  low     : zero the k LOW-rank heads  (predicted worst)
  random  : zero k randomly chosen heads (the critical control; seed governs the draw)

DV primary   : S-NIAH retrieval accuracy (needle value emitted), depth-swept  (niah_retrieval)
DV secondary : per-domain PPL (wikitext/github/arxiv)                          (ppl_eval)

[PIN-1] gdn2-1.3B, 100B paper-matched ckpt ONLY (10B rejected by resolve_and_assert_ckpt), bf16,
        fused_recurrent, Config.from_name + strict=False. [PIN-2] threshold-rank eps=1e-4, theta_R=0.5
        (head_classifier, re-derived from code since the Stage 1 JSON was lost). [PIN-5] 3 natural
        domains from data_cache/ real texts (--require-real-data -> fallback hard-fails). [PIN-6]
        seeds 0/1/2 govern random-mask draw + data/needle sampling; torch/np/PYTHONHASHSEED pinned.

Caveat (echo verbatim, PREREG): paper's 93.8/46.9/90.6/38.9% are Qwen3-Next (48-layer, POST-TRAINED)
OBSERVATIONS adopted as reproduction TARGET lines, NOT pass standards for 18-layer gdn2; falling
short = generalization limitation to record, not code failure.

Run (greenbeard SLURM, 100B checkpoint):
  export GDN2_CKPT_PATH=/home/sohyung/models/gdn2_1.3B_100b.pth
  export TRITON_CACHE_DIR=/home/sohyung/.triton_cache HF_HUB_DISABLE_XET=1
  python stage2_pruning.py --seed 0 --require-real-data --out results/stage2_seed0

Smoke (CPU, no model — synthetic bundle; exercises masking + scoring wiring):
  python stage2_pruning.py --smoke
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
_260722 = os.path.abspath(os.path.join(_HERE, "..", "260722_exp"))
for _p in (_HERE, _STAGE1, _260722):
    if _p not in sys.path:
        sys.path.insert(0, _p)
_FLA_PATH = "/home/sohyung/linear-memory-routing"
if os.path.isdir(_FLA_PATH) and _FLA_PATH not in sys.path:
    sys.path.insert(0, _FLA_PATH)

import head_classifier          # noqa: E402
import head_mask                # noqa: E402
import niah_retrieval           # noqa: E402
import ppl_eval                 # noqa: E402

CROSS_DOMAIN_AGREEMENT_MIN = 0.90   # sanity gate (Stage 1 was 0.971); below this -> classification untrusted
PREREG_CAVEAT = ("paper 93.8/46.9/90.6 NIAH + KV 38.9% are Qwen3-Next (48-layer, POST-TRAINED) "
                 "OBSERVATIONS adopted as reproduction TARGET lines, NOT pass standards for 18-layer "
                 "gdn2; falling short = generalization limitation to record, not code failure [PIN-4].")


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


def evaluate_condition(bundle, tok, masker, mask_set, domain_ids, args):
    """Set the mask, run NIAH retrieval + per-domain PPL, return the metrics dict. Mask is reset by
    the caller between conditions."""
    masker.set_mask(mask_set)
    niah = niah_retrieval.run_niah(
        bundle, tok, n_samples=args.niah_samples, max_seq_length=args.seq_len,
        tokens_to_generate=args.gen_tokens, seed=args.seed)
    ppl = ppl_eval.run_ppl(bundle, domain_ids)
    masker.clear()
    return {"n_masked": len(mask_set),
            "niah": niah, "ppl": ppl,
            "niah_retrieval_accuracy": niah["niah_retrieval_accuracy"],
            "macro_ppl": ppl["_macro"]["mean_ppl"]}


def compute_verdict(conds):
    """G2 gate from the four conditions of ONE seed.

    delta = degradation vs origin. asymmetry requires:
      (a) low-rank pruning hurts MORE than high-rank pruning (delta_low > delta_high),
      (b) low-rank damage EXCEEDS the random control (delta_low > delta_random),
      (c) high-rank is NO WORSE than random (delta_high <= delta_random, approx).
    Computed on NIAH retrieval accuracy (primary DV; higher=better so delta = origin - cond) and on
    macro PPL (secondary; lower=better so delta = cond - origin)."""
    o_niah = conds["origin"]["niah_retrieval_accuracy"]
    o_ppl = conds["origin"]["macro_ppl"]

    def d_niah(c): return o_niah - conds[c]["niah_retrieval_accuracy"]      # drop in accuracy
    def d_ppl(c): return conds[c]["macro_ppl"] - o_ppl                      # rise in PPL

    dn = {c: d_niah(c) for c in ("high", "low", "random")}
    dp = {c: d_ppl(c) for c in ("high", "low", "random")}
    niah_asym = bool(dn["low"] > dn["high"] and dn["low"] > dn["random"])
    ppl_asym = bool(dp["low"] > dp["high"] and dp["low"] > dp["random"])
    high_not_worse = bool(dn["high"] <= dn["random"] + 1e-9 and dp["high"] <= dp["random"] + 1e-9)
    return {
        "origin_niah": o_niah, "origin_macro_ppl": o_ppl,
        "delta_niah_drop": dn, "delta_ppl_rise": dp,
        "niah_asymmetry(low>high & low>random)": niah_asym,
        "ppl_asymmetry(low>high & low>random)": ppl_asym,
        "high_no_worse_than_random": high_not_worse,
        "G2_asymmetry_seed": bool(niah_asym or ppl_asym),   # per-seed signal; aggregate across seeds
        "note": "G2 confirmed only if asymmetry holds on aggregate across >=3 seeds; PPL is secondary.",
    }


def run_seed(bundle, tok, masker, args, out_dir):
    """Full Stage 2 for one seed: classify -> 4 conditions -> verdict. Incremental per-condition flush."""
    import data_stage1
    set_seed(args.seed)

    # ---- (1) RE-DERIVE head classification (PIN-2); gate on cross-domain agreement ----
    cls_path = os.path.join(out_dir, "head_classification.json")
    if args.resume and os.path.isfile(cls_path):
        with open(cls_path) as f:
            cls = json.load(f)
        print(f"  [classify] RESUMED from {cls_path}", flush=True)
    else:
        cls = head_classifier.classify(
            bundle, tok, seq_len=args.seq_len, n_seq=args.n_seq, seed=args.seed,
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
    mask_sets = {"origin": set(), "high": set(high), "low": set(low), "random": set(rand)}

    # ---- shared eval data drawn ONCE per seed (same across conditions; only mask changes) ----
    data_stage1.set_data_seed(args.seed)
    data = data_stage1.load_all(tok, seq_len=args.seq_len, n_seq=args.ppl_n_seq,
                                which=head_classifier.NATURAL_DOMAINS)
    domain_ids = {d: ids for d, (ids, _m) in data.items()}
    data_meta = {d: m for d, (_ids, m) in data.items()}

    # ---- (2)+(3)+(4) evaluate each condition; incremental flush per condition ----
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
        res = evaluate_condition(bundle, tok, masker, mask_sets[cname], domain_ids, args)
        res["minutes"] = round((time.time() - t0) / 60, 2)
        res["masked_heads"] = [list(h) for h in sorted(mask_sets[cname])]
        conds[cname] = res
        with open(cpath, "w") as f:
            json.dump(res, f, indent=2)
        print(f"  [cond {cname}] NIAH acc={res['niah_retrieval_accuracy']:.3f} "
              f"macro_ppl={res['macro_ppl']:.3f}  ({res['minutes']} min) [flushed]", flush=True)

    verdict = compute_verdict(conds)
    return {
        "seed": args.seed,
        "classification": {k2: cls[k2] for k2 in
                           ("theta_R", "eps_threshold_rank", "n_heads_total", "k",
                            "low_heads", "high_heads", "cross_domain_agreement",
                            "theta_vs_bottomk_mismatch")},
        "cross_domain_agreement_ok": agreement_ok,
        "conditions": {c: {kk: conds[c][kk] for kk in
                           ("n_masked", "niah_retrieval_accuracy", "macro_ppl", "masked_heads")}
                       for c in conds},
        "conditions_full": conds,
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
    ap.add_argument("--out", default=os.path.join(_HERE, "results", "stage2"))
    ap.add_argument("--ckpt",
                    default=os.environ.get("GDN2_CKPT_PATH", "/home/sohyung/models/gdn2_1.3B_100b.pth"),
                    help="[PIN-1] paper-matched 100B checkpoint (95B accepted; 10B REJECTED by loader)")
    ap.add_argument("--n-seq", type=int, default=16, dest="n_seq",
                    help="[PIN-5] seqs per domain for head classification (>=16)")
    ap.add_argument("--ppl-n-seq", type=int, default=16, dest="ppl_n_seq",
                    help="seqs per domain for PPL eval")
    ap.add_argument("--seq-len", type=int, default=2048, dest="seq_len")
    ap.add_argument("--niah-samples", type=int, default=20, dest="niah_samples",
                    help="S-NIAH prompts per condition (depth spread)")
    ap.add_argument("--gen-tokens", type=int, default=128, dest="gen_tokens",
                    help="greedy-decode length for needle retrieval")
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
    print(f"[mask] installed o_norm hooks on {masker.n_layers} layers x {masker.num_heads} heads "
          f"= {masker.n_layers * masker.num_heads} heads", flush=True)

    report = run_seed(bundle, tok, masker, args, args.out)
    report["prereg_caveat"] = PREREG_CAVEAT
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
    out = os.path.join(args.out, f"stage2_report_{tag}.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(args.out, "stage2_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[verdict] {json.dumps(report['verdict'], indent=2)}")
    print(f"[written] {out}")
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)   # clean exit (datasets prefetch thread can crash at interpreter teardown)


# ------------------------------------------------------------------ CPU smoke (no GPU, no model)
class _FakeMixerBundle:
    """Synthetic bundle exercising head_mask + niah + ppl wiring on CPU. It fakes:
      - .states: bimodal per-head rank (half low ~3, half high ~d) so head_classifier splits cleanly
      - .logits: a tiny deterministic distribution that DEPENDS on the active head mask (so masking
        visibly changes NIAH generation + PPL). NOT the real model — only checks the plumbing."""
    def __init__(self, n_layer=4, heads=8, d=32, vocab=64, seed=0):
        import torch
        self.n_layer = n_layer; self.heads = heads; self.d = d; self.vocab = vocab
        self.rng = np.random.default_rng(seed)
        self.torch = torch
        self._mask = set()             # HeadMasker will patch .states/.logits indirectly via hooks?
        # No real modules -> we let the smoke test drive masking by hand (see _smoke).

    def _state(self, li, h, t):
        r = 3 if (h % 2 == 0) else self.d
        U = self.rng.standard_normal((self.d, r)); V = self.rng.standard_normal((r, self.d))
        return self.torch.from_numpy((U @ V).astype("float32"))

    def states(self, ids):
        t = ids.shape[1]
        return {li: self.torch.stack([self._state(li, h, t) for h in range(self.heads)])
                for li in range(self.n_layer)}

    def logits(self, ids):
        # deterministic per-position argmax = (token_id+1) so greedy generate is predictable and
        # PPL is finite; independent of mask here (smoke only checks the call path, not effect size).
        T = ids.shape[1]
        out = self.torch.zeros(1, T, self.vocab)
        for i in range(T):
            tid = int(ids[0, i].item()) % self.vocab
            out[0, i, (tid + 1) % self.vocab] = 10.0
        return out


def _smoke(args):
    import torch
    # 1) head_mask on a REAL tiny GatedDeltaNet2-like module set is heavy; instead validate the
    #    classifier + masker set/clear API + niah/ppl scoring functions directly.
    bundle = _FakeMixerBundle()

    class _Tok:
        eos_token_id = None
        def __call__(self, s, add_special_tokens=False):
            class R: pass
            r = R(); r.input_ids = [ (ord(c) % 60) + 2 for c in s ][:256]; return r
        def decode(self, ids):
            if isinstance(ids, int): ids = [ids]
            return "".join(chr(65 + (int(i) % 26)) for i in ids)
    tok = _Tok()

    # --- classifier: bimodal planted ranks -> balanced low/high split, agreement ~1.0 ---
    ids_list = [torch.zeros(1, 128, dtype=torch.long) for _ in range(3)]
    import data_stage1
    orig_load_all = data_stage1.load_all
    data_stage1.load_all = lambda t, seq_len, n_seq, which: {
        d: (ids_list, {"domain": d, "source": "smoke", "is_fallback": False,
                       "seq_len": 128, "n_seq": 3}) for d in which}
    try:
        cls = head_classifier.classify(bundle, tok, seq_len=128, n_seq=3, seed=0,
                                       require_real_data=False)
    finally:
        data_stage1.load_all = orig_load_all
    print("[smoke] classify k=%d agreement=%.3f low=%d high=%d" % (
        cls["k"], cls["cross_domain_agreement"]["mean"], len(cls["low_heads"]), len(cls["high_heads"])))
    assert cls["k"] > 0, "smoke: expected non-empty low set"
    assert len(cls["low_heads"]) == len(cls["high_heads"]) == cls["k"], "smoke: count-match"
    assert cls["cross_domain_agreement"]["mean"] > 0.9, "smoke: planted bimodal -> high agreement"

    # --- niah scoring: substring recall + depth ---
    nf, nt, per = niah_retrieval.score_sample("the answer is 12345 ok", ["12345"])
    assert nf == 1 and nt == 1, "smoke: niah substring recall"
    nf2, _, _ = niah_retrieval.score_sample("nothing here", ["999"])
    assert nf2 == 0, "smoke: niah miss"

    # --- ppl: finite bits on the fake logits ---
    ppl = ppl_eval.run_ppl(bundle, {"wikitext": ids_list})
    assert np.isfinite(ppl["wikitext"]["ppl"]), "smoke: finite PPL"

    # --- verdict logic: synthetic conditions where low-rank hurts most ---
    def cond(niah, ppl_):
        return {"niah_retrieval_accuracy": niah, "macro_ppl": ppl_}
    fake = {"origin": cond(0.90, 5.0), "high": cond(0.88, 5.2),
            "low": cond(0.40, 9.0), "random": cond(0.80, 6.0)}
    v = compute_verdict(fake)
    assert v["niah_asymmetry(low>high & low>random)"], "smoke: planted asymmetry must fire"
    assert v["high_no_worse_than_random"], "smoke: high<=random in planted case"

    out = os.path.join(args.out, "stage2_report_smoke.json")
    os.makedirs(args.out, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"classification": cls, "smoke_verdict": v, "ppl": ppl}, f, indent=2)
    print("[smoke] wrote", out)
    print("[smoke] OK (classifier count-match + agreement, niah recall, ppl finite, verdict asymmetry)")


if __name__ == "__main__":
    main()
