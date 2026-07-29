"""Stage 3 OPTIONAL int-4 — linear-probe decodability of a task label from per-head state S_h.

Nice-to-have complement to int-2/int-3: if HIGH-rank heads carry GENUINE information (definition-A
rank is used), then a simple LINEAR probe should decode a task/domain label from their flattened
recurrent state S_h better than from LOW-rank heads (whose state is low-dimensional / near-idle). This
is a decodability read on the SAME dissociation the surgeries test causally.

Design (gated behind --probe in the driver, or run standalone):
  - label = the DOMAIN of the sequence (wikitext / github / arxiv) — a task the state must encode to
    predict next tokens well across domains. 3-way logistic regression (or ridge on one-hot), 5-fold
    CV, macro accuracy. NO retraining of the model; the probe is a cheap read-out.
  - features = the FINAL per-head state S_h flattened, concatenated over a head GROUP (high / low /
    random), z-scored. We compare group probe accuracy: HIGH >> LOW would corroborate G3.
  - We reuse bundle.states (the Stage1/2 path) for S_h and head_classifier for the groups.

This is deliberately small and CPU-friendly (sklearn LogisticRegression on a few hundred features x
~48 sequences). It is SECONDARY: PPL surgeries are the primary G3 evidence.

Usage (standalone, on GPU node after model load is cheap for states-only):
  python probe_decodability.py --seed 0 --require-real-data --out results/stage3_100b_seed0
Smoke (CPU): python probe_decodability.py --smoke
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_STAGE1 = os.path.abspath(os.path.join(_HERE, "..", "260725_stage1_rank_stratification"))
_STAGE2 = os.path.abspath(os.path.join(_HERE, "..", "260726_stage2_pruning_asymmetry"))
_260722 = os.path.abspath(os.path.join(_HERE, "..", "legacy", "260722_exp"))
for _p in (_HERE, _STAGE1, _STAGE2, _260722):
    if _p not in sys.path:
        sys.path.insert(0, _p)
_FLA_PATH = "/home/sohyung/linear-memory-routing"
if os.path.isdir(_FLA_PATH) and _FLA_PATH not in sys.path:
    sys.path.insert(0, _FLA_PATH)

import head_classifier          # noqa: E402


def _group_features(states_per_seq, group_heads):
    """Flatten S_h over a head group for each sequence -> (n_seq, n_features). states_per_seq is a
    list of {layer:(heads,dk,dv)} dicts (one per sequence)."""
    feats = []
    for st in states_per_seq:
        vec = []
        for (li, h) in group_heads:
            if li in st and h < st[li].shape[0]:
                vec.append(np.asarray(st[li][h]).reshape(-1))
        feats.append(np.concatenate(vec) if vec else np.zeros(1))
    n = min(len(v) for v in feats)
    return np.stack([v[:n] for v in feats], axis=0)


def _cv_logreg(X, y, seed=0, n_splits=5):
    """Macro-accuracy of a linear logistic-regression probe under stratified CV. Falls back to a
    ridge-classifier if sklearn LogisticRegression is unavailable. Returns mean+-std accuracy."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
    except Exception as e:
        return {"mean": float("nan"), "std": float("nan"), "n": 0,
                "error": f"sklearn unavailable: {type(e).__name__}"}
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    k = int(min(n_splits, counts.min())) if counts.size else 0
    if k < 2:
        return {"mean": float("nan"), "std": float("nan"), "n": 0,
                "error": "too few samples per class for CV"}
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    accs = []
    for tr, te in skf.split(X, y):
        clf = make_pipeline(StandardScaler(with_mean=True, with_std=True),
                            LogisticRegression(max_iter=2000, C=1.0, multi_class="auto"))
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        # macro accuracy = mean per-class recall
        recs = [np.mean(pred[y[te] == c] == c) for c in np.unique(y[te])]
        accs.append(float(np.mean(recs)))
    return {"mean": float(np.mean(accs)), "std": float(np.std(accs)), "n": len(accs),
            "chance": float(1.0 / len(classes))}


def run_probe(bundle, tok, groups, *, seq_len=2048, n_seq=16, seed=0,
              domains=None, require_real_data=True):
    """Decode DOMAIN label from per-head state S_h, per head group. Returns per-group probe accuracy.

    Reuses bundle.states over the natural domains; label = domain index. HIGH >> LOW probe accuracy
    corroborates the surgery result (high-rank state carries decodable task info)."""
    import data_stage1
    domains = domains or head_classifier.NATURAL_DOMAINS
    data_stage1.set_data_seed(seed)
    data = data_stage1.load_all(tok, seq_len=seq_len, n_seq=n_seq, which=list(domains))
    fb = [d for d, (_, m) in data.items() if m.get("is_fallback")]
    if fb and require_real_data:
        raise RuntimeError(f"[PIN-5] domains fell back to synthetic: {fb} (--require-real-data).")

    states_per_seq, labels = [], []
    for di, dom in enumerate(domains):
        ids_list, _ = data[dom]
        for ids in ids_list:
            states_per_seq.append(bundle.states(ids))
            labels.append(di)
    labels = np.asarray(labels)

    out = {}
    for g, heads in groups.items():
        X = _group_features(states_per_seq, [tuple(h) for h in heads])
        out[g] = _cv_logreg(X, labels, seed=seed)
        m = out[g]
        print(f"  [probe {g}] macro_acc={m.get('mean')} +- {m.get('std')} "
              f"(chance {m.get('chance')}, n_features={X.shape[1]})", flush=True)
    hi = out.get("high", {}).get("mean", float("nan"))
    lo = out.get("low", {}).get("mean", float("nan"))
    out["_contrast_high_minus_low"] = (float(hi - lo) if (np.isfinite(hi) and np.isfinite(lo))
                                       else float("nan"))
    out["_note"] = ("linear-probe decodability of DOMAIN from S_h; HIGH>>LOW corroborates that "
                    "high-rank state carries genuine decodable task info (int-4, SECONDARY).")
    return out


def _smoke(args):
    # synthetic: HIGH heads encode the label linearly, LOW heads are noise -> HIGH probe >> LOW.
    rng = np.random.default_rng(0)
    n_per, d = 12, 32
    groups = {"high": [(0, 1), (0, 3)], "low": [(0, 0), (0, 2)], "random": [(0, 4), (0, 5)]}

    class _B:
        n_layer = 1
        def states(self, ids):
            import torch
            lab = int(ids[0, 0].item())
            heads = 8
            S = np.zeros((heads, d, d), dtype="float32")
            for h in range(heads):
                if h in (1, 3):                      # HIGH: label-dependent structured state
                    S[h] = (rng.standard_normal((d, d)) * 0.1) + lab * 2.0
                else:                                # LOW/random: label-independent noise
                    S[h] = rng.standard_normal((d, d)) * 0.1
            return {0: torch.from_numpy(S)}

    class _Tok:
        def __call__(self, s, add_special_tokens=False):
            class R: pass
            r = R(); r.input_ids = [1]; return r

    import data_stage1, torch
    orig = data_stage1.load_all
    def fake_load(t, seq_len, n_seq, which):
        out = {}
        for di, dname in enumerate(which):
            ids_list = [torch.full((1, 8), di, dtype=torch.long) for _ in range(n_per)]
            out[dname] = (ids_list, {"domain": dname, "source": "smoke", "is_fallback": False,
                                     "seq_len": 8, "n_seq": n_per})
        return out
    data_stage1.load_all = fake_load
    try:
        res = run_probe(_B(), _Tok(), groups, seq_len=8, n_seq=n_per, seed=0,
                        require_real_data=False)
    finally:
        data_stage1.load_all = orig
    os.makedirs(args.out, exist_ok=True)
    json.dump(res, open(os.path.join(args.out, "probe_smoke.json"), "w"), indent=2)
    if "error" in res.get("high", {}):
        print("[probe smoke] SKIPPED (sklearn unavailable: %s) — probe is OPTIONAL (int-4); the "
              "sbatch pip-installs scikit-learn defensively. Wiring OK." % res["high"]["error"])
        return
    print("[probe smoke] high=%.3f low=%.3f contrast=%.3f" % (
        res["high"]["mean"], res["low"]["mean"], res["_contrast_high_minus_low"]))
    assert res["high"]["mean"] >= res["low"]["mean"], "smoke: HIGH probe should beat LOW"
    print("[probe smoke] OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(_HERE, "results", "stage3"))
    ap.add_argument("--ckpt",
                    default=os.environ.get("GDN2_CKPT_PATH", "/home/sohyung/models/gdn2_1.3B_100b.pth"))
    ap.add_argument("--seq-len", type=int, default=2048, dest="seq_len")
    ap.add_argument("--n-seq", type=int, default=16, dest="n_seq")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--require-real-data", action="store_true", dest="require_real_data")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.smoke:
        return _smoke(args)

    import loader_gdn2
    import common as gdn2_common
    from transformers import AutoTokenizer
    bundle = loader_gdn2.load(checkpoint_path=args.ckpt)
    tok = AutoTokenizer.from_pretrained(gdn2_common.TOKENIZER)
    cls = head_classifier.classify(bundle, tok, seq_len=args.seq_len, n_seq=args.n_seq,
                                   seed=args.seed, require_real_data=args.require_real_data)
    groups = {"high": cls["high_heads"], "low": cls["low_heads"]}
    all_heads = [(h["layer"], h["head"]) for h in cls["per_head"]]
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(all_heads), size=cls["k"], replace=False)
    groups["random"] = [all_heads[i] for i in sorted(idx.tolist())]
    res = run_probe(bundle, tok, groups, seq_len=args.seq_len, n_seq=args.n_seq, seed=args.seed,
                    require_real_data=args.require_real_data)
    json.dump(res, open(os.path.join(args.out, "probe_decodability.json"), "w"), indent=2)
    print(f"[probe] wrote {os.path.join(args.out, 'probe_decodability.json')}")
    os._exit(0)


if __name__ == "__main__":
    main()
