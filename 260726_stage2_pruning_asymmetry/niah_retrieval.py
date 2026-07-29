"""S-NIAH RETRIEVAL scorer for gdn2 — the real Stage 2 gap.

The existing 260722 niah path (niah_ruler.make_mk_niah -> run.py) only scored needle boundary-F1
(where the needles SIT in the context). Stage 2 needs the RETRIEVAL score: does the model, when
asked, actually EMIT the needle value? So here we:

  1. Build the RULER multikey prompt via niah_ruler.make_mk_niah (verbatim RULER niah_multikey_1),
     which returns (input_text, gt_positions, answers, query, meta). `answers` = the needle
     value(s) the model must recall for the queried key.
  2. Greedy-decode ~tokens_to_generate tokens after the prompt using the lit_gpt GPT forward
     (bundle.logits over the growing sequence; argmax each step). No sampling (seed-invariant given a
     fixed prompt; the seed governs which needles/depths are drawn, per PIN-6).
  3. Score EXACT / SUBSTRING match of each answer value in the decoded text (RULER uses substring
     recall over the string-formatted answer). Report per-sample and the depth (needle position).

Needle-DEPTH sweep: RULER inserts the queried needle at a sampled position; we expose the queried
needle's normalized token-depth (queried needle start / prompt length) so the DV "retrieval accuracy
vs needle depth/position" can be assembled per condition. We draw n_samples prompts per seed with
distinct sub-seeds so depths spread across [0,100].

Reuse: niah_ruler (make_mk_niah). Model call = bundle.logits (common.Bundle / Stage1Bundle).
"""
import os
import re
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_260722 = os.path.abspath(os.path.join(_HERE, "..", "legacy", "260722_exp"))
for _p in (_HERE, _260722):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import niah_ruler   # noqa: E402  (RULER niah_multikey_1, verbatim)


def _reset_shared_cache(bundle):
    """Clear the leftover recurrent-state cache on the canonical common.Bundle (bundle.base.shared),
    so the patched GatedDeltaNet2.forward runs stateless. See stage2_pruning.reset_shared_cache."""
    base = getattr(bundle, "base", bundle)
    shared = getattr(base, "shared", None)
    if isinstance(shared, dict) and shared.get("cache") is not None:
        shared["cache"] = None


@torch.no_grad()
def greedy_generate(bundle, tok, input_text, max_new_tokens=128, stop_newline_run=3):
    """Greedy (argmax) decode after `input_text`. Returns the decoded continuation string.

    Uses the full-forward bundle.logits each step (no KV cache needed for correctness on gdn2; the
    recurrent forward re-runs the prefix). Stops early on EOS or a run of blank lines (the model has
    clearly stopped answering) to save autoregressive steps.

    [fix blocker-1/2] Defensively clear any stale recurrent-state cache left by a prior
    bundle.states() (common.Bundle.states sets shared["cache"]=Cache() and never resets it). With the
    cache None the patched GatedDeltaNet2.forward takes the clean stateless branch, so each full
    re-forward over the growing prefix is correct and independent of whether classification ran."""
    _reset_shared_cache(bundle)
    dev = getattr(bundle, "base", bundle).model.lm_head.weight.device \
        if hasattr(getattr(bundle, "base", bundle).model, "lm_head") else "cuda"
    ids = tok(input_text, add_special_tokens=False).input_ids
    ids = torch.tensor([ids], dtype=torch.long, device=dev)
    eos = getattr(tok, "eos_token_id", None)
    new_ids = []
    blank_run = 0
    for _ in range(max_new_tokens):
        logits = bundle.logits(ids)              # (1, T, vocab)
        nxt = int(logits[0, -1].argmax().item())
        if eos is not None and nxt == eos:
            break
        new_ids.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], dtype=torch.long, device=ids.device)], dim=1)
        # cheap early-stop: decode the last token; count consecutive newlines
        piece = tok.decode([nxt])
        if piece.strip() == "":
            blank_run += 1
            if blank_run >= stop_newline_run:
                break
        else:
            blank_run = 0
    return tok.decode(new_ids)


def _norm(s):
    """RULER-style normalization for substring matching: lowercase, collapse whitespace."""
    return re.sub(r"\s+", " ", str(s).strip().lower())


def score_sample(gen_text, answers):
    """Substring recall of the needle value(s) in the generation (RULER recall metric).

    Returns (n_found, n_total, per_answer_found[list[bool]]). A value counts as retrieved if its
    normalized string appears anywhere in the normalized generation."""
    g = _norm(gen_text)
    found = [(_norm(a) in g) for a in answers]
    return int(sum(found)), len(answers), found


def _needle_depth(input_text, query, tok):
    """Normalized token-depth [0,100] of the QUERIED needle in the prompt (position DV).

    The queried needle is the one containing the query key; find its char index, convert to a token
    index, normalize by prompt length. Returns None if not locatable."""
    m = re.search(re.escape(query), input_text)
    if not m:
        return None
    prefix_tokens = len(tok(input_text[:m.start()], add_special_tokens=False).input_ids)
    total_tokens = max(1, len(tok(input_text, add_special_tokens=False).input_ids))
    return round(100.0 * prefix_tokens / total_tokens, 2)


def run_niah(bundle, tok, *, n_samples=20, max_seq_length=2048, tokens_to_generate=128,
             seed=0, type_needle_v="numbers", num_needle_k=4, num_needle_v=1, num_needle_q=1):
    """S-NIAH retrieval over n_samples prompts (distinct sub-seeds -> depth spread). Returns a dict.

    Per-sample: retrieval hit (all queried values recalled), fractional recall, queried-needle depth.
    Aggregate: mean retrieval accuracy (all-values-correct rate) + mean fractional recall, and a
    depth-bucketed accuracy table so the DV 'accuracy vs needle position' is assemblable per
    condition. Prompt construction is seeded (PIN-6) so the needle set/depths are reproducible."""
    rng = np.random.default_rng(seed)
    samples = []
    for i in range(n_samples):
        sub_seed = int(rng.integers(0, 2 ** 31 - 1))
        input_text, gt, answers, query, meta = niah_ruler.make_mk_niah(
            tok, max_seq_length=max_seq_length, type_needle_v=type_needle_v,
            num_needle_k=num_needle_k, num_needle_v=num_needle_v, num_needle_q=num_needle_q,
            tokens_to_generate=tokens_to_generate, seed=sub_seed)
        gen = greedy_generate(bundle, tok, input_text, max_new_tokens=tokens_to_generate)
        n_found, n_total, per_ans = score_sample(gen, answers)
        depth = _needle_depth(input_text, query, tok)
        samples.append({
            "sub_seed": sub_seed, "query": query, "answers": answers,
            "n_found": n_found, "n_total": n_total,
            "retrieved_all": bool(n_found == n_total and n_total > 0),
            "frac_recall": (n_found / n_total) if n_total else float("nan"),
            "depth_pct": depth, "n_tokens": meta.get("n_tokens"),
            "gen_preview": gen[:160],
        })
    acc = float(np.mean([s["retrieved_all"] for s in samples])) if samples else float("nan")
    frac = float(np.mean([s["frac_recall"] for s in samples])) if samples else float("nan")
    # depth-bucketed accuracy (5 buckets of 20% each)
    buckets = {f"{lo}-{lo+20}": [] for lo in range(0, 100, 20)}
    for s in samples:
        d = s["depth_pct"]
        if d is None:
            continue
        lo = min(80, int(d // 20) * 20)
        buckets[f"{lo}-{lo+20}"].append(s["retrieved_all"])
    depth_acc = {b: (float(np.mean(v)) if v else None) for b, v in buckets.items()}
    return {
        "niah_retrieval_accuracy": acc,          # DV primary: all queried values correct (0..1)
        "niah_frac_recall": frac,                # softer: fraction of values recalled
        "depth_bucketed_accuracy": depth_acc,    # DV: accuracy vs needle position
        "n_samples": len(samples),
        "config": {"max_seq_length": max_seq_length, "tokens_to_generate": tokens_to_generate,
                   "type_needle_v": type_needle_v, "num_needle_k": num_needle_k,
                   "num_needle_v": num_needle_v, "num_needle_q": num_needle_q, "seed": seed},
        "samples": samples,
    }
