"""Stage 4 induction probe — build repeated-sequence probes from REAL text and split the per-token
NLL into a LOCAL (first occurrence) vs a RECALL (second occurrence) segment.

Standard mech-interp induction-head probe (Olsson et al. 2022 "In-context Learning and Induction
Heads"; Elhage et al. 2021 "A Mathematical Framework for Transformer Circuits"), adapted to the SSM
KV-state setting. The ONLY new logic here is (a) repeating each real passage A once to form the probe
sequence seq = [A][A] and (b) the segment index masks; the loader, tokenizer, and per-token NLL
primitive are all REUSED unmodified (data_stage1, analysis.token_nll_bits) — NO synthetic /
distribution replacement (PIN-5: induction is real-text 2x repetition, not a distribution swap).

Definitions (ids A has length L; probe = concat(A, A) has length 2L):
  local  segment = positions 0 .. L-1   (A's first appearance; NO in-context recall possible)
  recall segment = positions L .. 2L-1  (A's second appearance; in-context recall IS possible)

token_nll_bits(bundle, seq) returns per-token surprisal for target = ids[:, 1:], so bits has length
2L-1 and bits[i] is the NLL of predicting token seq[i+1] from prefix seq[:i+1]. Therefore:
  local_bits  = mean(bits[0 : L-1])   -> predicts tokens seq[1..L-1]  (all inside the FIRST copy)
  recall_bits = mean(bits[L : 2L-1])  -> predicts tokens seq[L+1..2L-1] (all inside the SECOND copy)
The boundary token at index L-1 (predicting seq[L], the FIRST token of the second copy) is DROPPED
from both segments: its target is the copy seam, neither purely local nor a within-copy recall, so it
is excluded to keep the two segments clean. induction_gain = local_bits - recall_bits (>0 => the
model spends fewer bits on the repeated copy, i.e. in-context recall is working).

Reuse ONLY (no re-implementation): data_stage1 (loader + real texts), analysis.token_nll_bits.
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_STAGE1 = os.path.abspath(os.path.join(_HERE, "..", "260725_stage1_rank_stratification"))
_260722 = os.path.abspath(os.path.join(_HERE, "..", "260722_exp"))
for _p in (_HERE, _STAGE1, _260722):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import analysis   # noqa: E402  (token_nll_bits, canonical NLL primitive — DO NOT re-implement)

NATURAL_DOMAINS = ["wikitext", "github", "arxiv"]   # [PIN-5] SAME 3 natural domains as Stage 1/2


def build_induction_seqs(tok, passage_len=1024, n_passages=16, domains=NATURAL_DOMAINS,
                         seed=0, require_real_data=True):
    """Load real passages A (length L) per domain, return {domain: (list[seq[1,2L]], meta)}.

    Each real passage A of length L=passage_len is repeated once -> seq = concat(A, A) of length 2L.
    We reuse data_stage1.load_all to draw the passages (seed-dependent sampling, PIN-6); the loader
    already tokenizes real text from data_cache/ and hard-flags synthetic fallback in meta.
    """
    import data_stage1
    import torch
    data_stage1.set_data_seed(seed)   # [PIN-6] seed governs passage sampling
    # load_all cuts the corpus into n_seq chunks of seq_len tokens -> here each chunk IS a passage A.
    data = data_stage1.load_all(tok, seq_len=passage_len, n_seq=n_passages, which=list(domains))

    fb = [d for d, (_, m) in data.items()
          if m.get("is_fallback") or m.get("source") == "fallback"]
    if fb:
        msg = f"[PIN-5] induction passages fell back to SYNTHETIC data for domains: {fb}."
        if require_real_data:
            raise RuntimeError(msg + " (--require-real-data set -> INVALID, aborting).")
        print("  [WARN] " + msg + " Induction probe UNTRUSTWORTHY.", flush=True)

    out = {}
    for dom, (ids_list, meta) in data.items():
        seqs = []
        for ids in ids_list:                          # ids: [1, L] real passage A
            L = ids.shape[1]
            seq = torch.cat([ids, ids], dim=1)        # [1, 2L] = [A][A] (real-text 2x repetition)
            seqs.append(seq)
        m = dict(meta)
        m.update({"passage_len": passage_len, "seq_len_2L": 2 * passage_len,
                  "n_passages": len(seqs), "repetition": 2})
        out[dom] = (seqs, m)
        print(f"  [induction] {dom} ({meta['source']}): {len(seqs)} passages "
              f"L={passage_len} -> 2L={2 * passage_len} tok", flush=True)
    return out


def segment_bits(bundle, seq, passage_len):
    """Per-token NLL bits on one probe seq=[A][A], split into (local_bits, recall_bits).

    Uses analysis.token_nll_bits (canonical, length 2L-1; bits[i] predicts seq[i+1]).
    local segment  = bits[0 : L-1]   (targets seq[1..L-1], first copy)
    recall segment = bits[L : 2L-1]  (targets seq[L+1..2L-1], second copy)
    The seam token bits[L-1] (target seq[L]) is excluded from both. Returns the per-token arrays so
    the caller can pool across passages before averaging (equal weight per token, like ppl_eval).
    """
    L = int(passage_len)
    bits = np.asarray(analysis.token_nll_bits(bundle, seq), dtype=float)   # length 2L-1
    assert bits.shape[0] == 2 * L - 1, (
        f"segment_bits: expected 2L-1={2 * L - 1} per-token bits, got {bits.shape[0]} "
        f"(seq len {seq.shape[1]}, passage_len {L})")
    local = bits[0:L - 1]        # first-copy predictions (no recall available)
    recall = bits[L:2 * L - 1]   # second-copy predictions (recall available); drops seam bits[L-1]
    return local, recall


def induction_bits(bundle, seqs, passage_len):
    """Pool local/recall per-token bits over all passages of a domain, return the summary dict.

    Pools every token (concatenate across passages, then mean) so longer passages weigh more per the
    ppl_eval convention (equal weight per TOKEN, not per passage). induction_gain = local - recall.
    """
    loc_all, rec_all = [], []
    for seq in seqs:
        loc, rec = segment_bits(bundle, seq, passage_len)
        loc_all.append(loc)
        rec_all.append(rec)
    loc = np.concatenate(loc_all) if loc_all else np.array([])
    rec = np.concatenate(rec_all) if rec_all else np.array([])
    local_bits = float(np.mean(loc)) if loc.size else float("nan")
    recall_bits = float(np.mean(rec)) if rec.size else float("nan")
    gain = (local_bits - recall_bits) if (np.isfinite(local_bits) and np.isfinite(recall_bits)) \
        else float("nan")
    return {
        "local_bits": local_bits,
        "recall_bits": recall_bits,
        "induction_gain": gain,
        "n_passages": len(seqs),
        "n_local_tokens": int(loc.size),
        "n_recall_tokens": int(rec.size),
    }


def run_induction(bundle, domain_seqs, passage_len):
    """Segment-split PPL over a {domain: seqs} mapping. Returns {domain: {...}} + macro mean.

    Mirrors ppl_eval.run_ppl's per-domain + _macro aggregation pattern, but the DVs are the
    local/recall/gain triple instead of a single PPL.
    """
    out = {}
    for dom, seqs in domain_seqs.items():
        out[dom] = induction_bits(bundle, seqs, passage_len)
    locs = [out[d]["local_bits"] for d in out if np.isfinite(out[d]["local_bits"])]
    recs = [out[d]["recall_bits"] for d in out if np.isfinite(out[d]["recall_bits"])]
    gains = [out[d]["induction_gain"] for d in out if np.isfinite(out[d]["induction_gain"])]
    out["_macro"] = {
        "local_bits": float(np.mean(locs)) if locs else float("nan"),
        "recall_bits": float(np.mean(recs)) if recs else float("nan"),
        "induction_gain": float(np.mean(gains)) if gains else float("nan"),
    }
    return out
