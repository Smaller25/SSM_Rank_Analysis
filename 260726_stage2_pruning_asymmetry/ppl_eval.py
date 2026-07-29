"""Per-domain PPL harness (Stage 2 secondary DV) — reuses analysis.token_nll_bits.

For each of the 3 natural domains (wikitext/github/arxiv, real texts from data_cache/ via
data_stage1), compute teacher-forced token NLL in bits (analysis.token_nll_bits, the canonical
260722 primitive) and aggregate to a mean bits/token and PPL = 2^(mean bits/token). No
re-implementation of the NLL — we import the existing function.

The data (ids per domain) are drawn ONCE by the driver (seed-dependent, PIN-6) and re-used across
the four mask conditions so the ONLY thing that changes between conditions is the mask.
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_260722 = os.path.abspath(os.path.join(_HERE, "..", "legacy", "260722_exp"))
for _p in (_HERE, _260722):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import analysis   # noqa: E402  (token_nll_bits, canonical)


def domain_ppl(bundle, ids_list):
    """Mean bits/token + PPL over a domain's sequences (teacher-forced). Returns a dict."""
    bits = []
    for ids in ids_list:
        nll_bits = analysis.token_nll_bits(bundle, ids)   # per-token surprisal (bits), length T-1
        bits.append(np.asarray(nll_bits, float))
    allbits = np.concatenate(bits) if bits else np.array([])
    mean_bits = float(np.mean(allbits)) if allbits.size else float("nan")
    return {
        "mean_bits_per_token": mean_bits,
        "ppl": float(2.0 ** mean_bits) if np.isfinite(mean_bits) else float("nan"),
        "n_seq": len(ids_list),
        "n_tokens": int(allbits.size),
    }


def run_ppl(bundle, domain_ids):
    """Per-domain PPL over a {domain: ids_list} mapping. Returns {domain: {...}} + macro mean."""
    out = {}
    for dom, ids_list in domain_ids.items():
        out[dom] = domain_ppl(bundle, ids_list)
    ppls = [out[d]["ppl"] for d in out if np.isfinite(out[d]["ppl"])]
    bits = [out[d]["mean_bits_per_token"] for d in out if np.isfinite(out[d]["mean_bits_per_token"])]
    out["_macro"] = {
        "mean_ppl": float(np.mean(ppls)) if ppls else float("nan"),
        "mean_bits_per_token": float(np.mean(bits)) if bits else float("nan"),
    }
    return out
