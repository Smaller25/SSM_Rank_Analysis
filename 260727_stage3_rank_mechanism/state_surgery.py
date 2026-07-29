"""Stage 3 CORE — spectral-content surgery on per-head GDN2 recurrent state S_h (SEGMENTED intervention).

WHY SEGMENTED (the load-bearing design decision — task's "3-lens" mandate)
--------------------------------------------------------------------------
GDN2 readout is o_t = S_t q_t and lives INSIDE the fused/chunk kernel; the forward exposes ONLY the
FINAL state S_T (via output_final_state -> cache["recurrent_state"], shape [N, HV, K, V]).

  (a) READ lens   : gdn2.py forward runs ONE kernel call per forward over the whole sequence; the
                    recurrent_state written to the fla Cache is S_T only. Per-position S_t never
                    surfaces at the Python level.
  (b) KERNEL lens : fused_recurrent_gdn2 / chunk_gdn2 accept `initial_state` ([N,HV,K,V] fp32) and
                    return `final_state` when output_final_state=True -> segment-wise state carry is
                    NATIVE. The canonical loader's SHARED-cache monkeypatch (260722_exp/common.py)
                    already threads a single fla Cache across successive forwards.
  (c) INTENT lens : the G3 claim is about the spectral CONTENT of the state that drives the readout
                    ACROSS the sequence, not about the last token only.

  => Final-state-only surgery + a hand-rolled last-position readout is REJECTED: it perturbs exactly
     1 token, cannot move teacher-forced PPL, and does not test the intent. The tractable,
     intent-faithful method is SEGMENTED intervention:

        split the sequence into segments of `segment_len`; run the model per segment with the SHARED
        cache carrying state (use_cache forced True inside the monkeypatched forward); at each segment
        BOUNDARY read the returned per-head state S_h from the cache, apply SVD-top-r / spectrum-noise
        / zero to its SPECTRAL CONTENT for the targeted head GROUP, write it back, and let the next
        segment's readout flow through the surgered state.

     The readout of every token after the first boundary then depends on the surgered state -> the
     intervention is causal and PPL-visible. This module implements exactly that carry + surgery.

CONTROL (harness faithfulness, PREREG): with `segment_len >= seq_len` there is exactly ONE segment
and therefore ZERO boundaries, so NO surgery is applied and the segmented path reproduces the
single-shot origin PPL bit-for-bit. `run_segmented_ppl(..., surgery=None)` is likewise a pure control.

SURGERIES (all on CPU float per (layer,head), cast back to the state dtype/device):
  - "topr"      : replace S_h by its rank-`r` truncated SVD (Eckart-Young). r = round(ratio * cap).
                  ratio=1.0 -> identity (no-op). ratio=0 -> zero. Reduces threshold_rank to ~r for a
                  genuinely full-rank head; leaves an already-low-rank head ~unchanged.
  - "spectrum"  : S_noise = U_rand diag(sigma_orig) V_rand^T with Haar-random orthogonal U_rand,V_rand
                  (QR of a Gaussian, sign-fixed). PRESERVES the singular values (=> identical
                  threshold_rank, nuclear_norm, Frobenius energy) but RANDOMIZES the singular vectors
                  (destroys the stored CONTENT). The content-destroying null.
  - "zero"      : S_h <- 0 (ladder ceiling for damage; == the head_mask KV-state prune of that head).

Head GROUP selection: surgery is applied ONLY to the (layer,head) pairs in `target_heads`. Groups
{high, low, random} are supplied by the driver (reusing head_classifier's disjoint count-matched sets).

Reuse ONLY: 260722_exp/common (loader + SHARED cache + Bundle), rank_metrics (assert invariants).
"""
import os
import sys

import numpy as np
import torch

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


# ---------------------------------------------------------------- spectral surgeries (CPU, per head)
def _rng_from(seed, layer, head, salt=0):
    """Deterministic per-(layer,head) RNG so spectrum-noise is reproducible per seed (PIN-6)."""
    return np.random.default_rng((int(seed) * 1_000_003 + int(layer) * 1009 + int(head) * 31 + salt)
                                 & 0x7FFFFFFF)


def _haar_orthogonal(n, rng):
    """Haar-random n x n orthogonal matrix = Q from QR(Gaussian), with sign-fixed R diagonal so the
    distribution is exactly the orthogonal Haar measure (Mezzadri 2007)."""
    A = rng.standard_normal((n, n))
    Q, R = np.linalg.qr(A)
    d = np.sign(np.diag(R))
    d[d == 0] = 1.0
    return Q * d[np.newaxis, :]


def surgery_topr(M, ratio):
    """Rank-r truncated SVD of M (dk,dv); r = round(ratio*cap), cap=min(dk,dv). Returns np.float64.
    ratio>=1 -> M unchanged (identity); ratio<=0 -> zeros."""
    a = np.asarray(M, dtype=np.float64)
    cap = min(a.shape[-2], a.shape[-1])
    r = int(round(float(ratio) * cap))
    if r >= cap:
        return a.copy()
    if r <= 0:
        return np.zeros_like(a)
    U, s, Vt = np.linalg.svd(a, full_matrices=False)
    return (U[:, :r] * s[:r]) @ Vt[:r, :]


def surgery_spectrum_noise(M, seed, layer, head):
    """S_noise = U_rand diag(sigma_orig) V_rand^T. PRESERVES the singular values (rank / nuclear norm /
    Frobenius energy) exactly; RANDOMIZES the singular vectors (destroys content). Returns np.float64.

    U_rand is (dk x dk), V_rand is (dv x dv), Haar-random orthogonal. We rebuild with the ORIGINAL
    singular values in descending order, so np.linalg.svd of the result recovers the same spectrum
    (up to numerical noise) -> threshold_rank + nuclear_norm invariant (asserted in the smoke test)."""
    a = np.asarray(M, dtype=np.float64)
    dk, dv = a.shape[-2], a.shape[-1]
    s = np.linalg.svd(a, compute_uv=False)          # descending singular values, length min(dk,dv)
    rng = _rng_from(seed, layer, head, salt=7)
    U = _haar_orthogonal(dk, rng)
    V = _haar_orthogonal(dv, rng)
    Sig = np.zeros((dk, dv), dtype=np.float64)
    m = min(dk, dv)
    Sig[:m, :m] = np.diag(s[:m])
    return U @ Sig @ V.T


def surgery_zero(M):
    a = np.asarray(M, dtype=np.float64)
    return np.zeros_like(a)


def apply_surgery_to_matrix(M, kind, *, ratio=1.0, seed=0, layer=0, head=0):
    """Dispatch a single-head surgery. `M` is (dk,dv) numpy/torch; returns np.float64 (dk,dv)."""
    if kind is None or kind == "none" or kind == "origin":
        return np.asarray(M, dtype=np.float64).copy()
    if kind == "topr":
        return surgery_topr(M, ratio)
    if kind == "spectrum":
        return surgery_spectrum_noise(M, seed, layer, head)
    if kind == "zero":
        return surgery_zero(M)
    raise ValueError(f"unknown surgery kind {kind!r}")


# ------------------------------------------------------------------- in-place surgery on a cache tensor
def surgery_cache_state(recurrent_state, target_heads_at_layer, kind, *, ratio, seed, layer):
    """Apply `kind` surgery to the targeted heads of ONE layer's recurrent_state tensor in place.

    recurrent_state: torch tensor [N, HV, K, V] (the kernel's final_state; N=batch=1 here). We surgery
    head h = recurrent_state[0, h] (a (K,V)==(dk,dv) matrix). Only heads in `target_heads_at_layer`
    are touched; the rest are left exactly as the kernel produced them. Returns the (possibly new)
    tensor to store back into the cache."""
    if not target_heads_at_layer:
        return recurrent_state
    rs = recurrent_state
    if not torch.is_tensor(rs) or rs.dim() != 4:
        return rs
    dev, dt = rs.device, rs.dtype
    rs = rs.clone()
    for h in target_heads_at_layer:
        if not (0 <= h < rs.shape[1]):
            continue
        M = rs[0, h].detach().float().cpu().numpy()
        Ms = apply_surgery_to_matrix(M, kind, ratio=ratio, seed=seed, layer=layer, head=h)
        rs[0, h] = torch.from_numpy(np.ascontiguousarray(Ms)).to(device=dev, dtype=dt)
    return rs


# --------------------------------------------------------------------------- segmented forward driver
class SegmentedSurgeon:
    """Runs the model over a sequence in segments, applying spectral surgery to a head GROUP at each
    segment boundary, and returns per-token teacher-forced logits over the FULL sequence.

    Usage:
        surg = SegmentedSurgeon(bundle, segment_len=256)
        logits = surg.forward_logits(ids, target_heads=high_heads,
                                     surgery="topr", ratio=0.5, seed=0)
        # logits: (1, T, vocab) — identical concatenation order as a single-shot forward.

    Mechanism: the canonical loader monkeypatches GatedDeltaNet2.forward to read/write
    SHARED["cache"] (an fla Cache) and force use_cache=True whenever the cache is non-None
    (260722_exp/common.py:107). So if we (1) install a fresh Cache, (2) run segment 0, (3) reach into
    the Cache and surgery each layer's recurrent_state, (4) run segment 1 (which picks the surgered
    state up as initial_state), ... the readout in every later segment flows through the surgered
    state. This reuses the EXISTING carry path (no new forward patch), matching the plan's option to
    "reuse the existing Cache carry" and mirroring common.py's SHARED-cache design.

    surgery=None (or segment_len covering the whole seq -> 1 segment, 0 boundaries) => NO surgery =>
    reproduces the single-shot origin logits (harness-faithfulness control, PREREG)."""

    def __init__(self, bundle, segment_len=256):
        self.bundle = bundle
        self.base = getattr(bundle, "base", bundle)     # canonical common.Bundle (Stage1Bundle wraps it)
        self.shared = getattr(self.base, "shared", None)
        self.model = self.base.model
        self.n_layer = self.base.n_layer
        self.segment_len = int(segment_len)
        self.device = _model_device(self.model)
        # group target heads by layer once per call (set in forward_logits)
        self._by_layer = {}

    def _make_cache(self):
        from fla.models.utils import Cache
        return Cache()

    def _surgery_cache(self, kind, ratio, seed):
        """Reach into the live SHARED Cache and surgery every layer's recurrent_state for the target
        heads of that layer. Called AT a segment boundary (after a segment forward populated the
        cache, before the next segment consumes it as initial_state)."""
        cache = self.shared.get("cache")
        if cache is None:
            return
        for li in range(self.n_layer):
            heads = self._by_layer.get(li)
            if not heads:
                continue
            try:
                st = cache[li]
                rs = st.get("recurrent_state")
            except Exception:
                continue
            if rs is None:
                continue
            new_rs = surgery_cache_state(rs, heads, kind, ratio=ratio, seed=seed, layer=li)
            st["recurrent_state"] = new_rs

    @torch.no_grad()
    def forward_logits(self, ids, *, target_heads=None, surgery=None, ratio=1.0, seed=0):
        """Teacher-forced logits (1,T,vocab) over the full sequence, segmented with boundary surgery.

        target_heads: iterable of (layer,head) to surgery (the GROUP). surgery in
        {None,"topr","spectrum","zero"}. With surgery=None this is a plain segmented forward whose
        concatenated logits equal a single-shot forward (control)."""
        ids = ids.to(self.device)
        T = ids.shape[1]
        target_heads = list(target_heads or [])
        self._by_layer = {}
        for (l, h) in target_heads:
            self._by_layer.setdefault(int(l), []).append(int(h))

        seg = self.segment_len if self.segment_len > 0 else T
        # bounds of each segment
        starts = list(range(0, T, seg))
        # fresh carry cache for this sequence
        self.shared["cache"] = self._make_cache()
        chunks = []
        try:
            for si, s0 in enumerate(starts):
                s1 = min(T, s0 + seg)
                seg_ids = ids[:, s0:s1]
                out = self.model(seg_ids)            # patched forward: uses+updates SHARED cache
                chunks.append(out.float().cpu())
                # apply surgery at the boundary AFTER this segment (so the NEXT segment reads it),
                # skip after the final segment (no next segment to influence).
                if surgery not in (None, "none", "origin") and s1 < T:
                    self._surgery_cache(surgery, ratio, seed)
        finally:
            self.shared["cache"] = None              # leave the shared bundle in the stateless branch
        logits = torch.cat(chunks, dim=1)            # (1, T, vocab); segment order == token order
        return logits


def _model_device(model):
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- PPL over segmented-surgery logits
def _token_nll_bits_from_logits(logits, ids):
    """Teacher-forced per-token surprisal (bits), length T-1 — same primitive as analysis.token_nll_bits
    but on logits we already computed via the segmented forward (so the surgered state is reflected)."""
    import math
    logp = torch.log_softmax(logits, -1)
    tgt = ids[:, 1:].to(logp.device)
    nll = -logp[:, :-1, :].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[0]
    return nll.float().cpu().numpy() / math.log(2)


def segmented_domain_ppl(surgeon, ids_list, *, target_heads, surgery, ratio, seed):
    """Mean bits/token + PPL over a domain's sequences under a segmented surgery. Mirrors
    ppl_eval.domain_ppl but routes each forward through SegmentedSurgeon so the surgered state affects
    the readout. surgery=None + target_heads=[] == origin (segmented control)."""
    bits = []
    for ids in ids_list:
        logits = surgeon.forward_logits(ids, target_heads=target_heads, surgery=surgery,
                                        ratio=ratio, seed=seed)
        nb = _token_nll_bits_from_logits(logits, ids.to(logits.device))
        bits.append(np.asarray(nb, float))
    allbits = np.concatenate(bits) if bits else np.array([])
    mean_bits = float(np.mean(allbits)) if allbits.size else float("nan")
    return {"mean_bits_per_token": mean_bits,
            "ppl": float(2.0 ** mean_bits) if np.isfinite(mean_bits) else float("nan"),
            "n_seq": len(ids_list), "n_tokens": int(allbits.size)}


def run_segmented_ppl(surgeon, domain_ids, *, target_heads, surgery, ratio, seed):
    """Per-domain PPL under one segmented surgery condition. Returns {domain:{...}} + _macro (mirrors
    ppl_eval.run_ppl so aggregate/verdict code is shared)."""
    out = {}
    for dom, ids_list in domain_ids.items():
        out[dom] = segmented_domain_ppl(surgeon, ids_list, target_heads=target_heads,
                                        surgery=surgery, ratio=ratio, seed=seed)
    ppls = [out[d]["ppl"] for d in out if np.isfinite(out[d]["ppl"])]
    bits = [out[d]["mean_bits_per_token"] for d in out if np.isfinite(out[d]["mean_bits_per_token"])]
    out["_macro"] = {"mean_ppl": float(np.mean(ppls)) if ppls else float("nan"),
                     "mean_bits_per_token": float(np.mean(bits)) if bits else float("nan")}
    return out


# --------------------------------------------------------------------------------- invariant checks
def assert_spectrum_preserved(M, seed=0, layer=0, head=0, atol=1e-6, rtol=1e-4):
    """Verify spectrum-noise preserves threshold_rank AND nuclear_norm (correctness pitfall in the
    task). Returns a dict of the before/after invariants. Uses rank_metrics (Eq.6 eps=1e-4)."""
    from rank_metrics import threshold_rank, nuclear_norm
    Ms = surgery_spectrum_noise(M, seed, layer, head)
    r0, r1 = threshold_rank(M), threshold_rank(Ms)
    n0, n1 = nuclear_norm(M), nuclear_norm(Ms)
    e0 = float(np.linalg.norm(np.asarray(M, float)))
    e1 = float(np.linalg.norm(Ms))
    ok_rank = (r0 == r1)
    ok_nuc = bool(abs(n0 - n1) <= atol + rtol * abs(n0))
    ok_energy = bool(abs(e0 - e1) <= atol + rtol * abs(e0))
    return {"threshold_rank": [r0, r1], "nuclear_norm": [n0, n1], "frobenius": [e0, e1],
            "rank_preserved": bool(ok_rank), "nuclear_preserved": ok_nuc,
            "energy_preserved": ok_energy}


def assert_topr_reduces(M, ratio, atol_low=1.0):
    """Verify top-r truncation reduces threshold_rank to ~r for a full-rank M. Returns diagnostics."""
    from rank_metrics import threshold_rank, matrix_cap
    cap = matrix_cap(M)
    r_target = int(round(ratio * cap))
    Ms = surgery_topr(M, ratio)
    r_after = threshold_rank(Ms)
    return {"cap": cap, "r_target": r_target, "rank_before": threshold_rank(M),
            "rank_after": r_after, "ok": bool(r_after <= max(r_target, 0) + atol_low)}


# ------------------------------------------------------------------------- CPU self-test (no model)
def _selftest():
    rng = np.random.default_rng(0)
    # full-rank 128x128 with a decaying spectrum (like a genuine head state)
    U = _haar_orthogonal(128, rng); V = _haar_orthogonal(128, rng)
    s = np.linspace(10.0, 0.01, 128)
    Mhi = (U * s) @ V.T
    # low-rank 128x128 (rank 4) — an "already low-dim" head
    Mlo = rng.standard_normal((128, 4)) @ rng.standard_normal((4, 128))

    # spectrum-noise preserves rank + nuclear norm on the full-rank head
    inv = assert_spectrum_preserved(Mhi, seed=1, layer=2, head=3)
    assert inv["rank_preserved"], inv
    assert inv["nuclear_preserved"], inv
    assert inv["energy_preserved"], inv

    # top-r truncation drops the full-rank head's rank; leaves the low-rank head ~unchanged
    from rank_metrics import threshold_rank
    r_hi_full = threshold_rank(Mhi)
    r_hi_half = threshold_rank(surgery_topr(Mhi, 0.5))
    assert r_hi_half <= r_hi_full * 0.55 + 1.0, (r_hi_full, r_hi_half)
    r_lo_full = threshold_rank(Mlo)
    r_lo_half = threshold_rank(surgery_topr(Mlo, 0.5))
    # low-rank head already < half cap -> top-r at ratio 0.5 (r=64) keeps all its ~4 dims
    assert r_lo_half == r_lo_full, (r_lo_full, r_lo_half)

    # zero surgery -> zero
    assert np.allclose(surgery_zero(Mhi), 0.0)
    # ratio 1.0 topr is identity; ratio 0 is zero
    assert np.allclose(surgery_topr(Mhi, 1.0), Mhi, atol=1e-8)
    assert np.allclose(surgery_topr(Mhi, 0.0), 0.0)

    # cache-state surgery touches ONLY targeted heads
    rs = torch.from_numpy(np.stack([Mhi, Mlo, Mhi, Mlo])[None].astype("float32"))  # [1,4,128,128]
    out = surgery_cache_state(rs, [1, 3], "zero", ratio=1.0, seed=0, layer=0)
    assert torch.allclose(out[0, 0], rs[0, 0]) and torch.allclose(out[0, 2], rs[0, 2])
    assert torch.count_nonzero(out[0, 1]) == 0 and torch.count_nonzero(out[0, 3]) == 0
    print("[state_surgery selftest] OK", {k: inv[k] for k in ("rank_preserved", "nuclear_preserved")},
          "topr hi:", (r_hi_full, r_hi_half), "lo:", (r_lo_full, r_lo_half))


if __name__ == "__main__":
    _selftest()
