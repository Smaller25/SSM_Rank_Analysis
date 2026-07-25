"""3 parallel rank metrics for a single state matrix M (dk, dv).

[PIN-2] threshold-rank (paper Eq.(6)): Rank_eff = sum_i I(sigma_i > eps * sigma_1), eps=1e-4.
        Every rank number MUST be reported with its metric name, implementation and cap d=min(dk,dv).
[PIN-3] entropy eRank and stable rank MUST reuse capacity_utils.effective_rank / stable_rank
        (no re-implementation). We import them here rather than redefining.

All three metrics take a matrix (torch.Tensor or np.ndarray) and return a float. The threshold
count follows the numpy/torch matrix_rank convention (rtol against the largest singular value),
which is exactly Eq.(6) with eps=1e-4.
"""
import os
import sys

import numpy as np
import torch

# --- [PIN-3] reuse the canonical entropy eRank / stable rank ----------------------------------
# capacity_utils lives in <repo>/notebooks/. Add it to path robustly (worktree-relative).
_HERE = os.path.dirname(os.path.abspath(__file__))
_NB_CANDS = [
    os.path.join(_HERE, "..", "notebooks"),          # worktree layout
    os.path.join(_HERE, "..", "..", "notebooks"),    # nested-worktree fallback
]
for _c in _NB_CANDS:
    _c = os.path.abspath(_c)
    if os.path.isfile(os.path.join(_c, "capacity_utils.py")):
        if _c not in sys.path:
            sys.path.insert(0, _c)
        break

from capacity_utils import effective_rank as _cap_effective_rank      # noqa: E402
from capacity_utils import stable_rank as _cap_stable_rank            # noqa: E402

# Pre-registered threshold constant [PIN-2] — do NOT change without dated rationale.
EPS_THRESHOLD_RANK = 1e-4


def _to_numpy(M):
    if isinstance(M, torch.Tensor):
        return M.detach().cpu().float().numpy()
    return np.asarray(M, dtype=np.float64)


def matrix_cap(M):
    """cap d = min(dk, dv) — the algebraic upper bound to report alongside every rank."""
    a = _to_numpy(M)
    return int(min(a.shape[-2], a.shape[-1]))


def threshold_rank(M, eps=EPS_THRESHOLD_RANK):
    """Paper Eq.(6): Rank_eff = sum_i I(sigma_i > eps * sigma_1). eps=1e-4 fixed [PIN-2].

    Implemented via svdvals (CPU float, per nb3 convention). Returns an integer-valued float.
    A zero matrix (idle head) -> 0.
    """
    a = _to_numpy(M)
    s = np.linalg.svd(a, compute_uv=False)
    if s.size == 0 or s[0] <= 0:
        return 0.0
    return float(np.sum(s > eps * s[0]))


def entropy_erank(M):
    """[PIN-3] entropy eRank = exp(Shannon entropy of normalized spectrum), via capacity_utils."""
    return float(_cap_effective_rank(_to_numpy(M)))


def stable_rank(M):
    """[PIN-3] stable rank = ||M||_F^2 / ||M||_2^2, via capacity_utils."""
    return float(_cap_stable_rank(_to_numpy(M)))


def all_ranks(M, eps=EPS_THRESHOLD_RANK):
    """All 3 metrics + cap in one SVD-friendly call. Returns a dict with metric names + cap.

    Note: capacity_utils.effective_rank / stable_rank each run their own SVD; threshold_rank
    runs one more. This is intentional (PIN-3 forbids re-implementing eRank/stable to share the
    SVD). Cost is dominated by SVD of a small (dk x dv) matrix, negligible vs the forward pass.
    """
    return {
        "threshold_rank": threshold_rank(M, eps=eps),   # Eq.(6), eps=1e-4
        "entropy_erank": entropy_erank(M),              # capacity_utils.effective_rank
        "stable_rank": stable_rank(M),                  # capacity_utils.stable_rank
        "cap_d": matrix_cap(M),                         # min(dk,dv)
        "eps": float(eps),
    }


# --- small self-test (CPU, no GPU / no model needed) -------------------------------------------
def _selftest():
    rng = np.random.default_rng(0)
    # rank-3 matrix embedded in 128x128 -> threshold_rank should be exactly 3.
    U = rng.standard_normal((128, 3))
    V = rng.standard_normal((3, 128))
    M = U @ V
    r = all_ranks(M)
    assert abs(r["threshold_rank"] - 3.0) < 1e-9, r
    assert r["cap_d"] == 128
    # full-rank random: threshold_rank ~ cap, entropy_erank < cap (spectrum spread).
    F = rng.standard_normal((64, 64))
    rf = all_ranks(F)
    assert rf["threshold_rank"] == 64.0, rf
    assert rf["entropy_erank"] <= 64.0 and rf["stable_rank"] <= 64.0
    # zero matrix -> 0.
    assert all_ranks(np.zeros((128, 128)))["threshold_rank"] == 0.0
    print("[rank_metrics selftest] OK:", r, rf)


if __name__ == "__main__":
    _selftest()
