"""gdn2-1.3B loader adapter for Stage 1 — reuses the canonical 260722_exp/common.py [PIN-1].

[PIN-1] model=gdn2-1.3B checkpoint-10B, lit_gpt dscpkg Config.from_name + strict=False, bf16,
        fused_recurrent. We do NOT re-implement the loader; we import 260722_exp.common.load_model
        (the canonical, session-constant loader) and add TWO logging hooks on top:

  (A) per-head recurrent state  : common.Bundle.states already returns {layer: (heads, dk, dv)}.
      Stage 1 uses this directly (no change).

  (B) per-head decay r_bar       : r_bar_h = exp(E_t[log a_t^(h)]), the geometric-mean per-step
      decay used by the nb3 r_bar->rank regression and the theory curve min(d, e/(1-r_bar)).
      gdn2's per-token decay a_t enters the recurrence as S_t = a_t * S_{t-1} + ... . We capture it
      by monkeypatching the gdn2 forward path and reading the gate tensor. Because dscpkg/lit_gpt is
      VESSL-only, the exact tensor name can drift; we probe a list of candidate attributes and, if
      none match, fall back to reconstructing r_bar from A_log + dt_bias (gated-delta decay =
      exp(-softplus(dt_bias + ...)*exp(A_log)) style). If even that is unavailable we return NaN
      r_bar (regression is then skipped and flagged) rather than fabricating a value.

The checkpoint file is resolved by common.py's _find_weights(). Stage-1 override: set env
GDN2_CKPT_PATH=/root/gdn2_1.3B_10B.pth (the checkpoint-10B file named in the spec) before loading,
so the canonical loader picks it up. We assert on the resolved path and log it for reproducibility.
"""
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
# 260722_exp/common.py is the canonical loader.
_COMMON_CANDS = [
    os.path.join(_HERE, "..", "260722_exp"),
    os.path.join(_HERE, "..", "..", "260722_exp"),
]
for _c in _COMMON_CANDS:
    _c = os.path.abspath(_c)
    if os.path.isfile(os.path.join(_c, "common.py")):
        if _c not in sys.path:
            sys.path.insert(0, _c)
        break

import common as gdn2_common   # noqa: E402  (canonical loader; DEVICE/DTYPE/CONFIG_NAME live here)

DEVICE = gdn2_common.DEVICE
CONFIG_NAME = gdn2_common.CONFIG_NAME     # "gdn2_1.3B" [PIN-1]

# NOTE: gdn2's per-token log-decay g is a LOCAL variable inside forward (gdn2.py:311-314); it is
# never stored as a module attribute. So attribute-scraping can never capture it. Instead we
# deterministically RECONSTRUCT g from the module's own parameters + the forward input, using the
# exact same expression as the canonical forward:
#     g = -A_log.exp().repeat_interleave(head_k_dim) * softplus(f_proj(hidden_states) + dt_bias)
# g is per-channel (key_dim = num_heads * head_k_dim) log-decay in (-inf, 0]; a_t = exp(g) in (0,1];
# r_bar_h = exp(E_t[g]) aggregated over the head's head_k_dim channels. This path is always available
# (all of A_log/dt_bias/f_proj/head_k_dim are module attributes) so r_bar is never silently skipped.


class Stage1Bundle:
    """Wraps a canonical common.Bundle and adds per-head r_bar logging."""

    def __init__(self, base_bundle, decay_store, capture_flag):
        self.base = base_bundle
        self.n_layer = base_bundle.n_layer
        self._decay_store = decay_store   # {layer_idx: [ (heads, tokens) log-decay arrays ]}
        self._capture = capture_flag      # {"on": bool}; hook records only when True

    @torch.no_grad()
    def logits(self, ids):
        return self.base.logits(ids)

    @torch.no_grad()
    def states(self, ids):
        """{layer: tensor(heads, dk, dv)} CPU float — delegates to the canonical loader."""
        return self.base.states(ids)

    @torch.no_grad()
    def states_and_rbar(self, ids):
        """Run one forward that captures BOTH per-head final state and per-head r_bar.

        Returns (states, rbar) where states={layer:(heads,dk,dv)} and rbar={layer: np.array[heads]}
        with rbar_h = exp(mean_t log a_t^(h)). rbar is {} if decay capture is unavailable on this
        build (flagged by caller). We clear + refill the decay store around the forward.
        """
        self._decay_store.clear()
        self._capture["on"] = True         # enable the decay hook only for this forward
        try:
            states = self.base.states(ids)  # the canonical forward also populates the cache
        finally:
            self._capture["on"] = False
        rbar = {}
        for li, logs in self._decay_store.items():
            if not logs:
                continue
            # logs: list of (heads, tokens) per-head LOG-decay arrays (g <= 0) from the reconstruction
            # hook. r_bar_h = exp(E_t[g_h]) is the geometric-mean per-step decay used by the regression.
            cat = np.concatenate([np.asarray(x).reshape(np.asarray(x).shape[0], -1) for x in logs], axis=1)
            rbar[li] = np.exp(cat.mean(axis=1))   # (heads,)
        return states, rbar


@torch.no_grad()
def _reconstruct_log_decay(module, hidden_states):
    """Recompute the per-token per-head log-decay g exactly as gdn2.forward (gdn2.py:311-314).

    g = -A_log.exp().repeat_interleave(head_k_dim) * softplus(f_proj(hidden_states) + dt_bias)

    hidden_states: (B, T, hidden) as fed to the mixer forward. Returns a numpy array of shape
    (num_heads, T*B) of per-channel-averaged log-decay per head — i.e. E over the head's head_k_dim
    channels of g, kept per token, so the caller can average over t to get log r_bar_h. g is already
    log-space (<= 0); NO clip-to-(0,1]-then-log (that would log twice)."""
    import torch.nn.functional as F
    A_log = module.A_log            # (num_heads,)
    dt_bias = module.dt_bias        # (key_dim,) = (num_heads * head_k_dim,)
    hk = int(module.head_k_dim)
    nh = int(module.num_heads)
    # cast to fp32 like the canonical forward (numerical stability of the log-decay)
    hs = hidden_states.float()
    decay_rate = A_log.float().exp().repeat_interleave(hk)          # (key_dim,)
    g = -decay_rate * F.softplus(module.f_proj(hs).float() + dt_bias.float())  # (B, T, key_dim)
    # (B, T, key_dim) -> (B, T, num_heads, head_k_dim) -> mean over channels -> (B, T, num_heads)
    B, T, _ = g.shape
    g = g.reshape(B, T, nh, hk).mean(dim=-1)                        # per-head log-decay per token
    g = g.permute(2, 0, 1).reshape(nh, B * T)                       # (num_heads, B*T)
    return g.detach().cpu().numpy()


def _install_decay_probe(model, decay_store, capture_flag):
    """Attach forward hooks to every gdn2 mixer to record per-head log(a_t), via reconstruction.

    Strategy: the log-decay g is a local variable inside gdn2.forward and is never stored on the
    module, so we deterministically reconstruct it from the module's own A_log/dt_bias/f_proj and the
    forward INPUT hidden_states (the hook's `inp`). This always fires on the real model. The hook only
    records when capture_flag["on"] is True (set by states_and_rbar) so the prefix-sweep trajectory
    forwards don't accumulate. Returns True if the probe installed on at least one mixer."""
    installed = False
    try:
        # locate the gdn2 module class via the canonical loader's import path
        lit = gdn2_common._find_lit()
        if lit not in sys.path:
            sys.path.insert(0, lit)
        from lit_gpt.gdn2 import GatedDeltaNet2
    except Exception as e:
        print(f"  [rbar] gdn2 class import failed ({type(e).__name__}) -> r_bar disabled", flush=True)
        return False

    mixers = [m for m in model.modules() if isinstance(m, GatedDeltaNet2)]

    def make_hook(li):
        def hook(module, inp, out):
            if not capture_flag.get("on"):
                return
            # inp[0] is hidden_states (B, T, hidden) as passed to forward.
            hs = inp[0] if isinstance(inp, (tuple, list)) and len(inp) else None
            if not torch.is_tensor(hs) or hs.dim() != 3:
                return
            try:
                g = _reconstruct_log_decay(module, hs)   # (num_heads, B*T) log-decay per head/token
            except Exception as e:
                print(f"  [rbar] layer {li} log-decay reconstruction failed "
                      f"({type(e).__name__}: {e})", flush=True)
                return
            decay_store.setdefault(li, []).append(g)     # already log-space; do NOT log again
        return hook

    for li, mx in enumerate(mixers):
        mx.register_forward_hook(make_hook(li))
        installed = True
    print(f"  [rbar] decay probe installed on {len(mixers)} gdn2 mixers "
          f"(reconstruct g = -exp(A_log)*softplus(f_proj(h)+dt_bias); log-space)", flush=True)
    return installed


# [PIN-1] the reproduction is pinned to checkpoint-10B; any other checkpoint (esp. the HF 95B-token
# model-95b.pth fallback) invalidates the rank-stratification verdict because state rank is sensitive
# to training progress. We refuse to silently fall back.
CKPT_10B = "/root/gdn2_1.3B_10B.pth"


def _sha256_head(path, nbytes=1 << 20):
    """sha256 of the file's first nbytes (fast integrity fingerprint for a multi-GB .pth)."""
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(nbytes))
    return h.hexdigest()


def resolve_and_assert_ckpt(checkpoint_path=None):
    """Resolve the checkpoint via the canonical _find_weights() and ASSERT it is checkpoint-10B.

    Returns a provenance dict {path, realpath, size_bytes, sha256_head_1MB} for the report JSON.
    Raises RuntimeError (rather than silently using a different checkpoint or the HF 95B fallback)
    if the pinned checkpoint-10B is not what got resolved. [PIN-1]
    """
    target = os.path.realpath(checkpoint_path or CKPT_10B)
    if checkpoint_path:
        os.environ["GDN2_CKPT_PATH"] = checkpoint_path
    w = gdn2_common._find_weights()
    if w is None:
        raise RuntimeError(
            f"[PIN-1] checkpoint-10B not found (looked for {target}). Refusing to fall back to the HF "
            f"95B-token checkpoint ({gdn2_common.GDN2_CKPT_FILE}) — that is a different training amount "
            f"and would invalidate the rank-stratification verdict. Set GDN2_CKPT_PATH to the real "
            f"checkpoint-10B .pth and retry.")
    w_real = os.path.realpath(w)
    if w_real != target:
        raise RuntimeError(
            f"[PIN-1] resolved checkpoint {w_real} != pinned checkpoint-10B {target}. _find_weights() "
            f"picked a different candidate; refusing to run the reproduction on a non-pinned checkpoint.")
    prov = {"path": w, "realpath": w_real, "size_bytes": os.path.getsize(w_real),
            "sha256_head_1MB": _sha256_head(w_real)}
    print(f"  [ckpt] [PIN-1] resolved={w_real} size={prov['size_bytes']} "
          f"sha256[:16]={prov['sha256_head_1MB'][:16]}", flush=True)
    return prov


def load(checkpoint_path=None):
    """Load gdn2-1.3B via the canonical loader; install the r_bar decay probe.

    checkpoint_path: overrides which .pth the canonical loader resolves (spec: checkpoint-10B
    /root/gdn2_1.3B_10B.pth). We resolve + ASSERT it is checkpoint-10B before loading [PIN-1], so a
    missing file fails loudly instead of silently loading the HF 95B checkpoint.

    Returns a Stage1Bundle whose `.ckpt_provenance` records the resolved path/size/hash for the report.
    """
    prov = resolve_and_assert_ckpt(checkpoint_path)
    base = gdn2_common.load_model()          # [PIN-1] canonical: Config.from_name, strict=False, bf16, fused_recurrent
    decay_store = {}
    capture_flag = {"on": False}
    _install_decay_probe(base.model, decay_store, capture_flag)
    bundle = Stage1Bundle(base, decay_store, capture_flag)
    bundle.ckpt_provenance = prov
    return bundle
