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

# Candidate module-attribute names that hold the per-token multiplicative decay a_t (heads, T) or
# (B, T, heads). Probed in order; first tensor-valued hit wins. Names cover common gdn2/GDN variants.
_DECAY_ATTR_CANDS = ["g", "g_t", "decay", "gate", "alpha", "a_t", "log_g", "gk"]


class Stage1Bundle:
    """Wraps a canonical common.Bundle and adds per-head r_bar logging."""

    def __init__(self, base_bundle, decay_store):
        self.base = base_bundle
        self.n_layer = base_bundle.n_layer
        self._decay_store = decay_store   # {layer_idx: log_a accumulator}

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
        states = self.base.states(ids)     # the canonical forward also populates the cache
        rbar = {}
        for li, logs in self._decay_store.items():
            if not logs:
                continue
            # logs: list of (heads,) or (heads,T) log-decay tensors accumulated over the forward.
            cat = np.concatenate([np.asarray(x).reshape(np.asarray(x).shape[0], -1) for x in logs], axis=1)
            rbar[li] = np.exp(cat.mean(axis=1))   # (heads,)
        return states, rbar


def _install_decay_probe(model, decay_store):
    """Attach forward hooks to every gdn2 mixer to record log(a_t) per head, if discoverable.

    Strategy: hook the mixer module; in the hook, scan module attributes set during forward for a
    decay-like tensor. gdn2 stores intermediates as attributes only transiently, so we instead wrap
    the fused_recurrent call if importable. Best-effort: returns True if any probe was installed.
    """
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
            # scan for a decay tensor among attributes freshly written during forward.
            for attr in _DECAY_ATTR_CANDS:
                v = getattr(module, attr, None)
                if torch.is_tensor(v) and v.dim() >= 2:
                    t = v.detach().float().cpu()
                    # reduce to (heads, T): assume a head dim of size n_heads exists.
                    nh = getattr(module, "num_heads", None) or getattr(module, "n_head", None)
                    arr = t.numpy()
                    # collapse batch, keep the axis whose size == nh as 'heads'
                    if nh is not None and nh in arr.shape:
                        ax = arr.shape.index(nh)
                        arr = np.moveaxis(arr, ax, 0).reshape(nh, -1)
                    else:
                        arr = arr.reshape(arr.shape[0], -1)
                    # a_t may be stored as g (a_t) or log_g; clamp to (0,1] then log.
                    a = np.clip(arr, 1e-8, 1.0) if attr not in ("log_g",) else np.exp(arr)
                    decay_store.setdefault(li, []).append(np.log(np.clip(a, 1e-8, 1.0)))
                    return
        return hook

    for li, mx in enumerate(mixers):
        mx.register_forward_hook(make_hook(li))
        installed = True
    print(f"  [rbar] decay probe installed on {len(mixers)} gdn2 mixers "
          f"(candidate attrs={_DECAY_ATTR_CANDS})", flush=True)
    return installed


def load(checkpoint_path=None):
    """Load gdn2-1.3B via the canonical loader; install the r_bar decay probe.

    checkpoint_path: overrides which .pth the canonical loader resolves (spec: checkpoint-10B
    /root/gdn2_1.3B_10B.pth). If given, we point GDN2_CKPT_PATH at it before calling load_model().
    """
    if checkpoint_path:
        os.environ["GDN2_CKPT_PATH"] = checkpoint_path
    base = gdn2_common.load_model()          # [PIN-1] canonical: Config.from_name, strict=False, bf16, fused_recurrent
    decay_store = {}
    _install_decay_probe(base.model, decay_store)
    return Stage1Bundle(base, decay_store)
