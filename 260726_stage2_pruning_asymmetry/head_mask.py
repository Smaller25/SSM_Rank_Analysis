"""Per-head output masking for GatedDeltaNet2 (Stage 2 pruning site).

Masking site (spec / gdn2.py ~line 391): the mixer output `o` of shape (b, t, h, d) immediately
AFTER `o = self.o_norm(...)` and BEFORE `rearrange(o, "b t h d -> b t (h d)")` + `o_proj`. Zeroing a
head there removes that head's contribution to o_proj -> the residual stream, i.e. an output-side
head prune (the paper's §5 head pruning), while leaving the recurrent state math untouched.

Implementation = a forward HOOK on each mixer's `o_norm` submodule (FusedRMSNormSwishGate). Its
output IS exactly the (b, t, h, d) tensor at line 391, so zeroing head channels in the hook is the
precise masking site without editing gdn2.py. The hook is keyed by the mixer's `layer_idx` (set by
common.load_model) and reads a shared mask set {(layer_idx, head_idx)}; an EMPTY set == ORIGIN
(no masking), so the four conditions differ only by which (layer,head) pairs are in the set.

num_v_heads == num_heads == 16 (PIN-1, no GVA), so o axis 2 is indexed directly by head_idx.
"""
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_STAGE1 = os.path.abspath(os.path.join(_HERE, "..", "260725_stage1_rank_stratification"))
for _p in (_HERE, _STAGE1):
    if _p not in sys.path:
        sys.path.insert(0, _p)


class HeadMasker:
    """Installs o_norm forward hooks on every GatedDeltaNet2 mixer and toggles a shared mask set.

    Usage:
        masker = HeadMasker(bundle)          # installs hooks once (origin = empty set)
        masker.set_mask({(3, 5), (7, 0)})    # zero those (layer,head) outputs
        ... run eval ...
        masker.set_mask(set())               # back to ORIGIN
    """

    def __init__(self, bundle):
        self.mask = set()                    # {(layer_idx, head_idx)}; empty == origin
        self._handles = []
        self._install(bundle)

    def _install(self, bundle):
        import common as gdn2_common
        lit = gdn2_common._find_lit()
        if lit not in sys.path:
            sys.path.insert(0, lit)
        from lit_gpt.gdn2 import GatedDeltaNet2

        model = getattr(bundle, "base", bundle).model
        mixers = [m for m in model.modules() if isinstance(m, GatedDeltaNet2)]
        if not mixers:
            raise RuntimeError("HeadMasker: no GatedDeltaNet2 mixers found on the model.")
        # layer_idx is assigned by common.load_model (mm.layer_idx = i). Use it as the mask key.
        self.num_heads = int(mixers[0].num_heads)
        self.n_layers = len(mixers)
        mask_ref = self  # capture

        for mx in mixers:
            li = int(getattr(mx, "layer_idx", -1))

            def hook(module, inp, out, _li=li):
                # out is o = o_norm(...) of shape (b, t, h, d) (line 391 gdn2.py), pre-rearrange.
                if not mask_ref.mask:
                    return out                      # ORIGIN fast-path (no allocation)
                if not torch.is_tensor(out) or out.dim() != 4:
                    return out
                heads = [h for (l, h) in mask_ref.mask if l == _li]
                if not heads:
                    return out
                out = out.clone()
                for h in heads:
                    if 0 <= h < out.shape[2]:
                        out[:, :, h, :] = 0.0
                return out

            self._handles.append(mx.o_norm.register_forward_hook(hook))

    def set_mask(self, mask_set):
        """Set the active mask (a set/iterable of (layer_idx, head_idx) tuples). Empty == origin."""
        self.mask = {(int(l), int(h)) for (l, h) in mask_set}
        return self

    def clear(self):
        self.mask = set()
        return self

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []
