"""Per-head KV-STATE pruning for GatedDeltaNet2 (Stage 2 pruning site).

TRUE state pruning (not output ablation): we zero a head's VALUE input to the recurrence so the
head's recurrent state S_h = Σ_t (gated) v_h,t k_h,t^T stays 0 for all t -> the head stores NOTHING
in the KV state and contributes 0 to the readout. This is literally the paper's §5 head pruning:
the pruned head's KV-state memory footprint is freed. We hook each mixer's `v_conv1d` output
(gdn2.py:297, post-conv/activation, shape (b,t, num_v_heads*head_v_dim)) and zero the value channels
of masked heads: head h owns channels [h*head_v_dim : (h+1)*head_v_dim]. v_h=0 => S_h=0 (the erase
term (I - k(b⊙k)^T) acting on a 0 state is still 0), so the head is removed from the KV state.

Architectural note (why this is faithful AND why accuracy == output-ablation): GDN2 heads are
INDEPENDENT and o = o_norm(readout); with S_h=0 the head's readout is 0 and o_norm(0)=0, so for the
FORWARD-PASS RESULT zeroing the state is identical to zeroing the head output. The two differ only in
MEMORY: state-pruning frees the head's KV state (reported as kv_reduction), output-ablation does not.
We prune at the state so the KV-memory-reduction claim (cf. paper's 38.9%) is grounded in the literal
operation. num_v_heads == num_heads == 16 (PIN-1, no GVA), so value channels map 1:1 to head_idx.
"""
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_STAGE1 = os.path.abspath(os.path.join(_HERE, "..", "260725_stage1_rank_stratification"))
_260722 = os.path.abspath(os.path.join(_HERE, "..", "legacy", "260722_exp"))
for _p in (_HERE, _STAGE1, _260722):
    if _p not in sys.path:
        sys.path.insert(0, _p)


class HeadMasker:
    """Installs v_conv1d forward hooks on every GatedDeltaNet2 mixer and toggles a shared mask set.
    Zeroing a head's value channels removes it from the recurrent KV state (true state pruning).

        masker = HeadMasker(bundle)          # installs hooks once (origin = empty set)
        masker.set_mask({(3, 5), (7, 0)})    # prune those heads' KV state
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
        self.num_heads = int(mixers[0].num_heads)
        self.num_v_heads = int(getattr(mixers[0], "num_v_heads", mixers[0].num_heads))
        self.head_v_dim = int(mixers[0].head_v_dim)
        self.head_k_dim = int(getattr(mixers[0], "head_k_dim", mixers[0].head_dim))
        self.n_layers = len(mixers)
        mask_ref = self

        for mx in mixers:
            li = int(getattr(mx, "layer_idx", -1))
            hvd = int(mx.head_v_dim)
            nvh = int(getattr(mx, "num_v_heads", mx.num_heads))

            def hook(module, inp, out, _li=li, _hvd=hvd, _nvh=nvh):
                if not mask_ref.mask:
                    return out                      # ORIGIN fast-path
                heads = [h for (l, h) in mask_ref.mask if l == _li]
                if not heads:
                    return out
                # v_conv1d returns (v, conv_state) or a bare tensor; v is (b, t, num_v_heads*head_v_dim)
                v = out[0] if isinstance(out, tuple) else out
                if not torch.is_tensor(v) or v.dim() != 3:
                    return out
                v = v.clone()
                for h in heads:
                    if 0 <= h < _nvh:
                        v[:, :, h * _hvd:(h + 1) * _hvd] = 0.0   # zero this head's VALUE -> S_h = 0
                return (v,) + tuple(out[1:]) if isinstance(out, tuple) else v

            # hook v_conv1d if present (conv path), else fall back to v_proj (no-conv path)
            target = getattr(mx, "v_conv1d", None) or getattr(mx, "v_proj", None)
            if target is None:
                raise RuntimeError(f"HeadMasker: mixer {li} has no v_conv1d/v_proj to hook.")
            self._handles.append(target.register_forward_hook(hook))

    def set_mask(self, mask_set):
        self.mask = {(int(l), int(h)) for (l, h) in mask_set}
        return self

    def clear(self):
        self.mask = set()
        return self

    def kv_reduction(self, mask_set=None):
        """KV-state memory footprint freed by pruning these heads (each head's state is
        head_k_dim x head_v_dim). Returns {n_masked, total_heads, kv_reduction_fraction, kv_reduction_pct}.
        Uniform per-head state size, so the fraction is simply n_masked / (n_layers*num_heads)."""
        m = self.mask if mask_set is None else {(int(l), int(h)) for (l, h) in mask_set}
        total = self.n_layers * self.num_heads
        frac = (len(m) / total) if total else 0.0
        return {"n_masked": len(m), "total_heads": total,
                "per_head_state_elems": self.head_k_dim * self.head_v_dim,
                "kv_reduction_fraction": frac, "kv_reduction_pct": round(100.0 * frac, 2)}

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []
