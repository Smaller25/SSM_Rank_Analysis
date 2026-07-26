"""gdn2-1.3B loader adapter for Stage 1 — reuses the canonical 260722_exp/common.py [PIN-1].

[PIN-1] model = gdn2-1.3B **paper-matched checkpoint (100B tokens)** = `model-100b.pth`
        (LLM-OS-Models2/gdn2-1.3b-paper-matched). lit_gpt dscpkg Config.from_name + strict=False,
        bf16, fused_recurrent. The 95B checkpoint is ~identical to 100B (paper-matched) and is an
        ACCEPTED near-equivalent (warn); the **10B checkpoint is substantially different and is
        REJECTED** — state rank is sensitive to training amount. We do NOT re-implement the loader;
        we import 260722_exp.common.load_model (the canonical, session-constant loader) and add two
        logging hooks on top:

  (A) per-head recurrent state : common.Bundle.states returns {layer: (heads, dk, dv)}. Used as-is.

  (B) per-head decay r_bar      : r_bar_h = exp(E_t[log a_t^(h)]), the geometric-mean per-step decay.
      a_t CAPTURE (the critical primitive): gdn2's per-token log-decay `g` is passed to the module-
      level kernel `fused_recurrent_gdn2`. With `use_gate_in_kernel=False` (the canonical loader's
      setting) `g` is ALREADY the log-decay (a_t = exp(g)); with True the kernel computes it
      internally. So the ROBUST capture is to **intercept the kernel and read its `g` argument
      directly** (a PyTorch-level hook on the write/decay path), NOT to scrape a module attribute
      (g is a forward-local, never stored). We monkeypatch `lit_gpt.gdn2.fused_recurrent_gdn2`,
      gated by a capture flag, and record g per layer. If the kernel symbol is unavailable we FALL
      BACK to deterministically reconstructing g from the module's own params
      (`g = -exp(A_log).repeat_interleave(hk) * softplus(f_proj(h) + dt_bias)`, gdn2.py:311-314),
      which is bit-exact to the kernel input. r_bar is therefore never silently skipped.

Checkpoint resolution: pass `--ckpt <path to model-100b.pth>` (or env GDN2_CKPT_PATH); we resolve
via common._find_weights(), ASSERT it is the 100B (or 95B near-equiv) checkpoint, and log
path/size/sha for reproducibility. Refs: fla `fused_recurrent_gdn2` use_gate_in_kernel semantics
(github.com/fla-org/flash-linear-attention); GDN2 erase/write gates (arXiv:2605.22791).
"""
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
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

# [PIN-1] canonical checkpoint = paper-matched 100B. 95B is an accepted near-equivalent (paper-matched,
# ~identical); 10B is REJECTED (different training amount -> different state rank).
CKPT_100B_DEFAULT = "/home/sohyung/models/gdn2_1.3B_100b.pth"   # local default (Blackwell GPU)
_ACCEPT_TOKENS = ("100b", "100B", "95b", "95B")                 # filename must name one of these
_REJECT_TOKENS = ("10b", "10B")


def _g_to_per_head_logdecay(g, num_heads, head_k_dim):
    """Reduce a captured kernel/reconstructed log-decay tensor g to (num_heads, B*T).

    g may be (B, T, num_heads, head_k_dim) or (B, T, key_dim=num_heads*head_k_dim). We mean over the
    head's head_k_dim channels (so per-head, per-token log-decay) and flatten (B,T)->B*T. g is already
    log-space (<= 0); we NEVER log again."""
    gg = g.detach().float()
    if gg.dim() == 4:                                  # (B, T, H, hk)
        per = gg.mean(dim=-1)                           # (B, T, H)
    elif gg.dim() == 3:                                 # (B, T, key_dim)
        B, T, kd = gg.shape
        per = gg.reshape(B, T, num_heads, head_k_dim).mean(dim=-1)  # (B, T, H)
    else:
        raise ValueError(f"unexpected g shape {tuple(gg.shape)}")
    B, T, H = per.shape
    return per.permute(2, 0, 1).reshape(H, B * T).cpu().numpy()     # (H, B*T)


class Stage1Bundle:
    """Wraps a canonical common.Bundle and adds per-head r_bar logging (kernel-intercept a_t)."""

    def __init__(self, base_bundle, decay_store, capture_flag, reset_counter, capture_mode):
        self.base = base_bundle
        self.n_layer = base_bundle.n_layer
        self._decay_store = decay_store       # {layer_idx: [ (heads, tokens) log-decay arrays ]}
        self._capture = capture_flag          # {"on": bool}; hook records only when True
        self._reset_counter = reset_counter   # callable(): reset kernel-call -> layer counter
        self.capture_mode = capture_mode      # "kernel" | "reconstruct" | "none"

    @torch.no_grad()
    def logits(self, ids):
        return self.base.logits(ids)

    @torch.no_grad()
    def states(self, ids):
        return self.base.states(ids)

    @torch.no_grad()
    def states_and_rbar(self, ids):
        """One forward capturing per-head final state AND per-head r_bar via a_t = exp(g) capture.

        Returns (states, rbar), states={layer:(heads,dk,dv)}, rbar={layer: np.array[heads]} with
        rbar_h = exp(mean_t log a_t^(h)). rbar is {} only if capture_mode == 'none'."""
        self._decay_store.clear()
        self._reset_counter()
        self._capture["on"] = True
        try:
            states = self.base.states(ids)
        finally:
            self._capture["on"] = False
        rbar = {}
        for li, logs in self._decay_store.items():
            if not logs:
                continue
            cat = np.concatenate([np.asarray(x).reshape(np.asarray(x).shape[0], -1) for x in logs], axis=1)
            rbar[li] = np.exp(cat.mean(axis=1))   # (heads,) geometric-mean per-step decay
        return states, rbar


# ---------------------------------------------------------------- a_t capture (primary: kernel hook)
def _install_kernel_capture(model, decay_store, capture_flag):
    """Monkeypatch lit_gpt.gdn2.fused_recurrent_gdn2 to capture the exact per-head log-decay g that
    the forward passes to the kernel (a_t = exp(g)). Returns (ok, reset_counter)."""
    try:
        lit = gdn2_common._find_lit()
        if lit not in sys.path:
            sys.path.insert(0, lit)
        import lit_gpt.gdn2 as Gmod
        from lit_gpt.gdn2 import GatedDeltaNet2
    except Exception as e:
        print(f"  [rbar] kernel import failed ({type(e).__name__}) -> try reconstruction", flush=True)
        return False, None

    mixers = [m for m in model.modules() if isinstance(m, GatedDeltaNet2)]
    if not mixers:
        return False, None
    nl = len(mixers)
    nh = int(mixers[0].num_heads)
    hk = int(mixers[0].head_k_dim)
    orig = Gmod.fused_recurrent_gdn2
    ctr = {"i": 0}

    def wrapped(*a, **kw):
        if capture_flag.get("on"):
            g = kw.get("g", None)
            if g is None and len(a) >= 4:
                g = a[3]                          # positional (q, k, v, g, ...)
            if torch.is_tensor(g):
                li = ctr["i"] % nl
                ctr["i"] += 1
                try:
                    decay_store.setdefault(li, []).append(_g_to_per_head_logdecay(g, nh, hk))
                except Exception as e:
                    print(f"  [rbar] layer {li} g-capture failed ({type(e).__name__}: {e})", flush=True)
        return orig(*a, **kw)

    Gmod.fused_recurrent_gdn2 = wrapped
    print(f"  [rbar] a_t capture = KERNEL-INTERCEPT on fused_recurrent_gdn2 "
          f"({nl} gdn2 layers, nh={nh}, hk={hk}; a_t=exp(g), g=log-decay)", flush=True)
    return True, (lambda: ctr.__setitem__("i", 0))


# ------------------------------------------------------------- a_t capture (fallback: reconstruct g)
@torch.no_grad()
def _reconstruct_log_decay(module, hidden_states):
    """Fallback: recompute per-token per-head log-decay g exactly as gdn2.forward (gdn2.py:311-314):
        g = -A_log.exp().repeat_interleave(head_k_dim) * softplus(f_proj(hidden_states) + dt_bias)
    Returns (num_heads, B*T) per-head log-decay. Bit-exact to the kernel input."""
    import torch.nn.functional as F
    hk = int(module.head_k_dim)
    nh = int(module.num_heads)
    hs = hidden_states.float()
    decay_rate = module.A_log.float().exp().repeat_interleave(hk)                     # (key_dim,)
    g = -decay_rate * F.softplus(module.f_proj(hs).float() + module.dt_bias.float())  # (B,T,key_dim)
    return _g_to_per_head_logdecay(g, nh, hk)


def _install_reconstruct_probe(model, decay_store, capture_flag):
    """Fallback capture via module forward hook + deterministic reconstruction. Returns (ok, reset)."""
    try:
        lit = gdn2_common._find_lit()
        if lit not in sys.path:
            sys.path.insert(0, lit)
        from lit_gpt.gdn2 import GatedDeltaNet2
    except Exception as e:
        print(f"  [rbar] gdn2 class import failed ({type(e).__name__}) -> r_bar disabled", flush=True)
        return False, None
    mixers = [m for m in model.modules() if isinstance(m, GatedDeltaNet2)]
    if not mixers:
        return False, None

    def make_hook(li):
        def hook(module, inp, out):
            if not capture_flag.get("on"):
                return
            hs = inp[0] if isinstance(inp, (tuple, list)) and len(inp) else None
            if not torch.is_tensor(hs) or hs.dim() != 3:
                return
            try:
                decay_store.setdefault(li, []).append(_reconstruct_log_decay(module, hs))
            except Exception as e:
                print(f"  [rbar] layer {li} reconstruction failed ({type(e).__name__}: {e})", flush=True)
        return hook

    for li, mx in enumerate(mixers):
        mx.register_forward_hook(make_hook(li))
    print(f"  [rbar] a_t capture = RECONSTRUCT (fallback) on {len(mixers)} gdn2 mixers", flush=True)
    return True, (lambda: None)   # reconstruction is per-module-hook; no counter to reset


# -------------------------------------------------------------------------------- checkpoint [PIN-1]
def _sha256_head(path, nbytes=1 << 20):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(nbytes))
    return h.hexdigest()


def resolve_and_assert_ckpt(checkpoint_path=None):
    """Resolve the checkpoint and ASSERT it is the paper-matched 100B (or 95B near-equiv). REJECT 10B.

    Returns provenance {path, realpath, size_bytes, sha256_head_1MB, token_tag}. Raises RuntimeError
    if the resolved checkpoint is the 10B (or an unrecognized) file. [PIN-1]"""
    cand = checkpoint_path or os.environ.get("GDN2_CKPT_PATH") or CKPT_100B_DEFAULT
    if cand:
        os.environ["GDN2_CKPT_PATH"] = cand
    w = gdn2_common._find_weights()
    if w is None:
        raise RuntimeError(
            f"[PIN-1] checkpoint not found (looked for {cand}). Provide model-100b.pth "
            f"(LLM-OS-Models2/gdn2-1.3b-paper-matched) via --ckpt or GDN2_CKPT_PATH.")
    w_real = os.path.realpath(w)
    base = os.path.basename(w_real).lower()
    if any(t.lower() in base for t in _REJECT_TOKENS):
        raise RuntimeError(
            f"[PIN-1] resolved checkpoint {w_real} looks like the 10B checkpoint — REJECTED. The 10B "
            f"model differs substantially from the paper-matched 100B; rank-stratification must use "
            f"model-100b.pth (95B accepted as near-equivalent).")
    tag = next((t for t in _ACCEPT_TOKENS if t.lower() in base), None)
    if tag is None:
        print(f"  [ckpt] WARNING: resolved {w_real} does not name 100b/95b — proceeding but verify it "
              f"is the paper-matched checkpoint, not 10B.", flush=True)
    elif tag.lower().startswith("95"):
        print(f"  [ckpt] NOTE: using 95B checkpoint (paper-matched near-equivalent of 100B).", flush=True)
    prov = {"path": w, "realpath": w_real, "size_bytes": os.path.getsize(w_real),
            "sha256_head_1MB": _sha256_head(w_real), "token_tag": tag or "unknown"}
    print(f"  [ckpt] [PIN-1] resolved={w_real} size={prov['size_bytes']} tag={prov['token_tag']} "
          f"sha256[:16]={prov['sha256_head_1MB'][:16]}", flush=True)
    return prov


def load(checkpoint_path=None):
    """Load gdn2-1.3B (paper-matched 100B) via the canonical loader; install a_t capture.

    Primary a_t capture = kernel-intercept on fused_recurrent_gdn2 (ground-truth g). Falls back to
    deterministic reconstruction if the kernel symbol is unavailable. Returns a Stage1Bundle whose
    `.ckpt_provenance` records the resolved path/size/hash and `.capture_mode` the a_t method."""
    prov = resolve_and_assert_ckpt(checkpoint_path)
    base = gdn2_common.load_model()          # [PIN-1] canonical: Config.from_name, strict=False, bf16, fused_recurrent
    decay_store = {}
    capture_flag = {"on": False}
    ok, reset = _install_kernel_capture(base.model, decay_store, capture_flag)
    mode = "kernel"
    if not ok:
        ok, reset = _install_reconstruct_probe(base.model, decay_store, capture_flag)
        mode = "reconstruct" if ok else "none"
    if not ok:
        reset = (lambda: None)
        print("  [rbar] WARNING: a_t capture unavailable -> r_bar regression (G1b) will be skipped, "
              "flagged as 'instrumentation-unavailable' (NOT a reproduction failure).", flush=True)
    bundle = Stage1Bundle(base, decay_store, capture_flag, reset, mode)
    bundle.ckpt_provenance = prov
    return bundle
