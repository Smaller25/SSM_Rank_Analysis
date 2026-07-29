"""From-scratch MQAR study of update rules: state-saturation signals, decay vs key-subspace.

Trains tiny LMs (identical backbone, swappable fla token-mixer) from scratch on synthetic MQAR,
then measures, per update rule, recall(N) and state eRank(N) as load N grows. Answers:
  (1) which signal marks state saturation per model,
  (2) decay comparison (no-decay LinearAttention/DeltaNet vs decay GLA/RetNet/GatedDeltaNet/Mamba2),
  (3) is the low (~10%) eRank due to the decay term or a key subspace? -> if the *no-decay* rules ALSO
      show low state eRank, it is not decay.
Robust: each mixer in try/except; results written incrementally to state-saturation_results/.
Run on GPU (fla Triton kernels). Usage: python mqar_fromscratch.py [--smoke]
"""
import os, sys, json, math, argparse, time, contextlib
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

import fla.layers as FLA
from fla.modules import RMSNorm
from fla.models.utils import Cache

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state_saturation_results")
os.makedirs(RESULTS, exist_ok=True)

KEY_VOCAB, VAL_VOCAB = 256, 64
KEY_OFF, VAL_OFF = 0, KEY_VOCAB
VOCAB = KEY_VOCAB + VAL_VOCAB

# ------------------------------------------------------------------ MQAR data
def make_mqar(batch, n_pairs, gen):
    B, N = batch, n_pairs
    x = torch.empty(B, 3 * N, dtype=torch.long)
    tgt = torch.empty(B, N, dtype=torch.long)
    for b in range(B):
        keys = gen.choice(KEY_VOCAB, size=N, replace=False)
        vals = gen.integers(0, VAL_VOCAB, size=N)
        seq = np.empty(2 * N, dtype=np.int64); seq[0::2] = keys + KEY_OFF; seq[1::2] = vals + VAL_OFF
        order = gen.permutation(N)
        x[b, :2 * N] = torch.from_numpy(seq)
        x[b, 2 * N:] = torch.from_numpy(keys[order] + KEY_OFF)
        tgt[b] = torch.from_numpy(vals[order])
    qpos = torch.arange(2 * N, 3 * N)
    return x.to(DEVICE), qpos.to(DEVICE), tgt.to(DEVICE)

# ------------------------------------------------------------------ tiny model
class Block(nn.Module):
    def __init__(self, d, mixer):
        super().__init__()
        self.n1, self.mix, self.n2 = RMSNorm(d), mixer, RMSNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, h, past=None, use_cache=False):
        o = self.mix(self.n1(h), past_key_values=past, use_cache=use_cache)
        o = o[0] if isinstance(o, tuple) else o
        h = h + o
        return h + self.mlp(self.n2(h))

class TinyLM(nn.Module):
    def __init__(self, d, n_layer, ctor):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, d)
        self.blocks = nn.ModuleList([Block(d, ctor(i)) for i in range(n_layer)])
        self.norm, self.head = RMSNorm(d), nn.Linear(d, VOCAB, bias=False)

    def forward(self, x, past=None, use_cache=False):
        h = self.embed(x)
        for b in self.blocks:
            h = b(h, past, use_cache)
        return self.head(self.norm(h))

# ------------------------------------------------------------------ mixers (decay label)
def mixers(d, H):
    return {
        "LinearAttention":      ("no-decay", lambda i: FLA.LinearAttention(hidden_size=d, num_heads=H, layer_idx=i)),
        "DeltaNet":             ("no-decay", lambda i: FLA.DeltaNet(hidden_size=d, num_heads=H, layer_idx=i, use_gate=False)),
        "GatedLinearAttention": ("decay",    lambda i: FLA.GatedLinearAttention(hidden_size=d, num_heads=H, layer_idx=i)),
        "MultiScaleRetention":  ("decay",    lambda i: FLA.MultiScaleRetention(hidden_size=d, num_heads=H, layer_idx=i)),
        "GatedDeltaNet":        ("decay",    lambda i: FLA.GatedDeltaNet(hidden_size=d, num_heads=H, layer_idx=i)),
        "Mamba2":               ("decay",    lambda i: FLA.Mamba2(hidden_size=d, num_heads=H, layer_idx=i)),
    }

# ------------------------------------------------------------------ signals
def erank(m):
    s = np.linalg.svd(m, compute_uv=False); s = s / (s.sum() + 1e-12)
    return float(np.exp(-np.sum(s * np.log(s + 1e-12))))

@torch.no_grad()
def state_erank(model, x, n_layer):
    cache = Cache()
    model(x, past=cache, use_cache=True)
    ers, mx = [], None
    for i in range(n_layer):
        try:
            rs = cache[i]["recurrent_state"]
        except Exception:
            continue
        elems = rs if isinstance(rs, (list, tuple)) else [rs]
        for e in elems:
            if torch.is_tensor(e) and e.dim() >= 2:
                t = e.detach().float().cpu().reshape(-1, e.shape[-2], e.shape[-1])
                ers += [erank(t[h].numpy()) for h in range(t.shape[0])]
                mx = min(t.shape[-2], t.shape[-1])
    return (float(np.mean(ers)) if ers else float("nan")), mx

@torch.no_grad()
def recall(model, N, gen, n_batch=4, bs=64):
    hit = tot = 0
    for _ in range(n_batch):
        x, qpos, tgt = make_mqar(bs, N, gen)
        logits = model(x)[:, qpos, VAL_OFF:VAL_OFF + VAL_VOCAB]
        hit += (logits.argmax(-1) == tgt).sum().item(); tot += tgt.numel()
    return hit / tot

# ------------------------------------------------------------------ train + eval one mixer
def run_mixer(name, decay, ctor, d, H, n_layer, steps, bs, n_grid, train_nmax, seed=0):
    torch.manual_seed(seed)
    gen = np.random.default_rng(seed)
    model = TinyLM(d, n_layer, ctor).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    model.train()
    t0 = time.time()
    for step in range(steps):
        N = int(gen.integers(4, train_nmax + 1))
        x, qpos, tgt = make_mqar(bs, N, gen)
        logits = model(x)[:, qpos, :]
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), (tgt + VAL_OFF).reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    egen = np.random.default_rng(seed + 999)
    rows = []
    for N in n_grid:
        rc = recall(model, N, egen)
        x, _, _ = make_mqar(8, N, egen)
        er, mx = state_erank(model, x, n_layer)
        rows.append({"N": N, "recall": rc, "state_erank": er, "erank_maxrank": mx,
                     "erank_norm": (er / mx if (mx and not math.isnan(er)) else None)})
    return {"mixer": name, "decay": decay, "d_model": d, "n_head": H, "n_layer": n_layer,
            "steps": steps, "train_loss": float(loss.item()), "minutes": round((time.time() - t0) / 60, 2),
            "rows": rows}

# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        d, H, n_layer, steps, bs = 64, 2, 1, 20, 8
        grid, tnmax, names = [4, 8], 8, ["LinearAttention", "DeltaNet"]
    else:
        d, H, n_layer, steps, bs = 256, 4, 2, 4000, 64
        grid, tnmax = [4, 8, 16, 32, 48, 64, 96, 128], 64
        names = list(mixers(d, H).keys())

    reg = mixers(d, H)
    results = {}
    out = os.path.join(RESULTS, "smoke.json" if a.smoke else "state_saturation.json")
    for name in names:
        decay, ctor = reg[name]
        print(f"\n=== {name} ({decay}) ===", flush=True)
        try:
            r = run_mixer(name, decay, ctor, d, H, n_layer, steps, bs, grid, tnmax)
            results[name] = r
            for row in r["rows"]:
                print(f"  N={row['N']:>3} recall={row['recall']:.3f} eRank={row['state_erank']} "
                      f"norm={row['erank_norm']}", flush=True)
            print(f"  train_loss={r['train_loss']:.3f} time={r['minutes']}min", flush=True)
        except Exception as e:
            import traceback; results[name] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
            print(f"  FAILED: {results[name]['error']}", flush=True); traceback.print_exc()
        with open(out, "w") as f:
            json.dump(results, f, indent=2)   # incremental
    if not a.smoke:
        plot_results(results)
    print("\nDONE ->", out, flush=True)

def plot_results(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ok = {k: v for k, v in results.items() if "rows" in v}
    if not ok:
        return
    col = {"no-decay": "C0", "decay": "C3"}
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for name, r in ok.items():
        Ns = [x["N"] for x in r["rows"]]
        ls = "-" if r["decay"] == "no-decay" else "--"
        c = col[r["decay"]]
        ax[0].plot(Ns, [x["recall"] for x in r["rows"]], ls, marker="o", label=f"{name} ({r['decay']})")
        en = [x["erank_norm"] if x["erank_norm"] is not None else float("nan") for x in r["rows"]]
        ax[1].plot(Ns, en, ls, marker="s", label=name)
    ax[0].axhline(1 / VAL_VOCAB, color="k", ls=":", lw=.7)
    ax[0].set_title("recall vs load N (capacity)"); ax[0].set_ylabel("recall")
    ax[1].set_title("normalized state eRank vs N  (no-decay solid vs decay dashed)")
    ax[1].set_ylabel("state eRank / max rank")
    for a_ in ax:
        a_.set_xscale("log", base=2); a_.set_xlabel("# k-v pairs N"); a_.grid(alpha=.3); a_.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "state_saturation.png"), dpi=120)

if __name__ == "__main__":
    main()
