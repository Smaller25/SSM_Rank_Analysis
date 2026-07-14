"""Pretrained update-rule comparison on in-context MQAR: recall(N) + state eRank(N).

Loads well-trained checkpoints across 3 stacks and measures, per update rule, associative recall and
recurrent-state effective rank as load N grows. Answers: state-saturation signal per model; decay
comparison; and whether the low (~10%) eRank is decay- or key-subspace-driven (DeltaNet is the well-
trained NO-DECAY model — if its eRank is high while decay models are low, decay suppresses eRank).

Models (decay label):
  mamba2-370m   (SSD, input-dep decay)      -- mamba_ssm
  gdn2-370m     (gated delta, decay)        -- lit_gpt (long-gdn/dsc), checkpoint .pth
  deltanet-1.3B (delta rule, NO decay)      -- fla-hub
  gla-1.3B      (gated linear attn, decay)  -- fla-hub
  retnet-1.3B   (fixed decay)               -- fla-hub
Robust: per-model try/except; incremental JSON. GPU + Triton.
"""
import os, sys, json, math, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from capacity_utils import effective_rank

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "decay_mqar_results")
os.makedirs(RESULTS, exist_ok=True)

KEY_OFF, VAL_OFF, VAL_VOCAB = 1000, 5000, 64   # ids < 32000 (all model vocabs >= 32000)

MODELS = [
    ("mamba2-370m",   "decay",    "mamba2", "state-spaces/mamba2-370m"),
    ("gdn2-370m",     "decay",    "gdn2",   "gyung/gdn2-370m-fineweb-edu-100b"),
    ("deltanet-1.3B", "no-decay", "fla",    "fla-hub/delta_net-1.3B-100B"),
    ("gla-1.3B",      "decay",    "fla",    "fla-hub/gla-1.3B-100B"),
    ("retnet-1.3B",   "decay",    "fla",    "fla-hub/retnet-1.3B-100B"),
]
FLA_CLASSES = {
    "fla-hub/delta_net-1.3B-100B": ("fla.models.delta_net", "DeltaNetConfig", "DeltaNetForCausalLM"),
    "fla-hub/gla-1.3B-100B":       ("fla.models.gla",       "GLAConfig",       "GLAForCausalLM"),
    "fla-hub/retnet-1.3B-100B":    ("fla.models.retnet",    "RetNetConfig",    "RetNetForCausalLM"),
}
GDN2_CKPT = ("/data2/sohyung/hf_home/hub/models--gyung--gdn2-370m-fineweb-edu-100b/"
             "snapshots/2d859d6c96606314a2156d3decde84f7d2ad6510/checkpoint-6B-model-ckpt.pth")


class Bundle:
    def __init__(self, kind, model, extra=None):
        self.kind, self.model, self.extra = kind, model, extra

    @torch.no_grad()
    def logits(self, ids):
        ids = ids.to(DEVICE)
        if self.kind == "mamba2":
            return self.model(ids).logits.float()
        if self.kind == "gdn2":
            return self.model(ids).float()
        return self.model(ids).logits.float()          # fla

    @torch.no_grad()
    def states(self, ids):
        ids = ids.to(DEVICE)
        if self.kind == "mamba2":
            return _mamba2_states(self.model, ids)
        if self.kind == "gdn2":
            return _gdn2_states(self.model, ids, self.extra)   # extra = (SHARED, n_layer)
        return _cache_states(self.model, ids, self.model.config.num_hidden_layers)  # fla


def _reshape_state(rs):
    elems = rs if isinstance(rs, (list, tuple)) else [rs]
    mats = [e.detach().float().cpu().reshape(-1, e.shape[-2], e.shape[-1])
            for e in elems if torch.is_tensor(e) and e.dim() >= 2]
    return torch.cat(mats, 0) if mats else None


def _cache_states(model, ids, n_layer):
    out = model(ids, use_cache=True)
    cache = out.past_key_values
    st = {}
    for i in range(n_layer):
        try:
            r = _reshape_state(cache[i]["recurrent_state"])
        except Exception:
            r = None
        if r is not None:
            st[i] = r
    return st


def _mamba2_states(model, ids):
    from mamba_ssm.utils.generation import InferenceParams
    ip = InferenceParams(max_seqlen=ids.shape[1], max_batch_size=ids.shape[0])
    model(ids, inference_params=ip)
    return {li: ss[0].detach().float().cpu() for li, (cs, ss) in ip.key_value_memory_dict.items()}


def _gdn2_states(model, ids, extra):
    SHARED, n_layer = extra
    from fla.models.utils import Cache
    SHARED["cache"] = Cache()
    model(ids)
    st = {}
    for i in range(n_layer):
        try:
            r = _reshape_state(SHARED["cache"][i]["recurrent_state"])
        except Exception:
            r = None
        if r is not None:
            st[i] = r
    return st


def load(kind, hf):
    if kind == "mamba2":
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
        m = MambaLMHeadModel.from_pretrained(hf, device=DEVICE, dtype=torch.float32).eval()
        return Bundle("mamba2", m)
    if kind == "fla":
        import fla, importlib  # noqa
        mod, cfgn, mdln = FLA_CLASSES[hf]
        M = importlib.import_module(mod)
        Cfg, Mdl = getattr(M, cfgn), getattr(M, mdln)
        try: Mdl._tied_weights_keys = {}
        except Exception: pass
        m = Mdl.from_pretrained(hf, torch_dtype=torch.bfloat16).to(DEVICE).eval()
        return Bundle("fla", m)
    if kind == "gdn2":
        sys.path.insert(0, "/home/sohyung/long-gdn/dsc")
        from lit_gpt.config import Config
        from lit_gpt.model import GPT
        from lit_gpt.gdn2 import GatedDeltaNet2
        cfg = Config.from_name("gdn2_370M")
        m = GPT(cfg).to(DEVICE).to(torch.bfloat16).eval()
        ck = torch.load(GDN2_CKPT, map_location="cpu", weights_only=False)
        sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        m.load_state_dict(sd, strict=False)
        SHARED = {"cache": None}
        _orig = GatedDeltaNet2.forward
        def patched(self, hidden_states, attention_mask=None, past_key_values=None, use_cache=False, **kw):
            pkv = SHARED["cache"] if SHARED["cache"] is not None else past_key_values
            uc = True if SHARED["cache"] is not None else use_cache
            return _orig(self, hidden_states, attention_mask=attention_mask, past_key_values=pkv, use_cache=uc, **kw)
        GatedDeltaNet2.forward = patched
        for i, mm in enumerate([x for x in m.modules() if isinstance(x, GatedDeltaNet2)]):
            mm.layer_idx = i
        return Bundle("gdn2", m, extra=(SHARED, cfg.n_layer))
    raise ValueError(kind)


def make_mqar(bs, N, gen):
    x = torch.empty(bs, 3 * N, dtype=torch.long); tgt = torch.empty(bs, N, dtype=torch.long)
    for b in range(bs):
        keys = gen.choice(4000, size=N, replace=False) + KEY_OFF
        vals = gen.integers(0, VAL_VOCAB, size=N)
        seq = np.empty(2 * N, dtype=np.int64); seq[0::2] = keys; seq[1::2] = vals + VAL_OFF
        order = gen.permutation(N)
        x[b, :2 * N] = torch.from_numpy(seq); x[b, 2 * N:] = torch.from_numpy(keys[order])
        tgt[b] = torch.from_numpy(vals[order])
    return x, torch.arange(2 * N, 3 * N), tgt


@torch.no_grad()
def eval_model(bundle, N_grid, seeds=3, bs=16):
    val_ids = torch.arange(VAL_OFF, VAL_OFF + VAL_VOCAB, device=DEVICE)
    rows = []
    for N in N_grid:
        rc, ers, mxs = [], [], []
        for s in range(seeds):
            gen = np.random.default_rng(1000 * N + s)
            x, qpos, tgt = make_mqar(bs, N, gen)
            lg = bundle.logits(x)[:, qpos][:, :, val_ids]
            rc.append((lg.argmax(-1).cpu() == tgt).float().mean().item())
            st = bundle.states(x[:1])
            for S in st.values():
                ers += [effective_rank(S[h].numpy()) for h in range(S.shape[0])]
                mxs.append(min(S.shape[-2], S.shape[-1]))
        er = float(np.mean(ers)) if ers else float("nan")
        mx = int(np.median(mxs)) if mxs else None
        rows.append({"N": N, "recall": float(np.mean(rc)), "state_erank": er,
                     "maxrank": mx, "erank_norm": (er / mx if mx else None)})
    return rows


def main():
    N_grid = [4, 8, 16, 32, 48, 64, 96, 128]
    results = {}
    out = os.path.join(RESULTS, "decay_mqar.json")
    for name, decay, kind, hf in MODELS:
        print(f"\n=== {name} ({decay}, {kind}) ===", flush=True)
        try:
            t0 = time.time()
            b = load(kind, hf)
            rows = eval_model(b, N_grid)
            results[name] = {"decay": decay, "kind": kind, "rows": rows,
                             "minutes": round((time.time() - t0) / 60, 2)}
            for r in rows:
                print(f"  N={r['N']:>3} recall={r['recall']:.3f} eRank={r['state_erank']:.2f} "
                      f"norm={r['erank_norm']}", flush=True)
            del b; torch.cuda.empty_cache()
        except Exception as e:
            import traceback; results[name] = {"decay": decay, "error": f"{type(e).__name__}: {str(e)[:200]}"}
            print("  FAILED:", results[name]["error"], flush=True); traceback.print_exc()
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
    plot(results)
    print("\nDONE ->", out, flush=True)


def plot(results):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ok = {k: v for k, v in results.items() if "rows" in v}
    if not ok:
        return
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for name, r in ok.items():
        Ns = [x["N"] for x in r["rows"]]
        ls = "-" if r["decay"] == "no-decay" else "--"
        ax[0].plot(Ns, [x["recall"] for x in r["rows"]], ls, marker="o", label=f"{name} ({r['decay']})")
        en = [x["erank_norm"] if x["erank_norm"] is not None else float("nan") for x in r["rows"]]
        ax[1].plot(Ns, en, ls, marker="s", label=name)
    ax[0].axhline(1 / VAL_VOCAB, color="k", ls=":", lw=.7)
    ax[0].set_title("in-context MQAR recall vs N"); ax[0].set_ylabel("recall")
    ax[1].set_title("normalized state eRank vs N  (no-decay solid, decay dashed)")
    ax[1].set_ylabel("state eRank / max rank")
    for a_ in ax:
        a_.set_xscale("log", base=2); a_.set_xlabel("# k-v pairs N"); a_.grid(alpha=.3); a_.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "decay_mqar.png"), dpi=120)


if __name__ == "__main__":
    main()
