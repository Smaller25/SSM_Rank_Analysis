"""F6 / F7 / S14 on the PAPER-MATCHED gdn2-1.3B (100B checkpoint), local SLURM run.

Rebuilds three state-rank figures on model-100b.pth (LLM-OS-Models2/gdn2-1.3b-paper-matched),
replacing the earlier checkpoint-10B run (10B differs substantially from paper-matched 100B).

- F6: per-head decay r_bar = exp(E[log a_t]) vs eRank scatter + theory curve.
- F7: 2x2 decomposition {real decay, g=0} x {real key, NORM-MATCHED isotropic key} -> +decay / +aniso.
- S14: eRank(S_t) trajectory by data type (natural/math/code/knowledge), multi-seq +-std band.

a_t CAPTURE = kernel-intercept: monkeypatch lit_gpt.gdn2.fused_recurrent_gdn2 and read its `g` arg
(use_gate_in_kernel=False => g is log-decay; a_t = exp(g)). This is the ground-truth capture (a
PyTorch-level hook on the write/decay path), not attribute-scraping. Loads via 260722_exp/common.py.
"""
import sys, os, math, warnings
os.environ.setdefault("GDN2_CKPT_PATH", "/home/sohyung/models/gdn2_1.3B_100b.pth")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
sys.path.insert(0, "/home/sohyung/linear-memory-routing")            # vendored fla 0.5.2
sys.path.insert(0, "/home/sohyung/SSM_Rank_Analysis/260722_exp")     # common.py (canonical loader)
warnings.filterwarnings("ignore")
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import common

OUT = "/home/sohyung/SSM_Rank_Analysis/260722_exp/260726_f6f7s14_100b"
os.makedirs(OUT, exist_ok=True)

bundle = common.load_model()          # Config.from_name, strict=False, bf16, fused_recurrent, cache
tok = common.load_tokenizer()
NL = bundle.n_layer
import lit_gpt.gdn2 as G              # lit path inserted by common.load_model
orig_fr = G.fused_recurrent_gdn2
stash = {}; cnt = {"i": 0}; cap = {"on": False}
def _wrap(**kw):                       # kernel-intercept: capture per-layer (q,k,v,g,b,w,A_log,dt_bias)
    if cap["on"]:
        li = cnt["i"] % NL; cnt["i"] += 1
        stash[li] = {k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in kw.items()}
    return orig_fr(**kw)
G.fused_recurrent_gdn2 = _wrap

def erank(M):
    s = torch.linalg.svdvals(M.float().cpu()); s = s / (s.sum() + 1e-12)
    return float(torch.exp(-(s * (s + 1e-12).log()).sum()))
def capture(ids):
    stash.clear(); cnt["i"] = 0; cap["on"] = True
    try: bundle.states(ids.to(common.DEVICE))
    finally: cap["on"] = False
def run_state(kw, gzero=False, iso_k=False):
    k = kw["k"]
    if iso_k:
        r = torch.randn_like(k); k = r * (k.norm(dim=-1, keepdim=True) / (r.norm(dim=-1, keepdim=True) + 1e-8))
    g = torch.zeros_like(kw["g"]) if gzero else kw["g"]
    o, st = orig_fr(q=kw["q"], k=k, v=kw["v"], g=g, b=kw["b"], w=kw["w"], A_log=kw["A_log"],
                    dt_bias=kw["dt_bias"], output_final_state=True,
                    use_qk_l2norm_in_kernel=True, use_gate_in_kernel=False)
    return st[0]
def state_at(kw, t):                   # state after first t tokens (kernel truncation)
    sl = lambda x: x[:, :t] if torch.is_tensor(x) and x.dim() >= 2 and x.shape[1] >= t else x
    o, st = orig_fr(q=sl(kw["q"]), k=sl(kw["k"]), v=sl(kw["v"]), g=sl(kw["g"]), b=sl(kw["b"]),
                    w=sl(kw["w"]), A_log=kw["A_log"], dt_bias=kw["dt_bias"], output_final_state=True,
                    use_qk_l2norm_in_kernel=True, use_gate_in_kernel=False)
    return st[0]

NAT = ("Rivers shape the land over millions of years, carving valleys and depositing sediment far downstream. "
"The economy of a coastal town depends on fishing, tourism, and trade, each rising and falling with the seasons. "
"A good teacher notices when a student is confused before the student says a word. Winters in the north are long. "
"Music can carry a memory more vividly than a photograph, folding years into a single chord. History is a road. ")
MATH = ("Let x be a positive integer. If 3x + 7 = 22 then x = 5. The sum 1 + 2 + ... + n equals n(n+1)/2. "
"The derivative of x^3 is 3x^2 and the integral of 2x dx is x^2 + C. A 3 4 5 triangle is right since 9 + 16 = 25. "
"Solve 2y - 4 = 10 to get y = 7. The probability of independent events multiplies. A prime has two divisors. ")
CODE = ("def fib(n):\n    a,b=0,1\n    for _ in range(n): a,b=b,a+b\n    return a\n"
"class Stack:\n    def __init__(self): self.xs=[]\n    def push(self,x): self.xs.append(x)\n    def pop(self): return self.xs.pop()\n"
"for i in range(10):\n    if i%2==0: print(i)\n    else: continue\n")
KNOW = ("The capital of France is Paris. Water boils at 100 degrees Celsius at sea level. Everest is the highest mountain. "
"The human heart has four chambers. The speed of light is about 299792 km per second. Photosynthesis converts sunlight. "
"DNA carries genetic information. The Great Wall of China spans thousands of kilometers. Gold is a dense metal. ")

def toks(txt, n=512, rep=4):
    ids = tok(txt * rep, return_tensors="pt").input_ids[:, :n]
    return ids
LAYERS = [int(round(x)) for x in np.linspace(2, NL - 2, 6)]; LAYERS = sorted(set(LAYERS))
print("n_layer", NL, "LAYERS", LAYERS, flush=True)

# ================= F6 + F7 (natural input, kernel-intercept a_t, norm-matched iso-k) =================
capture(toks(NAT))
dk = stash[LAYERS[0]]["k"].shape[-1]
rb = []; er6 = []
for li in LAYERS:
    g = stash[li]["g"]; H = stash[li]["k"].shape[-2]; st = run_state(stash[li])
    for h in range(H):
        gh = g[..., h, :] if g.dim() >= 3 else g
        rb.append(float(torch.exp(gh.float().mean()))); er6.append(erank(st[h]))
rb = np.array(rb); er6 = np.array(er6)
np.save(f"{OUT}/F6_data.npy", {"rbar": rb, "erank": er6, "cap": dk})
fig, ax = plt.subplots(figsize=(6.2, 4.6))
col = np.where(rb >= 0.99, '#c0392b', np.where(rb >= 0.9, '#e67e22', '#2471a3'))
ax.scatter(rb, er6, c=col, s=22, alpha=.6)
xs = np.linspace(0.02, 0.995, 200); ax.plot(xs, np.minimum(dk, math.e / (1 - xs)), 'k--', lw=1,
        label='theory e/(1-r̄), cap %d' % dk)
ax.set_xlabel("r̄ = exp(E[log a_t])  (per head)"); ax.set_ylabel("effective rank of state")
ax.set_title("F6 (gdn2-1.3B 100B, natural): decay vs eRank per head"); ax.legend(fontsize=8)
ax.set_ylim(0, min(dk, 90)); plt.tight_layout(); plt.savefig(f"{OUT}/F6_100b.png", dpi=150)
print("F6 erank %.1f-%.1f cap %d" % (er6.min(), er6.max(), dk), flush=True)

torch.manual_seed(0)
conds = {"real_g+real_k": (0, 0), "g=1+real_k": (1, 0), "real_g+iso_k": (0, 1), "g=1+iso_k": (1, 1)}
res = {c: [] for c in conds}
for li in LAYERS:
    kw = stash[li]
    for c, (zg, rk) in conds.items():
        st = run_state(kw, gzero=bool(zg), iso_k=bool(rk))
        res[c].append(np.mean([erank(st[h]) for h in range(st.shape[0])]))
mn = {c: float(np.mean(v)) for c, v in res.items()}
dec = mn["g=1+real_k"] - mn["real_g+real_k"]; ani = mn["real_g+iso_k"] - mn["real_g+real_k"]
inter = mn["g=1+iso_k"] - mn["real_g+real_k"] - dec - ani
np.save(f"{OUT}/F7_data.npy", res)
print("F7 real %.2f | +decay %.2f | +aniso %.2f | interaction %.2f | both %.2f"
      % (mn["real_g+real_k"], dec, ani, inter, mn["g=1+iso_k"]), flush=True)
fig, ax = plt.subplots(figsize=(5.4, 3.9)); labels = list(conds); vals = [mn[c] for c in labels]
ax.bar(range(4), vals, color=["#2471a3", "#e67e22", "#c0392b", "#7f8c8d"])
ax.set_xticks(range(4)); ax.set_xticklabels(["real g\nreal k", "g=1\nreal k", "real g\niso k\n(norm-matched)", "g=1\niso k"], fontsize=8)
ax.set_ylabel("effective rank of state")
ax.set_title("F7 (gdn2-1.3B 100B, natural, norm-matched iso-k)\n+decay %.1f  +aniso %.1f  interaction %.1f" % (dec, ani, inter), fontsize=10)
for i, v in enumerate(vals): ax.text(i, v + 0.15, "%.1f" % v, ha="center", fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/F7_100b.png", dpi=150); print("F7 saved", flush=True)

# ================================ S14 (data-type trajectories, multi-seq band) ================================
DATA = {"natural": NAT, "math": MATH, "code": CODE, "knowledge": KNOW}
LAYERS_T = [l for l in LAYERS if l <= NL - 2][:3] or LAYERS[:3]
T = 256; POS = [16, 32, 48, 64, 96, 128, 160, 192, 224, 256]; NSEQ = 4
s14 = {}
for name, txt in DATA.items():
    big = tok(txt * 12, return_tensors="pt").input_ids[0]
    seqs = [big[s * T:s * T + T].unsqueeze(0) for s in range(NSEQ) if s * T + T <= big.shape[0]]
    trajs = []
    for ids in seqs:
        capture(ids)
        tr = [float(np.mean([np.mean([erank(state_at(stash[li], t)[h]) for h in range(stash[li]["k"].shape[-2])])
                             for li in LAYERS_T])) for t in POS]
        trajs.append(tr)
    trajs = np.array(trajs); s14[name] = {"pos": POS, "mean": trajs.mean(0).tolist(), "std": trajs.std(0).tolist()}
    print("S14 %s last=%.2f n_seq=%d" % (name, trajs.mean(0)[-1], len(seqs)), flush=True)
np.save(f"{OUT}/S14_data.npy", s14)
fig, ax = plt.subplots(figsize=(7, 4.6)); cols = {"natural": "#2471a3", "math": "#c0392b", "code": "#27ae60", "knowledge": "#e67e22"}
for name in DATA:
    r = s14[name]; m_ = np.array(r["mean"]); sd = np.array(r["std"])
    ax.plot(POS, m_, "o-", color=cols[name], label=name); ax.fill_between(POS, m_ - sd, m_ + sd, color=cols[name], alpha=.15)
ax.set_xlabel("sequence position t"); ax.set_ylabel("state eRank(S_t)  (layers %s, head-mean)" % LAYERS_T)
ax.set_title("S14 (gdn2-1.3B 100B): eRank trajectory by data type\n(mean ± std over %d sequences per type)" % NSEQ)
ax.legend(fontsize=9); plt.tight_layout(); plt.savefig(f"{OUT}/S14_100b.png", dpi=150)
print("ALL_DONE", flush=True)
