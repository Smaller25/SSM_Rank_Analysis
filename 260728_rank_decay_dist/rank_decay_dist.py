"""Per-head threshold-rank + decay r̄ distributions on real gdn2-1.3B (100B), real text.
Shows: (a) rank histogram (stratification shape), (b) decay r̄ histogram, (c) rank-vs-decay scatter.
Reuses loader_gdn2 (kernel-intercept a_t) + rank_metrics + data_stage1 from the Stage-1/2 folders."""
import sys, os
S1="/home/sohyung/SSM_Rank_Analysis/260725_stage1_rank_stratification"
sys.path.insert(0,S1); sys.path.insert(0,"/home/sohyung/SSM_Rank_Analysis/legacy/260722_exp")
sys.path.insert(0,"/home/sohyung/linear-memory-routing")
os.environ.setdefault("GDN2_CKPT_PATH","/home/sohyung/models/gdn2_1.3B_100b.pth")
import numpy as np, torch, warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import loader_gdn2, rank_metrics, data_stage1, common
OUT="/home/sohyung/SSM_Rank_Analysis/260728_rank_decay_dist"
bundle=loader_gdn2.load(); tok=common.load_tokenizer()
data_stage1.set_data_seed(0)
d=data_stage1.load_all(tok, seq_len=2048, n_seq=8, which=["wikitext","github","arxiv"])
ranks=[]; rbars=[]  # per (domain,layer,head) then average over domains per (layer,head)
acc={}  # (layer,head) -> {"rank":[], "rbar":[]}
for dom,(ids_list,meta) in d.items():
    for ids in ids_list[:8]:
        states,rbar=bundle.states_and_rbar(ids.to(common.DEVICE))
        for li,S in states.items():
            for h in range(S.shape[0]):
                key=(li,h)
                acc.setdefault(key,{"rank":[],"rbar":[]})
                acc[key]["rank"].append(rank_metrics.threshold_rank(S[h].numpy() if hasattr(S[h],'numpy') else S[h]))
                if li in rbar and h < len(rbar[li]): acc[key]["rbar"].append(float(rbar[li][h]))
rows=[]
for (li,h),v in acc.items():
    r=float(np.mean(v["rank"])); rb=float(np.mean(v["rbar"])) if v["rbar"] else float("nan")
    rows.append((li,h,r,rb))
R=np.array([x[2] for x in rows]); RB=np.array([x[3] for x in rows])
cap=128
np.save(f"{OUT}/rank_decay_per_head.npy",{"rows":rows,"cap":cap})
print("n_heads",len(rows),"rank range %.1f-%.1f"%(R.min(),R.max()),"rbar range %.3f-%.3f"%(np.nanmin(RB),np.nanmax(RB)),flush=True)
fig,ax=plt.subplots(1,3,figsize=(15,4.2))
ax[0].hist(R,bins=40,color="#2471a3"); ax[0].set_xlabel("threshold-rank (ε=1e-4)"); ax[0].set_ylabel("# heads"); ax[0].set_title("(a) rank distribution (cap %d)"%cap)
ax[1].hist(RB[np.isfinite(RB)],bins=40,color="#c0392b"); ax[1].set_xlabel("decay r̄ = exp(E[log a_t])"); ax[1].set_title("(b) decay distribution")
m=np.isfinite(RB); ax[2].scatter(RB[m],R[m],s=10,alpha=.4,color="#7f3fbf")
from numpy import corrcoef
rho=float(np.corrcoef(RB[m],R[m])[0,1])
ax[2].set_xlabel("decay r̄"); ax[2].set_ylabel("threshold-rank"); ax[2].set_title("(c) rank vs decay  (Pearson r=%.2f)"%rho)
plt.tight_layout(); plt.savefig(f"{OUT}/rank_decay_dist.png",dpi=140)
print("PEARSON rank~decay = %.3f"%rho,flush=True); print("ALL_DONE",flush=True)
