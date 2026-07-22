"""랩미팅용 요약 figure: chunk-boundary 신호 × 데이터셋 (F1/z 히트맵 + D6/D7 막대 + 요약).

사용: python make_lab_figure.py [results_dir] [out.png]
"""
import sys, os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "260722_results")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "260722_labmeeting_summary.png")

GROUPS = [("control", ["fixed", "log"]),
          ("existing (DLA)", ["DLA_frobenius"]),
          ("info-score", ["surprisal", "epiplexity"]),
          ("state-level (ours)", ["erank", "state_entropy", "nuclear", "participation", "numrank"])]
METHODS = [m for _, ms in GROUPS for m in ms]
GCOLOR = {"control": "#9e9e9e", "existing (DLA)": "#d1495b",
          "info-score": "#edae49", "state-level (ours)": "#2e86ab"}
M2G = {m: g for g, ms in GROUPS for m in ms}
DSETS = ["D2", "D3", "D4", "D5", "D6", "D7"]
DLABEL = {"D2": "D2\nstruct", "D3": "D3\ntopic", "D4": "D4\nlang",
          "D5": "D5\nMQAR", "D6": "D6\nNIAH-toy", "D7": "D7\nNIAH-real"}

recs = {}
for d in DSETS:
    p = os.path.join(RES, f"{d}.json")
    if os.path.exists(p):
        recs[d] = json.load(open(p))
DSETS = [d for d in DSETS if d in recs]

F1 = np.full((len(METHODS), len(DSETS)), np.nan)
Z = np.full((len(METHODS), len(DSETS)), np.nan)
NULL = {}
for j, d in enumerate(DSETS):
    ms = recs[d]["methods"]
    NULL[d] = np.mean([a.get("null_f1", 0) for a in ms.values()])
    for i, m in enumerate(METHODS):
        if m in ms:
            F1[i, j] = ms[m]["f1"]; Z[i, j] = ms[m]["z"]

fig = plt.figure(figsize=(16, 10))
gs = GridSpec(2, 3, height_ratios=[1.25, 1.0], hspace=0.4, wspace=0.5)


def heat(ax, M, title, cmap, vmin, vmax, center=None):
    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(DSETS))); ax.set_xticklabels([DLABEL[d] for d in DSETS], fontsize=8.5)
    ax.tick_params(axis="x", pad=2)
    ax.set_yticks(range(len(METHODS))); ax.set_yticklabels(METHODS, fontsize=8)
    for i in range(len(METHODS)):
        for j in range(len(DSETS)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if (cmap == "viridis" and M[i, j] < (vmin + vmax) / 2) else "black")
    # 그룹 경계선 + 색 라벨
    off = 0
    for g, mm in GROUPS:
        ax.add_patch(plt.Rectangle((-0.5, off - 0.5), len(DSETS), len(mm), fill=False,
                                   edgecolor=GCOLOR[g], lw=2.5))
        off += len(mm)
    ax.set_title(title, fontsize=11, weight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


axf = fig.add_subplot(gs[0, 0]); heat(axf, F1, "Boundary-recovery F1 (count-matched)", "viridis", 0, 1)
axz = fig.add_subplot(gs[0, 1]); heat(axz, Z, "z vs random-boundary null", "RdBu_r", -3, 3)

# 범례 (그룹 색)
axl = fig.add_subplot(gs[0, 2]); axl.axis("off")
for k, (g, _) in enumerate(GROUPS):
    axl.add_patch(plt.Rectangle((0.05, 0.85 - k * 0.09), 0.08, 0.05, color=GCOLOR[g]))
    axl.text(0.16, 0.875 - k * 0.09, g, fontsize=10, va="center")
axl.text(0.02, 0.45,
         "Read F1 only where null is low\n(D6/D7). D2 dense-GT (null≈.91)\n& D3–D5 sparse-GT → use z.\n\n"
         "null_f1 per dataset:\n" + "  ".join(f"{d}:{NULL[d]:.2f}" for d in DSETS),
         fontsize=9, va="top", family="monospace")
axl.set_title("legend / caveats", fontsize=10)


def bars(ax, d):
    ms = recs[d]["methods"]
    items = sorted([(m, ms[m]["f1"]) for m in METHODS if m in ms], key=lambda kv: -kv[1])
    names = [m for m, _ in items]; vals = [v for _, v in items]
    cols = [GCOLOR[M2G[m]] for m in names]
    ax.bar(range(len(names)), vals, color=cols)
    ax.axhline(NULL[d], ls="--", color="k", lw=1, label=f"null={NULL[d]:.2f}")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05); ax.set_ylabel("F1"); ax.legend(fontsize=8)
    ax.set_title(f"{d}  ({recs[d]['meta']['name']}, nGT={0 if recs[d]['gt'] is None else len(recs[d]['gt'])})",
                 fontsize=10, weight="bold")


bars(fig.add_subplot(gs[1, 0]), "D6")
bars(fig.add_subplot(gs[1, 1]), "D7")

axt = fig.add_subplot(gs[1, 2]); axt.axis("off")
takeaway = (
    "Takeaways (gdn2-1.3B)\n"
    "──────────────────────\n"
    "• Existing dynamic (DLA ‖ΔRMS-state‖) is\n"
    "  WEAK everywhere — worst on D6 (0.40),\n"
    "  0.0 on real RULER D7.\n"
    "• State-level signals WIN on toy NIAH\n"
    "  (D6: state_entropy 0.91 > all) …\n"
    "  but COLLAPSE on real MK-NIAH (D7: 0).\n"
    "• Only surprisal/epiplexity keep signal on\n"
    "  D7 (0.25); epiplexity best on language\n"
    "  (D4: 1.0).\n"
    "• fixed/log score only by coincidence.\n"
    "• D5 (MQAR) & D7 (real) unsolved by ALL\n"
    "  → open problem: capacity-aware signal.")
axt.text(0.0, 1.0, takeaway, fontsize=9.5, va="top", family="monospace")

fig.suptitle("Which signal builds linear-attention multi-states correctly?  "
             "chunk-boundary signal vs GT  (gdn2-1.3B)", fontsize=13, weight="bold")
fig.savefig(OUT, dpi=130, bbox_inches="tight")
print("saved", OUT)
