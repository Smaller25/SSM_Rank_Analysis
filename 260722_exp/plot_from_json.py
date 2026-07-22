"""저장된 <Dx>.json(신호 series 포함)에서 per-signal 궤적 PNG 재생성 (GPU 불필요).

사용: python plot_from_json.py 260722_results/D2.json [D3.json ...]
각 신호(주황) + GT(초록 점선) + 그 방법의 예측경계(빨강)를 겹쳐 그림.
"""
import sys, os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_rec(path):
    rec = json.load(open(path))
    grid = rec["grid"]; gt = rec.get("gt"); sigs = rec.get("signals", {})
    if not sigs:
        print(f"[skip] {path}: no signal series (run.py를 최신본으로 재실행 필요)"); return None
    names = list(sigs.keys())
    fig, axes = plt.subplots(len(names), 1, figsize=(12, 1.5 * len(names)), sharex=True)
    if len(names) == 1:
        axes = [axes]
    for ax, sn in zip(axes, names):
        vals = np.array([np.nan if v is None else v for v in sigs[sn]["values"]], float)
        ax.plot(grid, np.nan_to_num(vals), "-o", ms=2, color="darkorange")
        if gt:
            for b in gt:
                ax.axvline(b, color="green", ls="--", lw=1, alpha=.6)
        for b in rec["methods"].get(sn, {}).get("B", []):
            ax.axvline(b, color="crimson", lw=1, alpha=.6)
        f1 = rec["methods"].get(sn, {}).get("f1", float("nan"))
        ax.set_ylabel(sn, fontsize=7)
        ax.set_title(f"{sn}  (F1={f1:.2f}, {sigs[sn]['kind']})  green=GT red=pred", fontsize=7, loc="left")
    axes[-1].set_xlabel("token position t")
    fig.suptitle(f"{os.path.basename(path)}  ({rec['meta']['name']})", fontsize=9)
    fig.tight_layout()
    out = path.replace(".json", ".png")
    fig.savefig(out, dpi=110); plt.close(fig)
    print("saved", out); return out


if __name__ == "__main__":
    for p in sys.argv[1:]:
        plot_rec(p)
