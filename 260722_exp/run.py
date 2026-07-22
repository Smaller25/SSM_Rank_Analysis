"""multi-state 생성 방법론 비교 (GT 정렬).

방법: fixed / log (control) · DLA_frobenius (기존 dynamic) · surprisal · epiplexity ·
      erank · state_entropy · nuclear · participation · numrank (state-signal dynamic).
각 방법에 |GT|개 경계를 주고(count-matched) 위치 F1 + null 대비 z 를 잰다.

사용:  python run.py                       # D2~D7 (GT 있는 것)
       python run.py --datasets D2 D5 D7  --stride 8 --plot
       python run.py --niah-len 2048
결과:  260722_results/<Dx>.json  (+ --plot 시 <Dx>.png), 요약표는 stdout.
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import common, data, analysis

METHODS_SIGNAL = ["DLA_frobenius", "surprisal", "epiplexity", "erank",
                  "state_entropy", "nuclear", "participation", "numrank"]


def run_dataset(name, bundle, tok, stride, niah_len, layer_stride, plot, out_dir):
    make = data.ALL[name]
    ii, gt, meta = make(tok, max_seq_length=niah_len) if name == "D7" else make(tok)
    T = ii.shape[1]
    w = max(16, 2 * stride)
    print(f"\n=== {name} ({meta['name']}) T={T} n_GT={0 if gt is None else len(gt)} w={w} ===")

    nll = analysis.token_nll_bits(bundle, ii)
    grid, feats, dla = analysis.state_trajectory(bundle, ii, stride=stride, layer_stride=layer_stride)
    sigs = analysis.build_signals(grid, feats, dla, nll)

    rec = {"meta": meta, "T": int(T), "stride": stride, "tolerance_w": w,
           "gt": gt, "grid": grid, "methods": {}}

    if gt is None:  # D1: 포화 캘리브레이션만
        er = feats["erank"]; sat = float(er[-1])
        idx = int(np.argmax(er >= 0.95 * sat)) if sat > 0 else -1
        rec["saturation"] = {"erank_final": sat, "erank95_pos": grid[idx] if idx >= 0 else None,
                             "dla_series": dla.tolist(), "erank_series": er.tolist()}
        print(f"  [calib] eRank_final={sat:.2f}, 95% at t={rec['saturation']['erank95_pos']}")
    else:
        k = len(gt)
        # control: fixed / log
        for mname, B in [("fixed", analysis.fixed_boundaries(T, k)),
                         ("log", analysis.log_boundaries(T, k))]:
            rec["methods"][mname] = analysis.alignment(B, gt, T, w)
            rec["methods"][mname]["B"] = B
        # signal-based (count-matched) + natural-count
        for sname in METHODS_SIGNAL:
            vals, kind = sigs[sname]
            B = analysis.signal_boundaries(grid, vals, kind, k, min_gap_tok=w, stride=stride)
            a = analysis.alignment(B, gt, T, w)
            a["B"] = B
            a["natural_count"] = analysis.threshold_count(vals, kind)
            rec["methods"][sname] = a
        # 요약 출력
        rows = sorted(rec["methods"].items(), key=lambda kv: -kv[1]["f1"])
        print(f"  {'method':14s} {'F1':>5s} {'z':>6s} {'prec':>5s} {'rec':>5s} {'nat#':>5s}")
        for mn, a in rows:
            print(f"  {mn:14s} {a['f1']:5.2f} {a['z']:6.2f} {a['prec']:5.2f} {a['rec']:5.2f} "
                  f"{a.get('natural_count', '-'):>5}")

    if plot:
        _plot(name, grid, sigs, gt, rec, out_dir, w)

    with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)
    return rec


def _plot(name, grid, sigs, gt, rec, out_dir, w):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = list(sigs.keys())
    fig, axes = plt.subplots(len(order), 1, figsize=(12, 1.5 * len(order)), sharex=True)
    for ax, sname in zip(axes, order):
        vals, kind = sigs[sname]
        ax.plot(grid, np.nan_to_num(vals), "-o", ms=2, color="darkorange")
        if gt:
            for b in gt:
                ax.axvline(b, color="green", ls="--", lw=1, alpha=.6)
        B = rec["methods"].get(sname, {}).get("B", [])
        for b in B:
            ax.axvline(b, color="crimson", lw=1, alpha=.6)
        f1 = rec["methods"].get(sname, {}).get("f1", float("nan"))
        ax.set_ylabel(sname, fontsize=7)
        ax.set_title(f"{sname}  (F1={f1:.2f}, {kind})  green=GT red=pred", fontsize=7, loc="left")
    axes[-1].set_xlabel("token position t")
    fig.suptitle(f"{name}  ({rec['meta']['name']})", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{name}.png"), dpi=110)
    plt.close(fig)
    print(f"  saved {name}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["D2", "D3", "D4", "D5", "D6", "D7"])
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--layer-stride", type=int, default=1)
    ap.add_argument("--niah-len", type=int, default=2048)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "260722_results"))
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    tok = common.load_tokenizer()
    print("tokenizer vocab:", tok.vocab_size)
    bundle = common.load_model()

    summary = {}
    for name in args.datasets:
        try:
            rec = run_dataset(name, bundle, tok, args.stride, args.niah_len,
                              args.layer_stride, args.plot, args.out)
            summary[name] = {m: {"f1": a["f1"], "z": a["z"]}
                             for m, a in rec.get("methods", {}).items()}
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  !! {name} 실패: {e}")
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\n저장:", args.out)


if __name__ == "__main__":
    main()
