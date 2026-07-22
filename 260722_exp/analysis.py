"""신호 계산 + 청킹 방법 + GT 정렬 평가.

신호 후보 (경계/포화 판단):
  DLA(기존 dynamic) : I_t = ||RMS(S_t)-RMS(S_{t-1})||_F / (||RMS(S_{t-1})||_F + eps)   [상대 변화율]
  surprisal (논문)  : per-token NLL(bits) 블록 평균
  epiplexity        : 점근선 초과 NLL 블록 합
  erank             : state effective rank (spectral entropy exp)
  state_entropy     : state 원소분포 Shannon entropy (spread)
  nuclear/participation/numrank : 저장된 k-v 개수 추정군
청킹 방법:
  fixed(L), log-schedule  : content-agnostic (control)
  dynamic-threshold(signal): DLA 및 state-signal (exp)
평가:
  count-matched  : 모든 방법에 |GT|개 경계 → 위치 F1 (신호 우열의 공정 비교)
  natural-count  : threshold(z>thr)에서 몇 개 자르나 → 과/과소분할 경향
"""
import math
import numpy as np
import torch

EPS = 1e-8


# ─────────────────────────── logits 기반 신호 ───────────────────────────
@torch.no_grad()
def token_nll_bits(bundle, ids):
    """per-token surprisal (bits), 길이 T-1."""
    logp = torch.log_softmax(bundle.logits(ids), -1)
    tgt = ids[:, 1:].to(logp.device)
    nll = -logp[:, :-1, :].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[0]
    return nll.float().cpu().numpy() / math.log(2)


# ─────────────────────────── state 특성 ───────────────────────────
def _mat_feats(M):
    """M: (dk,dv) → (erank, nuclear, participation, numrank, state_entropy)."""
    s = torch.linalg.svdvals(M.float())
    s = s[s > 0]
    if s.numel() == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    ssum = float(s.sum())
    p = s / ssum
    erank = float(torch.exp(-(p * torch.log(p)).sum()))
    nuclear = ssum
    participation = float((ssum ** 2) / (float((s ** 2).sum()) + EPS))
    numrank = float((s > 1e-3 * float(s[0])).sum())
    v = M.abs().reshape(-1)
    q = v / (v.sum() + EPS)
    state_entropy = float(-(q * torch.log(q + EPS)).sum())
    return erank, nuclear, participation, numrank, state_entropy


def _rms(M):
    return M / torch.sqrt((M.float() ** 2).mean() + EPS)


def state_trajectory(bundle, ids, stride=8, min_t=8, layer_stride=1):
    """prefix[:t] grid를 훑으며 state 특성 + DLA(I_t) 계산.
    반환: grid(list t), feats(dict name->np array), dla(np array; [0]=nan)."""
    T = ids.shape[1]
    grid = list(range(min_t, T + 1, stride))
    if grid[-1] != T:
        grid.append(T)
    names = ["erank", "nuclear", "participation", "numrank", "state_entropy", "fro"]
    feats = {n: [] for n in names}
    dla, prev_rms = [], None
    for t in grid:
        st = bundle.states(ids[:, :t])
        layers = sorted(st.keys())[::layer_stride]
        per = {n: [] for n in names}
        rms_parts = []
        for li in layers:
            S = st[li]                                   # (heads, dk, dv)
            for h in range(S.shape[0]):
                M = S[h]
                er, nu, pr, nr, se = _mat_feats(M)
                per["erank"].append(er); per["nuclear"].append(nu)
                per["participation"].append(pr); per["numrank"].append(nr)
                per["state_entropy"].append(se); per["fro"].append(float(M.float().norm()))
                rms_parts.append(_rms(M).reshape(-1))
        for n in names:
            feats[n].append(float(np.mean(per[n])) if per[n] else float("nan"))
        rflat = torch.cat(rms_parts) if rms_parts else None
        if prev_rms is None or rflat is None:
            dla.append(float("nan"))
        else:
            num = float((rflat - prev_rms).norm())
            den = float(prev_rms.norm()) + EPS
            dla.append(num / den)                        # I_t = ||RMS(S_t)-RMS(S_{t-1})|| / ||RMS(S_{t-1})||
        prev_rms = rflat
    feats = {n: np.array(v, float) for n, v in feats.items()}
    return grid, feats, np.array(dla, float)


# ─────────────────────────── 신호 묶기 ───────────────────────────
def _block_agg(grid, per_token, reduce="mean"):
    """per_token(길이 T-1)을 grid 블록으로 집계. 블록 i = (grid[i-1], grid[i]] 토큰들."""
    out = [float("nan")]
    for i in range(1, len(grid)):
        a, b = grid[i - 1], grid[i]
        seg = per_token[max(0, a - 1):max(0, b - 1)]
        if len(seg) == 0:
            out.append(0.0)
        else:
            out.append(float(np.mean(seg)) if reduce == "mean" else float(np.sum(seg)))
    return np.array(out, float)


def build_signals(grid, feats, dla, nll_bits, tail_frac=0.2):
    """각 후보 신호를 {name: (values, kind)} 로. kind: 'change'(값 자체 peak) / 'level'(증분 peak)."""
    k = max(1, int(len(nll_bits) * tail_frac))
    asymptote = float(nll_bits[-k:].mean())
    excess = np.clip(nll_bits - asymptote, 0, None)
    surprisal = _block_agg(grid, nll_bits, "mean")
    epiplexity = _block_agg(grid, excess, "sum")
    return {
        "DLA_frobenius": (dla, "change"),           # 기존 dynamic (상대 RMS-ΔS)
        "surprisal":     (surprisal, "change"),     # 논문 baseline
        "epiplexity":    (epiplexity, "change"),    # 제안
        "erank":         (feats["erank"], "level"), # 제안 (rank)
        "state_entropy": (feats["state_entropy"], "level"),
        "nuclear":       (feats["nuclear"], "level"),        # k-v 개수 추정
        "participation": (feats["participation"], "level"),  # k-v 개수 추정
        "numrank":       (feats["numrank"], "level"),        # k-v 개수 추정
    }


# ─────────────────────────── 청킹 방법 ───────────────────────────
def fixed_boundaries(T, k):
    if k <= 0:
        return []
    step = T / (k + 1)
    return sorted({int(round(step * (i + 1))) for i in range(k)})


def log_boundaries(T, k, base=2.0):
    """logarithmic schedule: 청크 길이가 기하급수적으로 증가 → 경계는 뒤로 갈수록 성김."""
    if k <= 0:
        return []
    raw = np.cumsum([base ** i for i in range(k)])
    b = sorted({int(round(x / raw[-1] * T)) for x in raw[:-1]})
    return [x for x in b if 0 < x < T][:k]


def _topk_peaks(pos, score, k, min_gap_idx):
    order = np.argsort(-score)
    picked = []
    for i in order:
        if np.isnan(score[i]):
            continue
        if all(abs(i - j) >= min_gap_idx for j in picked):
            picked.append(i)
        if len(picked) >= k:
            break
    return sorted(int(pos[i]) for i in picked)


def signal_boundaries(grid, values, kind, k, min_gap_tok=16, stride=8):
    """count-matched: 신호에서 |GT|=k 개 경계 위치. level 신호는 증분(|diff|)의 peak."""
    v = np.array(values, float)
    if kind == "level":
        d = np.abs(np.diff(v, prepend=v[0]))
        score = np.nan_to_num(d, nan=-1)
    else:
        score = np.nan_to_num(v, nan=-1)
    min_gap_idx = max(1, int(round(min_gap_tok / stride)))
    return _topk_peaks(np.array(grid), score, k, min_gap_idx)


def threshold_count(values, kind, z_thr=1.5):
    """natural-count: z>z_thr 넘는 지점 수 (과/과소분할 경향)."""
    v = np.array(values, float)
    x = np.abs(np.diff(v, prepend=v[0])) if kind == "level" else np.nan_to_num(v, nan=0.0)
    x = np.nan_to_num(x, nan=0.0)
    z = (x - x.mean()) / (x.std() + EPS)
    return int((z > z_thr).sum())


# ─────────────────────────── 평가 (GT 정렬) ───────────────────────────
def f1_at(Bp, Bg, w):
    if not Bp or not Bg:
        return 0.0, 0.0, 0.0
    tp_p = sum(any(abs(a - b) <= w for b in Bg) for a in Bp)
    tp_r = sum(any(abs(a - b) <= w for a in Bp) for b in Bg)
    prec = tp_p / len(Bp); rec = tp_r / len(Bg)
    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    return f1, prec, rec


def alignment(Bp, Bg, T, w, n_shuffle=1000, seed=0):
    f1, prec, rec = f1_at(Bp, Bg, w)
    rng = np.random.default_rng(seed)
    null = np.array([f1_at(sorted(rng.integers(0, T, len(Bp)).tolist()), Bg, w)[0]
                     for _ in range(n_shuffle)]) if Bp else np.zeros(n_shuffle)
    z = (f1 - null.mean()) / (null.std() + EPS)
    return {"f1": f1, "prec": prec, "rec": rec, "n_pred": len(Bp),
            "null_f1": float(null.mean()), "z": float(z), "p_ge": float((null >= f1).mean())}
