# Stage 1 결과 — gdn2-1.3B (100B)에서 rank stratification 재현 (2026-07-26)

대조 논문 arXiv:2602.02195 "State Rank Dynamics in Linear Attention LLMs"의 Stage 1 관측(head별
rank stratification + 시간적 order-preservation)을 **pure GDN(gdn2-1.3B, paper-matched 100B)**에서
재현했다. 실제 3도메인(WikiText-103/GitHub code/arXiv) + App.D 반복공격 2종, **seed 3개(0,1,2)**.

## 세팅
- 모델 gdn2-1.3B model-100b.pth(sha256[:16]=4b03319f), Config.from_name + strict=False(missing=0),
  bf16 fused_recurrent, 18층×16head, threshold-rank ε=1e-4(Eq.6).
- a_t 캡처 = **kernel-intercept**(fused_recurrent_gdn2의 g 직접, a_t=exp(g)).
- 데이터 = 로컬 캐시 실텍스트(fallback 0건, `--require-real-data` 통과), seed 종속 샘플링.
- 시간일관성 PRIMARY = **논문의 separated-pair**(early anchor t≈d vs late 256/512/1024/2048),
  growth-adjacent은 secondary 진단.
- 실행 = 로컬 SLURM(gpu01, RTX PRO 6000 Blackwell), venv torch 2.13+cu130.

## 결과 (3 seed 집계, mean ± std)
| 게이트/지표 | 값 | 판정 |
|---|---|---|
| **G1a** stratification+time | **YES** (3/3 seed) | **재현** |
| bimodal stratification (BC) | 1.00 ± 0.00 | 이분 분포 존재 |
| separated-pair ρ>0.90 (PRIMARY, rank) | **0.982 ± 0.015** | 순위 order-preservation 실재 |
| growth-adjacent ρ>0.90 (secondary) | 1.00 | (자명, 진단용) |
| **G1b** r̄→threshold-rank R² | **0.897 ± 0.006** → **pass** | stratification=decay stratification |
| **G1c** entropy-eRank↔threshold-rank | 0.963 (strong) | 지표 일치 |
| cross-domain head 분류 일치 | 0.971 ± 0.003 | 도메인 불변 |
| nuclear-norm cosine (separated) | **0.72 ± 0.08** | 논문 0.98 **미달** |

## 게이트 판정
- **G1a = YES.** bimodal stratification(BC 1.0) + separated-pair 순위 ρ 0.98(>0.90, 96~100% 층).
  → 논문의 rank stratification·시간 order-preservation이 **pure GDN(hybrid·스케일·post-training 없이)
  에서도 재현**된다. [PIN-7] Stage 2(pruning 비대칭) 착수 게이트 해제.
- **G1b = pass (R²≈0.90).** head별 감쇠 r̄=exp(E[log a_t])가 threshold-rank를 R²~0.9로 예측 →
  stratification은 사실상 **decay(gate) stratification의 재기술**이다. 즉 "high-rank head" ≈
  **저감쇠(고보존) head**. 이는 우리 F7(decay 지배, decay:뭉침 ≈4.7:1)과 정확히 연결된다.
- **G1c = strong (0.96).** entropy eRank와 threshold-rank가 head 순위에서 일치 → 지표 아티팩트 없음.

## 객관적 사실
1. **논문의 rank stratification은 pure GDN-1.3B(100B)에서 재현된다** (G1a=YES, 3 seed, 실데이터).
   → 현상은 hybrid/스케일/post-training 특이적이 아니다.
2. **stratification의 주기전은 decay다** (G1b R²~0.9). 고랭크 head = 저감쇠 head. F7과 일관.
3. **단, nuclear-norm 크기 order-preservation은 부분적** — separated-pair norm-cosine이 **0.72**로
   논문의 0.98에 못 미친다. 즉 **어느 head가 고/저랭크인지(순위)는 성장→포화까지 lock-in**되지만,
   각 head의 **nuclear-norm 크기 자체는 시간에 따라 더 드리프트**한다. (norm-cos는 aux 지표, 게이트
   대상 아님. 논문은 Qwen3-Next에서 0.98 보고 — 모델/데이터 차이 가능. 정직히 편차로 기록.)

## 한계
- 임계(ρ>0.90 등)는 Qwen3-Next(48층 post-trained) 관측치를 재현 목표선으로 채택한 것(PREREG 명기).
- 자연 3도메인은 로컬 캐시 실텍스트(WikiText-103 test / codeparrot-clean-valid / ccdv arxiv-summ);
  논문의 RankViz(미공개) 대체(PIN-5).
- seed 3개, 결정론 forward. norm-cos 편차의 원인(모델 규모·post-training·데이터) 미규명.

## 재현
- `sbatch run_stage1.sbatch` (SLURM gpu01, GDN2_CKPT_PATH=model-100b.pth). 3 seed → `results/stage1_100b_seed{0,1,2}/`.
- 판정 코드: `stage1_repro.py`(verdict), 지표: `rank_metrics.py`(Eq.6/Thm3.1/Thm4.4). 데이터: `data_stage1.py`+`data_cache/`.
