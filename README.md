# SSM Rank Analysis

SSM 및 linear-attention 모델의 recurrent state를 분석하는 실험 저장소입니다. 주된 질문은 **state rank가 실제 용량·recall·pruning 가능성을 나타내는가**입니다.

## 핵심 결론

- **eRank는 보편적인 capacity 또는 saturation 지표가 아닙니다.** 부하가 증가할 때 eRank가 올라가도 recall은 떨어질 수 있고, 서로 다른 update rule 사이에서도 eRank와 recall의 순서는 일치하지 않습니다.
- **용량은 모델이 state에서 실제로 읽어내는 recall로 측정해야 합니다.** Delta 계열 update가 additive linear-attention보다 key 간섭을 더 잘 관리합니다.
- **pure GDN에서는 high-rank head가 잉여가 아니라 load-bearing recall 유닛입니다.** high-rank head는 낮은 decay/높은 retention과 연결되고, pruning·SVD 절단·내용 파괴 실험에서 가장 큰 성능 손상을 일으킵니다.
- 따라서 `high rank → oversaturated → prunable`이라는 해석은 아키텍처 의존적이며, pure GDN 결과를 hybrid 모델에 그대로 일반화할 수 없습니다.

## 구조

```text
SSM_Rank_Analysis/
├── README.md                         # 이 문서
├── REPORT.ko.md / REPORT.md          # 용량·recall 종합 보고서(한국어/영어)
├── MODEL_SETUP.md                    # 모델·환경·체크포인트 안내
├── theory/                           # 연구 목표와 random-matrix 이론 메모
├── 260720_stage0_capacity_diagnostics/                        # Mamba-2/GDN 용량·chunking·recall 실험
│   ├── capacity_results/
│   ├── chunking_results/
│   ├── state_capacity_results/
│   ├── stored_vs_used_results/
│   ├── decay_mqar_results/
│   └── state_saturation_results/
├── legacy/                           # 초기·보조 실험 보관 영역
│   ├── 260722_exp/
│   ├── figures/                       # 이전 대표 그림
│   └── mamba3_analysis/              # 이전 통합 분석 suite
├── 260725_stage1_rank_stratification/  # rank stratification 재현
├── 260726_stage2_pruning_asymmetry/    # high/low-rank pruning 비교
├── 260727_stage3_rank_mechanism/       # genuine capacity 메커니즘 검증
├── 260728_recall_role/                 # induction 기반 recall 역할 검증
├── 260728_rank_decay_dist/             # head별 rank–decay 분포
└── working_note/                     # 실험 리뷰, 통합 보고서, 논문 계획
```

## 실험 흐름

### 1. State capacity와 eRank

[`260720_stage0_capacity_diagnostics/README.md`](260720_stage0_capacity_diagnostics/README.md)에 노트북별 질문이 정리되어 있습니다.

- MQAR load를 늘리면 recall은 `1.00 → 0.43`으로 하락하지만 eRank는 `1.9 → 4.4`로 증가합니다.
- Dynamic chunk 길이는 합성 데이터에서 epiplexity와 강하게 연관(`Spearman ρ≈0.94`)되지만, 자연 텍스트에서는 관계가 약합니다.
- Mamba-2 SSD와 plain gated-delta의 상태 단위당 실현 recall 효율은 비슷하고, MoM wrapper는 비효율적입니다.
- 다섯 update rule 비교에서 eRank와 recall의 상관은 사실상 없으며, recall 차이는 update rule의 간섭 관리 방식과 더 밀접합니다.

자세한 수치와 한계는 [`REPORT.ko.md`](REPORT.ko.md)를 기준으로 합니다.

### 2. Pure GDN rank/pruning 실험

모델은 gdn2-1.3B 100B checkpoint이며, 실제 WikiText/GitHub/arXiv 텍스트와 3개 seed를 사용했습니다.

| 단계 | 질문 | 핵심 결과 |
|---|---|---|
| Stage 1 | rank stratification이 존재하는가? | 존재하며 decay/retention stratification으로 설명됨 (`R²≈0.90`, 순위 `ρ≈0.98`) |
| Stage 2 | high-rank head가 prunable한가? | 아니오. 43% KV pruning에서 PPL `12.6 → 854.3` |
| Stage 3 | high-rank full rank가 실제 용량인가? | 예. SVD 절단과 spectrum-preserving noise에 매우 민감 |
| Stage 4 | high-rank head의 역할은 무엇인가? | induction recall 유닛. pruning 시 gain `2.38 → −0.006` |

- [Stage 1 결과](260725_stage1_rank_stratification/STAGE1_RESULTS.md)
- [Stage 2 결과](260726_stage2_pruning_asymmetry/STAGE2_RESULTS.md)
- [Stage 3 결과](260727_stage3_rank_mechanism/STAGE3_RESULTS.md)
- [Stage 4 결과](260728_recall_role/STAGE4_RESULTS.md)

## 문서와 산출물 사용법

- 전체 해석은 [`REPORT.ko.md`](REPORT.ko.md)를 먼저 읽습니다.
- pure GDN의 논문 반박 서사는 [`working_note/260728_3stage_integrated_report.md`](working_note/260728_3stage_integrated_report.md)와 [`working_note/260728_recall_role_paper_plan_handoff.md`](working_note/260728_recall_role_paper_plan_handoff.md)에 있습니다.
- 각 Stage의 재현 코드는 해당 폴더의 `README.md`, `PREREGISTRATION.md`, `run_*.sbatch`를 확인합니다.
- Stage 2~4에는 seed별 JSON과 aggregate JSON이 저장되어 있습니다. Stage 1은 현재 보고서와 코드·데이터 캐시 중심으로 보관되어 있습니다.

## 해석상의 주의점

- eRank는 spectral concentration이며 algebraic rank와 다릅니다.
- NIAH 원본 성능이 낮은 단계에서는 PPL을 주지표로 사용했습니다.
- 대부분의 recall 결과는 MQAR 또는 induction에 대한 in-context 평가이며, 해당 task로 모델을 학습한 결과가 아닙니다.
- pure GDN과 Qwen3-Next 같은 hybrid 모델의 pruning 의미는 다를 수 있습니다. attention offload 가설은 아직 직접 비교하지 않았습니다.

## Legacy

초기 multi-state 경계 탐지, 기존 figures, Mamba-2 통합 분석은 [`legacy/`](legacy/)에 보관되어 있습니다. 결과와 코드는 보존하지만, 현재의 핵심 결론을 대표하는 최신 실험군은 아닙니다.
