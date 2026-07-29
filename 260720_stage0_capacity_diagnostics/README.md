# Stage 0 — State Capacity Diagnostics (2026-07-02 ~ 2026-07-20)

Mamba-2와 Gated DeltaNet의 recurrent state를 대상으로 수행한 초기 진단 실험 묶음입니다. 이후의 pure GDN Stage 1~4 실험에 앞서, **eRank·state saturation·recall·dynamic chunking**이 어떤 관계를 갖는지 확인했습니다.

## 이 단계의 위치

```text
Stage 0  state capacity / signal diagnostics  ← 이 폴더
Stage 1  rank stratification 재현
Stage 2  high/low-rank pruning 비대칭
Stage 3  genuine capacity 메커니즘
Stage 4  recall 역할(induction)
```

실험은 Git 기록상 2026-07-02부터 2026-07-20 사이에 수행되었습니다. 폴더명 `260720`은 Stage 1 직전의 정리 기준일입니다.

## 실험 구성

| 파일/폴더 | 질문 | 현재 해석 |
|---|---|---|
| `mamba2_effective_rank.ipynb` | state eRank가 context 길이와 함께 어떻게 증가하고 head별 포화점 `T*`가 있는가? | 초기 rank/saturation 가설 |
| `exp1_state_analysis.ipynb` | 모델 크기와 head 유형(Type A/B/C)에 따라 rank saturation이 달라지는가? | head 이질성·slow head 분석 |
| `exp2_state_similarity.ipynb` | 같은 주제의 문서 state가 서로 가까운가? | state-as-retrieval-key 가능성 탐색 |
| `exp3_state_injection.ipynb` | 저장된 state를 주입하면 문서를 읽은 것처럼 동작하는가? | state injection/RAG 가능성 탐색 |
| `information_capacity_signals.ipynb` | eRank, entropy, epiplexity, ground-truth bits, participation rank, TwoNN 중 어떤 신호가 state 변화를 설명하는가? | eRank는 capacity 신호가 아님 |
| `state_capacity_decodable.ipynb` | recall 병목이 load인지 horizon인지, eRank가 recall을 따르는가? | load가 증가하면 recall↓, eRank↑ |
| `dynamic_chunking_by_density.ipynb` | 정보 밀도가 chunk 길이를 결정하는가? | 합성 데이터에서 epiplexity가 더 강함 |
| `stored_vs_used_gap.ipynb` | state에 저장된 정보와 실제 읽히는 정보의 차이는 무엇인가? | stored probe는 chance; realized recall만 유효 |
| `pretrained_decay_mqar.py` | update rule별 eRank와 recall은 함께 움직이는가? | eRank와 recall은 거의 독립; delta 계열이 우수 |
| `260720_boundary_vs_state_saturation.ipynb` | 정보 점수 경계가 실제 state saturation을 포착하는가? | 초기 boundary/saturation 검증 |

## 결과 디렉터리

- `capacity_results/`: S1~S6 신호 궤적과 모델 비교
- `state_capacity_results/`: load 대 horizon, recall 대 eRank
- `chunking_results/`: 정보 밀도별 dynamic chunk 결과
- `stored_vs_used_results/`: update rule별 실현 용량·압축비
- `decay_mqar_results/`: 5개 pretrained update rule의 MQAR recall/eRank
- `state_saturation_results/`: saturation smoke/diagnostic 결과

## 종합 결론

1. **eRank는 capacity meter가 아닙니다.** MQAR load가 증가할 때 recall은 `1.00 → 0.43`으로 감소하지만 eRank는 `1.9 → 4.4`로 증가합니다.
2. **실제 capacity는 모델의 state readout recall입니다.** update rule 간 비교에서도 eRank보다 delta/error-correction 방식이 recall을 더 잘 설명합니다.
3. **epiplexity는 합성 입력의 밀도와 chunk 길이를 eRank보다 잘 설명합니다**(`ρ≈0.94` 대 `0.73`). 단, 자연 텍스트 범위에서는 효과가 약합니다.
4. **stored capacity와 used capacity의 차이는 아직 측정되지 않았습니다.** 외부 probe가 내부 B/C addressing과 정렬되지 않아 chance 수준이었습니다.

상세 종합 보고서는 [`../REPORT.ko.md`](../REPORT.ko.md), 현재 핵심 실험은 [`../README.md`](../README.md)와 Stage 1~4 결과 문서를 참조합니다.
