# `notebooks/` 로드맵 — 파일별 "무엇을 보려고 했는가?"

Mamba/SSM의 **recurrent state**를 분석하는 노트북 모음. 아래는 파일별 한 줄 목표.
큰 흐름: **초기**엔 "state의 rank/포화(T\*)와 그 state를 저장·재사용(RAG처럼)"에 관심 → **이번 세션**엔
"그 포화/용량을 재는 *신호*가 뭔지, 특히 eRank가 용량 신호로 쓸 만한지" 검증 (결론: eRank는 용량 신호가
아님 — [`../REPORT.md`](../REPORT.md), 로드맵 [`../theory/research_goal.md`](../theory/research_goal.md)).

---

## 1. 초기 실험 (직접 수행) — state의 rank·포화와 재사용 가능성
원조 실험 `mamba2_effective_rank`를 관심사별로 쪼갠 것이 `exp1~exp3`.

| 파일 | 무엇을 보려고 했는가? |
|---|---|
| **`mamba2_effective_rank.ipynb`** | (원조) SSM hidden state `h_t`의 **effective rank가 토큰 수 T에 따라 어떻게 커지고**, head별 **포화 시점 `T*`**는 어디이며, 그걸로 chunking 경계 `T*_chunk`를 정할 수 있나? |
| **`exp1_state_analysis.ipynb`** | (T\*/rank를 스케일·head로 확장) 130m→2.7b 스케일별 **head 이질성**(`A_disc` vs eRank, Type A/B/C 분류), slow(Type-A) head의 rank saturation, **모델이 커지면 long-term memory head 비율이 느나?** |
| **`exp2_state_similarity.ipynb`** | **같은 토픽 문서는 비슷한 Type-A head state를 갖는가?** (intra vs inter-topic cosine) → state 자체를 **retrieval key**로 쓸 수 있나 (임베딩 모델 없이 검색+생성 통합)? |
| **`exp3_state_injection.ipynb`** | **문서로 만든 state를 저장했다 주입하면 그 문서를 읽은 것처럼 동작하나?** (A Oracle vs B Inject vs C No-context vs D Wrong-state) → state-injection RAG 가능성. |
| `mamba3_analysis/` | 위 exp1~3 + state linearity + document-position effect를 Mamba3 대비로 통합한 suite. |

## 2. 이번 세션 추가 — "포화/용량을 재는 신호가 무엇인가?" (capacity signal 연구)

| 파일 | 무엇을 보려고 했는가? |
|---|---|
| `capacity_utils.py` | (인프라, 실험 아님) 3스택(mamba_ssm / lit_gpt / fla) 모델 로더 + recurrent state 추출 + 신호 **S1 eRank·S2 predictive entropy·S3 epiplexity·S4 gt-bits·S5 Rényi rank·S6 intrinsic dim** 공용 모듈. |
| **`information_capacity_signals.ipynb`** | **어떤 정보이론 신호(S1~S6)가 "입력에 따른 정보 용량"을 가장 잘 보여주나?** (신호 × 3데이터[MQAR/WikiText/A5] × 2모델[mamba2, GDN-MoM]; state 활용도·신호 궤적) |
| **`dynamic_chunking_by_density.ipynb`** | **정보 밀도에 따라 (capacity 신호 포화 지점 기준) dynamic chunk 길이가 달라지나?** (epiplexity가 밀도→길이를 예측; 자연 passage 교차검증) |
| **`state_capacity_decodable.ipynb`** | **용량 = recall인가? 병목이 부하(키 개수 N)냐 문맥 길이(L)냐? eRank가 recall을 따라가나?** (load축 vs horizon축 분리; 결론: eRank는 부하에서 recall과 반상관) |
| **`stored_vs_used_gap.ipynb`** | **상태에 담겼지만 모델이 안 읽는 정보(stored, 선형 프로브) vs 실제 recall(used)의 격차** + update rule별 **압축비**(state당 키 수). |
| **`pretrained_decay_mqar.py`** | **decay/no-decay update rule 5종(SSD·gated-delta·delta·GLA·RetNet)에서 eRank vs recall** — eRank가 recall(용량)을 예측하나? (결론: **eRank ⊥ recall**; recall은 delta/오차보정 update가 가름) |
| `mqar_fromscratch.py` | (보류) from-scratch tiny 모델로 update rule 비교 시도 — **학습 confound**(GatedDeltaNet만 학습됨)로 pretrained 결론엔 미사용. |
| `*_results/` | 각 실험 산출물 (JSON + PNG). `capacity_results/`, `chunking_results/`, `state_capacity_results/`, `stored_vs_used_results/`, `decay_mqar_results/`, `state_saturation_results/`. |

## 3. 핵심 결론 (이번 세션)
- **eRank는 용량/recall 신호가 아니다** — 모델 내부(부하↑ 시 recall과 반상관)에서도, update rule 간(RetNet은
  eRank 최고인데 recall 최악)에서도 decoupled. → state-saturation/chunking **트리거로 eRank 쓰면 안 됨.**
- **recall(연상 용량)을 가르는 건 update rule의 오차보정(delta) 갱신·내용 기반 선택**이지 상태 spread(eRank)가 아님.
- 자세히: [`../REPORT.md`](../REPORT.md) (KO: [`../REPORT.ko.md`](../REPORT.ko.md)), 모델 로딩 꿀팁 [`../MODEL_SETUP.md`](../MODEL_SETUP.md).

## 4. 다음 (본론은 linear-memory-routing 프로젝트)
snapshot→route→reuse로 청크당 용량을 확장하는 end-to-end 시스템(H3), 유효 stored-용량 probe, 실제-내용 긴
문맥 horizon 비교, gdn2(lit_gpt) 점 채우기 — 계획은 [`../theory/research_goal.md`](../theory/research_goal.md).
