> 🇬🇧 English version: [`REPORT.md`](REPORT.md)

# 고정 크기 recurrent state의 정보 용량 — 중간 보고서

고정 크기 recurrent LM(Mamba-2 SSD 대 Gated DeltaNet)에서 연상 recall을 무엇이 제한하는지, 어떤 정보이론 신호가 그것을 잘 추적하는지, 정보 밀도가 dynamic chunking 길이를 어떻게 정하는지를 다룬다. 모든 결과는 RTX PRO 6000(Blackwell)에서 SLURM으로 돌렸다. 코드는 `notebooks/`, 공용 헬퍼는 `notebooks/capacity_utils.py`, 이론 노트는 `theory/`에 있다.

> **범위.** 이 보고서는 진단 성격의 하위 연구(어떤 신호가 용량인가, 상태가 얼마나 담는가, 청킹을 언제 촉발할까)를 다룬다. 실제 end-to-end로 상태를 스냅샷하고 라우팅해서 재사용하는 시스템(H3)은 여기서 의도적으로 제외했고, linear-memory-routing 프로젝트에서 다룬다.

## 요약
1. **eRank는 용량이 아니다.** 고정 크기 상태는 max rank의 약 7~11%만 쓰며, eRank는 부하가 커질 때 recall과 오히려 반대로 움직인다(eRank는 오르는데 recall은 떨어진다). eRank는 상태 스펙트럼이나 입력 구성의 진단 지표이지 용량계가 아니다.
2. **용량은 모델 recall이다.** 상태 자신의 `C`-read이며, 연상 읽기는 `value ≈ C·h`로 이뤄진다.
3. **병목은 연상 부하(키-값 쌍 개수)이고, 길이와 부하는 얽혀 있다.** §3의 주의를 보라. "거리에 강건하다"는 우리의 horizon 결과는 정보량이 거의 없는 filler 패딩이 만든 착시가 크다. 실제 내용으로 채우면 문맥이 길어진다는 건 곧 키가 많아진다는 뜻이고, 그만큼 간섭도 늘어난다.
4. **정보 밀도가 청크 길이를 정한다.** 누적 **epiplexity**는 밀도와 청크 길이의 관계를 잘 예측했고(Spearman ρ≈0.94), eRank는 약하게만 예측했다(ρ≈0.73). 이 관계는 포화(saturation) 효과다.
5. **update rule별 압축비**: 상태 메모리 단위당으로 보면 **Mamba-2 SSD ≈ plain Gated-DeltaNet**이다(약 10~11 recalled-bits / Mfloat). **MoM mixture wrapper는 훨씬 비효율적**이다(2.1). "GDN이 recall을 더 잘한다"는 통념은 여기서는 gated-delta라는 update rule 덕분이라고 보기 어렵다.

모델: `state-spaces/mamba2-370m`(SSD), `linear-moe-hub/Gated-Deltanet-340M`(plain gated delta, fused-MLP/`attn.D` weight adapter로 로드), `linear-moe-hub/MoM-Gated-Deltanet-340M`(MoM). 셋 다 사전학습된 범용 LM이며 합성 MQAR 과제로는 **학습하지 않았다**(따라서 recall은 in-context다).

---

## 1. 신호 카탈로그와 상태 활용도 (`notebooks/information_capacity_signals.ipynb`)
여섯 신호 — **S1** effective rank(eRank), **S2** predictive entropy / bits-per-token, **S3** in-context epiplexity, **S4** ground-truth bits, **S5** Rényi-2 / participation-ratio rank, **S6** TwoNN intrinsic dimension — 을 세 데이터셋(MQAR, WikiText-2, A5 state-tracking)과 두 모델에 걸쳐 쟀다.

**상태 활용도는 낮고 데이터에 거의 무관하다.** MQAR에서 peak eRank는 6.99 / 64 = **10.9%**(mamba2), 27.52 / 256 = **10.7%**(GDN-MoM)다. 두 모델 모두 상태 rank의 약 90%를 안 쓴다.

위치별 궤적(Δ = final − initial)과 raw 대 normalized 신호 곡선:

![Δ signal matrix](notebooks/capacity_results/full_matrix_delta.png)
![signal trajectories (normalized)](notebooks/capacity_results/signal_trajectories.png)
![signal trajectories (raw)](notebooks/capacity_results/signal_trajectories_raw.png)

Worked example — S1 eRank 대 MQAR load, 두 모델(상태 행렬 크기가 다른 점에 유의):

![eRank vs MQAR load](notebooks/capacity_results/worked_example_S1_D1_both.png)

---

## 2. eRank ≠ capacity, 용량 = recall (`notebooks/state_capacity_decodable.ipynb`)
모델 recall(상태 자신의 `C`-read)이 용량 신호다. padded MQAR로 **load**(쌍 개수 `N`)와 **horizon**(문맥 길이 `L`)을 분리하고 eRank를 겹쳐 봤다.

- **LOAD** (`L=512` 고정, `N` 변화): recall이 N=2→200에서 **1.00 → 0.43**으로 떨어지는 동안 eRank는 1.9 → 4.4로 **오른다**. → **반상관**: eRank가 큰 건 여유가 아니라 과부하의 징후다.
- **HORIZON** (`N=8` 고정, `L=32→4096`으로 패딩): recall은 0.92 → 0.82, eRank는 3.4 → 1.8로 **떨어진다**. → 무관(decoupled)하다.

![load vs horizon](notebooks/state_capacity_results/load_vs_horizon.png)

**주의(중요, 앞선 과장을 정정).** horizon 축은 정보량이 낮은 filler 토큰 반복으로 패딩했는데, selective SSM은 이걸 거의 무시하기 때문에 쌍들이 4k까지 살아남는다. 실제 긴 문맥은 다양한 내용, 즉 더 많은 키와 간섭이다. 그래서 **길이와 부하는 얽혀 있다.** 정직하게 해석하면, 유한한 상태에는 연상 용량 한계가 있고 긴 문맥은 내용을 더 실어오기 때문에 그 한계를 압박한다는 것이며, 고정 상태 모델이 긴 문맥 recall에서 용량 제한을 받는다는 표준 견해와 일치한다. filler가 섞이지 않은 깨끗한 결과는 LOAD 축이다.

---

## 3. 정보 밀도 기반 dynamic chunking (`notebooks/dynamic_chunking_by_density.ipynb`)
상태의 용량 신호가 포화할 때 청크를 자른다면, 청크 길이가 입력 정보 밀도에 따라 달라질까? 밀도는 반복 knob으로 통제하고 bits/token으로 측정했다.

- **합성(레벨당 10 seeds):** 청크 길이가 **밀도에 따라 커진다** — epiplexity 기준 Spearman ρ = **0.94**(95% CI [0.89, 0.96]), eRank 기준 ρ = **0.73**([0.53, 0.86]). 포화 관점과 맞는다(밀도가 낮은 반복적 입력은 상태를 빨리 포화시켜 청크가 짧아진다).
- **자연 passage 교차검증(WikiText passage 48개):** 자연 텍스트의 좁은 밀도 대역 안에서는 관계가 약하거나 퇴화한다 — eRank ρ ≈ **−0.01**([−0.30, 0.28]), epiplexity는 퇴화(청크 1개). → 합성에서의 강한 효과는 반복이 만든 넓은 밀도 범위 때문인 면이 있다.

![chunk boundaries on eRank(t)](notebooks/chunking_results/worked_example_boundaries.png)
![chunk length vs density (synthetic, 10 seeds)](notebooks/chunking_results/chunk_by_density.png)
![chunk length vs density (natural passages)](notebooks/chunking_results/natural_passage_chunks.png)

정리: **epiplexity가 더 나은 contents-based 청크 길이 신호**지만, 이는 상태가 아니라 loss/데이터 밀도 측정이며, 밀도→길이 효과는 더 넓은 밀도에서 검증할 필요가 있다.

---

## 4. update rule 압축비 (`notebooks/stored_vs_used_gap.ipynb`)
상태별 **용량** `C = recall ≥ 0.9를 유지하는 최대 키 수`를 상태 크기로 정규화(같은 메모리)하면, update rule이 메모리를 얼마나 잘 쓰는지 순위 매기는 **압축비**가 된다.

| update rule | state (Mfloat) | C_used@0.9 (raw keys) | **recalled bits / Mfloat** |
|---|---|---|---|
| Mamba-2 (SSD) | 12.58 | 23.3 | **11.13** |
| plain Gated-DeltaNet (gated delta) | 6.29 | 10.7 | **10.17** |
| MoM Gated-DeltaNet (mixture) | 31.46 | 11.0 | 2.11 |

![stored vs used, both models](notebooks/stored_vs_used_results/stored_vs_used.png)

- **SSD ≈ plain gated-delta**가 메모리 단위당 비슷하다(약 10~11 bits/Mfloat). Mamba-2의 raw 용량이 더 높은 것(23 대 11)은 대부분 상태가 약 2배 크기 때문이다(12.6 대 6.3 Mfloat). 용량은 상태 크기에 대체로 선형으로 붙으므로, 이 세팅에서 update rule 자체는 byte당 효율을 크게 바꾸지 않는다.
- **MoM은 메모리 비효율적이다**: raw recall 이득 없이 상태만 5배 쓴다(유휴 메모리가 크기를 부풀린다). MoM은 update rule이 아니라 mixture wrapper이므로 update-rule 결론에서는 제외한다.

**주의.** raw recurrent state에 있지만 못 읽힌 정보를 재려던 `stored` 쪽(bilinear probe)은 여기서도, 앞선 변형에서도 **chance** 수준으로 읽혔다. 외부 probe가 random / 입력 임베딩 키 특징을 쓰면 모델 내부의 `B`/`C` 주소지정과 정렬되지 않기 때문이다. 그래서 stored≥used 격차는 **재지 못했고**, 위 용량 수치는 이론상 최대가 아니라 실제로 꺼내진(realized) 용량이다. 격차를 재려면 `B/C`를 반영한 probe나 recurrent-state SAE가 필요하다.

---

## 5. 주의와 한계 (종합)
- **길이와 부하는 얽혀 있다.** horizon의 "거리 강건성"은 정보 없는 filler가 부풀린 것이다(§2).
- **`stored` probe는 무효다**(chance) — realized(recall) 용량만 있고 stored≥used 격차는 아직 없다(§4).
- **사전학습이지 MQAR 학습이 아니다** — recall은 in-context이며, 학습된 모델은 다를 수 있다.
- **eRank(spectral) ≠ algebraic rank** — random이든 학습된 가중치든 거의 확실히 full algebraic rank이며, 낮은 eRank는 동역학에 의한 스펙트럼 집중이다(`theory/random_matrix_full_rank.md` 참조).
- `N`/밀도 격자가 성기고, MoM은 native(부풀려진) 상태 크기로 비교했다.

## 6. 결론과 다음
확립된 것(진단): 용량 = recall, eRank는 용량이 아님(부하에서 반상관), epiplexity는 밀도/청크 길이 신호, plain gated-delta의 메모리당 압축은 SSD와 비슷함. **아직 안 한 것:** (a) 실제 내용으로 긴 문맥/horizon 축에서 update rule을 비교(망각 메커니즘을 실제로 가르는 축), (b) 유효한 stored 용량 probe, (c) 청크당 용량을 확장된 유효 용량으로 바꾸는 end-to-end 스냅샷→라우팅→재사용 시스템(H3) — 이는 linear-memory-routing 프로젝트에서 진행한다.

*로드맵과 가설: `theory/research_goal.md`. 이론 노트: `theory/random_matrix_full_rank.md`.*
