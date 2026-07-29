# 통합 보고서 — State Rank는 pruning 신호로 오해의 소지: pure GDN에서 3-Stage 검증 (2026-07-28)

대조 논문 **arXiv:2602.02195 "State Rank Dynamics in Linear Attention LLMs"**(Sun et al.)를 pure
GDN(gdn2-1.3B, paper-matched 100B)에서 3단계로 검증·반박했다. 모든 실험은 **실 pretrained 모델 +
실텍스트(WikiText-103/GitHub code/arXiv)**, 3 seed, 로컬 SLURM(gpu01), 부풀리지 않은 실측 수치.

---

## 0. 배경과 중심 질문

논문은 Qwen3-Next(48층 하이브리드, post-trained)에서 head별 **rank stratification**(저/고 랭크 이분)을
관측하고, **high-rank head를 지워도 성능 손실 없이 KV 38.9% 절감**된다고 보고하며, 이를 "high-rank =
oversaturated(포화되어 잉여)"로 해석한다. 즉 인과 사슬 = **`high rank ⟹ oversaturated ⟹ prunable`**.

핵심은 **saturation의 두 정의**를 구분하는 것이다:
- **(A) 선형대수적**: full rank (state가 쓸 수 있는 방향을 다 씀).
- **(B) 기능적(oversaturation)**: 용량 초과로 간섭이 정보를 파괴 — 많이 썼는데 꺼낼 게 없음.

논문은 (A)를 곧 (B)로 단정한다. 우리는 pure GDN에서 **(A) ≠ (B)** 임을 3단계로 보인다.

---

## 1. Stage 1 — rank stratification 재현 (관측)
- **목적**: 논문의 head별 rank stratification + 시간적 order-preservation이 pure GDN에도 존재하나.
- **세팅**: gdn2-1.3B(100B), threshold-rank(Eq.6, ε=1e-4), 시간일관성 PRIMARY = 논문의 separated-pair
  (early anchor t≈d vs late 256/512/1024/2048). 3 seed, 실 3도메인.
- **결과(보수적)**: **G1a = YES** — bimodal stratification(BC 1.0) + separated-pair 순위 ρ **0.982±0.015**.
  **G1b = pass (R² 0.897±0.006)**: 감쇠 r̄가 threshold-rank를 R²~0.9로 예측 → **stratification = decay
  stratification** (high-rank = 저-decay = 고-retention head). G1c strong(0.963), cross-domain 0.971.
- **정직한 편차**: nuclear-norm cosine(separated) = 0.72 (논문 0.98 미달) — 순위는 lock-in되나 norm
  크기는 더 드리프트. 순수-rank 분류는 논문 JRNP Saturation Score(Eq.14, α 미공개)로부터의 명시적 이탈.
- **한 줄**: 현상은 pure GDN에도 존재하고, 그 정체는 **감쇠(retention) 구조**다.

## 2. Stage 2 — pruning 비대칭 (기능, 역전)
- **목적**: high-rank pruning이 low-rank보다 성능을 덜 해치나(논문 예측)? — KV-state pruning으로 검증.
- **세팅**: 진짜 KV-state pruning(head value→0 → S_h=0, KV 메모리 제거), 4조건(origin/high/low/random,
  동수), 124/288 head = **KV 43% 절감**. NIAH origin=0.18 floor → **PPL이 primary**. 3 seed.
- **결과(보수적, macro PPL)**: origin 12.6 / **high 854.3±101** / low 39.4±3.0 / random 34.4±1.7.
  PPL 상승 vs origin: **high +841.7 ≫ low +26.8 ≈ random +21.8**.
- **판정: G2 = NO — 논문과 정반대.** high-rank pruning이 파국적(~32×), low-rank ≈ random(거의 잉여).
  pure GDN에선 **high-rank = load-bearing**. (논문의 "high-rank prunable"은 Qwen3가 하이브리드라
  attention이 부하를 대신 지어 SSM high-rank가 잉여였을 가능성.)

## 3. Stage 3 — high-rank는 oversaturated인가 (메커니즘, 반박)
- **목적**: high-rank head의 full rank가 (A)선형대수 포화일 뿐인지, 실제 쓰이는 (genuine) 용량인지.
- **세팅**: head state S_h의 spectral content를 겨냥한 3 개입(실모델·실데이터, high/low/random, 3 seed):
  (1) prune-fraction sweep(count-matched), (2) SVD top-r 절단, (3) 스펙트럼-보존 noise(rank·에너지 유지,
  방향만 랜덤 → 내용만 파괴). segment-control drift ≈ 0 확인.
- **결과(보수적, 3-seed mean±std)**:
  - **① prune sweep**: 모든 압축률에서 high가 파국. 논문 **38.9% KV 지점: high 655.8 vs low 34.9 vs
    random 33.1**. (Stage 2의 단일점을 곡선으로 확장, 압축률 매칭 애매함 해소.)
  - **② SVD top-r 기울기**: high **5.86±0.48** vs low 0.033±0.007 vs random 1.14 (high/low ~176×) →
    high-rank full rank가 실제 쓰임(절단하면 급락), low-rank는 이미 저차원.
  - **③ 스펙트럼-noise delta**: high **16.6±0.8** vs low 0.12±0.03 vs random 5.47 (high/low ~141×) →
    rank·에너지 그대로 두고 **내용만 부수면 high-rank 파국** → 저장된 정보가 중요, 단순 rank 아님.
- **판정: G3 = YES.** high-rank head는 rank 용량을 **실제로 사용** → **full rank = genuine capacity,
  oversaturated 아님.** 논문의 `high-rank ⟹ oversaturated` 단계를 반박.

---

## 4. 통합 서사와 논문화 포인트

세 단계가 한 줄로 이어진다:
> pure GDN에서 rank stratification은 **감쇠 구조**이고(S1), 고랭크 head는 **load-bearing**이며(S2),
> 그 full rank는 **실제로 쓰이는 용량**이다(S3). 즉 **선형대수적 full rank ≠ 기능적 oversaturation.**

**contribution / 비교우위**:
1. 논문의 **`rank ⟹ oversaturation ⟹ prunable`** 추론은 pure linear model에서 **역전**하고(S2), 그
   **메커니즘 해석(oversaturation)은 틀렸다**(S3, full rank는 genuine capacity). eRank/full-rank를 pruning
   신호로 쓰는 것은 **기계적으로 오독**이며 **아키텍처 의존적**이다.
2. pure linear model에선 high-rank(=고-retention)가 오히려 **메모리를 지는 head** — 논문 가정의 정반대.
3. 우리 연구 프로그램의 논지 **eRank ≠ capacity**의 실모델·정량 증거.

---

## 5. 한계 (정직)
- **hybrid 미포함**: 매칭 hybrid 체크포인트(swa_gdn2-1.3B)가 공개 부재 → "hybrid에선 왜 prunable한가"의
  직접 증명(attention offload 가설)은 **future work**. 현재는 pure-linear 메커니즘 반박에 한정.
- **NIAH floored**(origin 0.18) → 주지표는 PPL. (base 모델이 RULER-multikey를 거의 못 함.)
- **순수-rank 분류** (≠논문 JRNP Saturation Score, α 미공개). nuclear-norm 편차(S1, 0.72<0.98).
- 개입은 **사후 state surgery**(재귀가 만든 state를 readout 직전 조작). 3 seed, 단일 체크포인트.
- 논문 수치(93.8/46.9/90.6/38.9%)는 Qwen3-Next(48층 post-trained) **target**이지 18층 gdn2의 pass
  기준이 아님.

## 6. Future work
- **hybrid vs pure 통제 비교**: swa_gdn2-1.3B(또는 Qwen3-Next)에서 같은 S2/S3 실행 → "prunability =
  attention redundancy" 가설 직접 검증. attention offload가 답이면 "같은 rank 신호가 아키텍처에 따라
  정반대 의미"라는 서사가 완성됨.
- Saturation-Score(Eq.14) arm(α sweep) 추가로 분류 기준 이탈 보강.

## 7. 재현
- 코드·결과 = `260725_stage1_rank_stratification/`, `260726_stage2_pruning_asymmetry/`,
  `260727_stage3_rank_mechanism/` (각 STAGE*_RESULTS.md + results/aggregate_*.json + seed 리포트).
- go/no-go 리뷰 = `working_note/260725·260726·260727_exp_review_*.md`.
- 모델 = gdn2-1.3B model-100b.pth(sha256[:16]=4b03319f), 로더 = `260725_.../loader_gdn2.py`(kernel-intercept
  a_t), 데이터 = `data_stage1.py + data_cache/`(실텍스트). 실행 = 로컬 SLURM gpu01, venv torch 2.13+cu130.
