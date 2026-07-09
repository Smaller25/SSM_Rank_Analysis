# Random matrices are full rank with probability 1

이론적 검토용 참고 노트. 무작위(Gaussian/Uniform 등 **연속분포**) 난수로 채운 가중치 행렬 `W`가
**확률 1로 full rank(최대 계수)** 가 된다는 고전적 사실과 그 근거·인용·프로젝트 함의를 정리.

## 1. 정리 (statement)

> 성분이 **연속 확률분포**에서 독립 추출된 행렬은 확률 1로 full rank이다.
> *"A matrix with entries drawn from a continuous distribution is full rank with probability 1."*

특정 논문이 최초 발표한 것이 아니라 **Random Matrix Theory(RMT)**에서 기본 상식으로 정립된 성질.

## 2. 대표 문헌 (citations)

- **Edelman, A. (1988). "Eigenvalues and condition numbers of random matrices."** *SIAM J. Matrix Anal. Appl.*
  가우시안 무작위 행렬의 고윳값 분포·조건수(condition number) 분석의 기념비적 연구.
  무작위 행렬이 singular(rank-deficient)가 될 확률이 정확히 **0**임을 보임.
- **Rudelson, M. & Vershynin, R. (2010). "Non-asymptotic theory of random matrices: extremes of singular values."** *ICM.*
  현대 ML·고차원 통계에서 가장 많이 인용되는 RMT 논문 중 하나. **이산분포(예: 무작위 ±1)**를 따르더라도
  차원이 커질 때 full rank일 확률적 하한을 엄밀히 증명 (연속분포 가정 없이도 성립).
- 교재 인용 시: *"It is a well-known fact in random matrix theory that a matrix with entries drawn
  from a continuous distribution is full rank with probability 1."*

## 3. 왜 확률이 1인가 (proof intuition)

- **rank-deficient ⇔ det(W)=0.** 행 벡터들이 선형종속 ⇒ 행렬식이 정확히 0이어야 함.
- **연속분포의 성질:** 연속확률변수가 특정 값(예: 0)에 정확히 명중할 확률 = 0.
  `{det(W)=0}` 은 계수공간에서 측도 0(measure-zero)인 대수적 다양체(algebraic variety).
- **초평면(hyperplane) 논증:** N차원에서 N−1개의 무작위 벡터가 span하는 공간은 '두께 0'의 초평면.
  N번째 무작위 벡터가 하필 그 초평면 위에 정확히 떨어질 확률 = 0.
- ∴ `P(W not full rank) = 0`, `P(full rank) = 1`.

## 4. 딥러닝에서의 의의 (ML relevance)

- **Random Kitchen Sinks (Rahimi & Recht, 2007):** 무작위 고정 가중치의 대형 레이어가 입력의 기하 구조를
  훼손하지 않고 고차원으로 embedding → 선형분리를 가능케 함.
- **Deep linear networks:** 무작위 초기화 `W`는 full rank라 초기 상태에서 정보 붕괴(information collapse)가
  일어나지 않는 '이상적 기하 뼈대' 역할.

## 5. 이 프로젝트와의 함의 — **algebraic rank ≠ effective rank** ⚠️

우리 작업([[../notebooks/capacity_utils.py]]의 `effective_rank`, S1)은 가중치가 아니라 **recurrent state**의
**effective rank(eRank = exp(정규화 특이값 스펙트럼의 Shannon 엔트로피))**를 잰다. 핵심 구분:

- 무작위 행렬은 a.s. **full *algebraic* rank** (특이값이 모두 ≠ 0) — 하지만 특이값이 **균일하지 않으면**
  **eRank는 full rank보다 훨씬 작다** (Marchenko–Pastur 스펙트럼처럼 퍼져 있어도 eRank < n).
- 따라서 우리가 관측한 **낮은 utilization (eRank/max_rank ≈ 7–11%; mamba2·gdn 공통)** 은
  가중치가 rank-deficient라서가 아니다 (무작위/학습가중치 모두 a.s. full algebraic rank).
  이는 **선택적 SSM 동역학이 상태의 스펙트럼을 소수 방향에 집중**시킨 결과 = 학습·구조적 성질.
- **함의:** "정보 용량(capacity)"을 논할 때 full-rank(=0 확률로 미달)와 eRank(=실제 사용 차원)를 혼동하면 안 됨.
  capacity 신호로서 의미 있는 건 **eRank/S5(참여율)/intrinsic-dim** 이지 algebraic rank가 아니다.
  무작위 baseline의 eRank(=상태가 아무 구조도 안 쓸 때의 값)를 대조군으로 두면 utilization 해석이 더 엄밀해진다.

## 6. 한 줄 요약

연속분포 난수로 채운 `W`는 `det=0`이 될 확률이 측도 0이라 **Edelman(1988) 등 RMT에 의해 확률 1로
full (algebraic) rank**. 단, **effective rank(eRank)는 이와 별개**이며, 우리의 낮은 상태 utilization은
weight rank가 아니라 SSM 동역학의 스펙트럼 집중에서 온다.
