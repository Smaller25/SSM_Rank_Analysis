# 인수인계 — "State Rank = 기능적 타입 서명" 논문 계획 (pure GDN, 2026-07-28)

> codex 인수인계용 요약. 지금까지 한 작업(Stage 1~4), 방금 발견한 방법론 문제, 그리고 이걸
> Transformer급 interpretability 논문으로 끌어올리기 위한 다음 실험 계획을 담는다.
> 작성자 컨텍스트: sohyung. repo = `/home/sohyung/SSM_Rank_Analysis` (github Smaller25/SSM_Rank_Analysis, main).

---

## 0. 프로젝트 중심 주장

연구 프로그램의 논지 = **"eRank ≠ capacity"**. 구체적으로 대조 논문
**arXiv:2602.02195 "State Rank Dynamics in Linear Attention LLMs"** (Sun et al., Meituan/CUHK-SZ)를
pure GDN(gated DeltaNet-2)에서 재현·반박한다.

- 논문 인과 사슬: **high state-rank ⟹ oversaturated(포화·잉여) ⟹ prunable** (high-rank head 지워도
  성능 유지하며 KV 38.9% 절감). 대상 모델 = Qwen3-Next(48층 하이브리드, post-trained).
- 우리 반박: pure linear model에선 이 사슬이 **역전**한다. saturation의 두 정의를 분리:
  - (A) 선형대수적 = full rank (방향 다 씀).
  - (B) 기능적 oversaturation = 간섭이 정보 파괴(많이 썼는데 꺼낼 게 없음).
  - 논문은 (A)⟹(B)로 단정. 우리는 pure GDN에서 **(A) ≠ (B)** 를 보인다.

모델 = **gdn2-1.3B, paper-matched 100B checkpoint** (`model-100b.pth`, sha256[:7]=6cb3072;
로컬 경로 `/home/sohyung/models/gdn2_1.3B_100b.pth`). 10B 체크포인트는 동역학이 크게 달라 사용 금지,
95B≈100B. 18층. head 288개(=layer×head). 실행은 **로컬 SLURM**(partition main, `--gres=gpu:rtx6000:1`),
venv `/home/sohyung/sh_gdn2_venv` (torch 2.13+cu130). 로그인 노드 직접 CUDA 금지(cuInit=100).

---

## 1. 완료된 작업 (Stage 1~4, 모두 실 pretrained + 실텍스트, 3 seed, 부풀림 없음)

데이터 = wikitext-103 / github code(codeparrot-clean-valid) / arxiv(ccdv/arxiv-summarization),
로컬 `data_cache/` prefetch. **합성 데이터·untrained 소형 모델 절대 금지**(interpretability에서 데이터
임의 교체 불가; MQAR류 synthetic 무의미 — 사용자 강한 원칙).

### Stage 1 — rank stratification 재현 (관측) — `260725_stage1_rank_stratification/`
- threshold-rank(Eq.6, ε=1e-4). 시간일관성 PRIMARY = 논문식 separated-pair(early anchor t≈d vs
  late 256/512/1024/2048).
- **G1a = YES**: bimodal stratification(BC 1.0), separated-pair 순위 ρ **0.982±0.015**.
- **G1b = pass (R²=0.897±0.006)**: 감쇠 r̄=exp(E[log a_t])가 threshold-rank를 R²~0.9로 예측 →
  **stratification = decay stratification**. ★ 핵심 knob: **high-rank = 저-decay = 장기 보존 /
  low-rank = 고-decay = 단기 보존.** (다음 계획의 mechanism 근거.)
- 정직한 편차: nuclear-norm cosine(separated)=0.72 (논문 0.98 미달). 순수-rank 분류(≠논문 JRNP
  Saturation Score Eq.14, α 미공개).
- 로더 = `loader_gdn2.py`: 100b 체크포인트, **kernel-intercept로 a_t 캡처**(primary) + reconstruction
  fallback. `Stage1Bundle.states_and_rbar(ids)` → (states {layer:(heads,dk,dv)}, rbar {layer:array}).

### Stage 2 — pruning 비대칭 (기능, 역전) — `260726_stage2_pruning_asymmetry/`
- **진짜 KV-state pruning**(output-ablation 아님): `head_mask.py` HeadMasker가 v_conv1d 훅으로 head
  value 채널을 0 → S_h=0 (KV 메모리 실제 제거). 4조건 origin/high/low/random 동수, 124/288 head =
  **KV 43.1% 절감**(linear-param 기준으론 15.1%; 두 방식 다 보고).
- `head_classifier.py`: `k=min(#low,#high)`로 low/high **disjoint 보장**, count-matched.
- NIAH origin=0.18 floor → **PPL이 primary**. 결과(macro PPL): origin 12.6 / **high 854±101** /
  low 39.4±3.0 / random 34.4±1.7.
- **G2 = NO, 논문과 정반대**: high-rank pruning 파국(~32×), low≈random(거의 잉여). pure GDN에선
  **high-rank = load-bearing**.

### Stage 3 — high-rank는 oversaturated인가 (메커니즘, 반박) — `260727_stage3_rank_mechanism/`
- head state S_h의 spectral content 겨냥 3 개입(실모델·실데이터, high/low/random, 3 seed):
  ① prune-fraction sweep(count-matched), ② SVD top-r 절단, ③ 스펙트럼-보존 noise(rank·에너지 유지,
  방향만 랜덤 → 내용만 파괴). segment-control drift≈0 확인.
- 결과: ① 모든 압축률에서 high 파국(38.9% 지점 high 655.8 vs low 34.9 vs random 33.1). ② SVD top-r
  기울기 high 5.86±0.48 vs low 0.033 (~176×). ③ noise delta high 16.6±0.8 vs low 0.12 (~141×).
- **G3 = YES**: high-rank는 rank 용량을 **실제로 사용** → **full rank = genuine capacity, oversaturated
  아님.** 논문의 `high-rank ⟹ oversaturated` 반박.

### Stage 4 — high-rank head의 역할: recall인가 (induction probe) — `260728_recall_role/`
- Olsson(2022) induction probe: 실 passage A(len 1024)를 [A][A](seq 2048)로 이어, per-token NLL을
  local(첫 복사, 문맥 前) vs recall(둘째 복사, 문맥 後)로 분할. `induction_gain = local − recall`.
  코드 = `induction_probe.py`, `stage4_recall_role.py`, `aggregate_seeds.py`. sanity로 seed 0 선행 후 3-seed.
- 3-seed 결과: origin gain 2.38±0.02, **high prune gain −0.006**(recall 완전 붕괴, recall==local),
  low prune 3.04, random 1.35. headroom(origin gain>0.30) 충족.
- **현 판정 = "high-rank가 induction 메커니즘을 진다"까지는 견고**(high prune 시 recall이 local
  baseline까지 무너짐 = 차이 artifact 아님).

---

## 2. ★ 방금 발견한 문제 2가지 (다음 작업에서 반드시 반영)

### (A) 질문 자체가 좁았다 — "recall(memory)"가 아니라 **memory냐 skill이냐**가 진짜 질문
사용자 원래 의도: Transformer 해석 문헌의 **"MLP = 지식(knowledge) / self-attention = routing"**
flow처럼, **rank가 head를 memory(지식·저장) 유닛과 skill(지능·연산/routing) 유닛으로 나누는가**를
보는 **rank별 interpretability**. 그런데:
- induction은 문헌상 오히려 **in-context 학습 "skill/알고리즘"의 대표 회로**(복사 규칙)이지 지식 저장이
  아님. 내가 "recall=memory"로 라벨을 잘못 붙임(문맥 회수 vs 지식 저장을 혼동).
- 매핑 정정: linear SSM에서 **recurrent state head = attention KV 아날로그(in-context 기질)**,
  **MLP = knowledge**(우리가 안 건드림). 그래서 rank별 구분은 이 기질 *안에서* **연상 memory(장거리
  key→value 저장·회수) vs 국소 routing/skill(저차원 고정 연산: 직전토큰·위치·평활)** 로 나뉜다.

### (B) `induction_gain`은 개입 하에서 신뢰 못 할 headline 지표 (사용자가 캐치)
- low prune에서 gain이 2.38→3.04로 "올랐다"고 발표할 뻔함. 하지만 절대 bits로는 **local·recall 둘 다
  악화**(low: local 3.83→5.42, recall 1.45→2.38). gain은 *차이*라, local이 recall보다 더 망가지면
  파괴적 개입이 지표를 올리는 함정. → **절대 bits + "recall-specific 손상 = Δrecall − Δlocal"** 로
  재기술. gain은 보조로 강등.
- 그런데 이 "이상함" 자체가 가설의 신호: (Δrecall−Δlocal) 부호 = **high +2.4 / low −0.7 / random +1.0**.
  low-rank(단기보존)를 지우면 **local(국소 예측)** 이 더 상하고 recall은 살아남은 high-rank가 대신
  저장 → low=국소 skill, high=장거리 memory 가설과 정합. **단 거리 조작 없는 단일 점이라 암시일 뿐.**

---

## 3. 프레이밍 판단 (사용자와 합의)

지금처럼 "prune→측정 한 판"은 **characterization("분석해봤어요")** 수준이지 강한 주장이 아니다.
Transformer급 interp 논문(induction-heads, FFN=key-value memory)이 강했던 3요소:
1. **깨끗한 인과 dissociation** ← 현재 비어 있음(채울 대상).
2. **알려진 knob로 설명되는 mechanism** ← 있음: Stage 1의 decay/retention 축(rank↔보존시간).
3. **예측력/실행적 결론** ← 있음: 대조 논문의 "high-rank 지워라"를 **뒤집는** 실용 함의
   ("high-rank=장거리 memory라 지우면 안 됨").

**목표 thesis(프레이밍)**:
> **"State rank는 head의 기능적 타입 서명이다 — 장거리 연상 memory 유닛인지 국소 routing/skill 유닛인지를
> rank가 예측하며, 이는 (i) 선행 논문의 prune 권고를 뒤집고 (ii) decay knob으로 기계적으로 설명된다."**

**정직한 최대 약점**: pure **1.3B 단일 모델**. 리뷰어는 scale/model 일반성을 물을 것 → 2차 모델
(pure Mamba2 1.3B official weight 등)에서 재현되면 크게 강해짐. 프레이밍보다 이게 더 큰 리스크.

---

## 4. 다음 실험 계획 (사용자 결정: "빠른 판별 먼저")

### 실험 X1 — 거리(range) crossover: memory vs skill dissociation (즉시)
**설계**: Stage 4 induction probe를 재사용하되 **반복 구간의 의존 거리(gap)** 만 조작.
- passage A와 그 반복 사이 거리를 gap ∈ {16, 64, 256, 1024} 로 쓸어감(또는 A 내부에서 참조
  거리별로 recall 토큰을 층화). high/low/random pruning × gap.
- **주지표(수정)**: 절대 recall_bits, 그리고 **recall-specific 손상 = Δrecall − Δlocal**(개입별·거리별).
  induction_gain 단독 금지.
**예측(가설 H)**: high-rank pruning 손상은 **장거리 gap**에 몰리고, low-rank pruning 손상은 **단거리
gap**에 몰린다 → **crossover** 발생.
**결정 규칙**:
- crossover 있음 → thesis 성립. full 서사·그림 + (여력 시) Mamba2 2차 모델 확인으로 확장.
- crossover 없음 → "rank는 그냥 retention축일 뿐 memory/skill 안 가름"의 정직한 null → 방향 재검토.
**비용**: Stage 4가 seed당 ~1분(teacher-forced forward only). gap 스윕 붙여도 저렴. 3 seed.

### 실험 X2 (조건부, X1 crossover 확인 시) — skill 축 직접 조작
X1이 거리로 암시를 주면, **국소 skill 과제**(직전토큰/국소 syntax 예측, 문맥 memory 불필요)를 별도로
넣어 low-rank가 그걸 전담하는지 직접 확인. (구현·검증 1건 추가 필요.)

### 실험 X3 (논문화 필수 조건) — 2차 모델 일반성
pure Mamba2 1.3B(official weight)에서 Stage 1(decay↔rank) + X1(crossover) 재현. 되면 "아키텍처
공통 원리"로 격상. (mamba-ssm은 torch 2.13+cu130 빌드 차단 이력 → 환경 별도 확인 필요.)

---

## 5. 재현 포인터 (codex가 바로 이어받을 것)

- 체크포인트: `/home/sohyung/models/gdn2_1.3B_100b.pth` (6cb3072). env `GDN2_CKPT_PATH`로 주입.
- venv: `source /home/sohyung/sh_gdn2_venv/bin/activate`. 실행은 `sbatch`(rtx6000). 로그
  `/home/sohyung/sh_logs/`.
- Stage 4 실행 예: `sbatch 260728_recall_role/run_recall_role.sbatch <seed>` → 3개 후
  `run_aggregate.sbatch`(afterok 의존). 헬퍼 `submit_recall_role_seeds.sh`.
- 판정 로직 규약: headroom 게이트(origin gain>0.30, 다수 seed), 효과크기 margin=0.10 bits,
  seed 부호 일관(다수결). **단 headline은 induction_gain이 아니라 절대 bits + (Δrecall−Δlocal)로 교체.**
- 결과 JSON: `260728_recall_role/results/aggregate_stage4.json`, `..._seed{0,1,2}/stage4_report*.json`.
- 통합 서사 문서: `260728_3stage_integrated_report.md` (Stage 1~3). Stage 4는 `260728_recall_role/STAGE4_RESULTS.md`.
- go/no-go 리뷰: `working_note/260725·26·27·28_exp_review_*.md`.
- 실험 게이트 하네스: `sh_experiment` 워크플로(구현→3렌즈 검증→논문 대조→go/no-go). 실험 코드 요청 시
  이걸 쓸지 먼저 물어볼 것.

## 6. 절대 지켜야 할 원칙 (사용자)
- 합성 데이터·untrained 소형 모델 금지. 실 pretrained + 실텍스트만.
- 세팅은 세션 간 CONSTANT(main-table-ready, 파일럿 금지). 수치 부풀림 금지, 한계 정직 기술.
- 파일명 날짜 프리픽스 `YYMMDD_`. 커밋/푸시는 사용자 요청 시에만.
- **induction_gain 단독 판정 금지**(위 (B)). memory/skill 분리는 거리 crossover(X1)로만 주장.
