# 실험 리뷰 보고서: Stage 4 recall-role induction probe

- 작성일: 2026-07-28
- slug: recall-role-induction
- 브랜치: `sh_exp/recall-role-induction`
- 작업 경로: `/home/sohyung/SSM_Rank_Analysis/.sh_exp_worktrees/recall-role-induction/260728_recall_role`
- **판정: GO-WITH-FIXES**

## 실험 요약

Stage 4는 신규 폴더 `260728_recall_role/`에서 pure gdn2-1.3B(100B paper-matched 체크포인트)의 HIGH-rank head가 in-context RECALL(기억) 유닛인지를 role 차원에서 검정한다. Stage 2가 high-rank KV-state pruning이 PPL을 파국시킴(12.6→854, 약 32배)을 이미 보였으나 PPL은 뭉뚱그린 지표라 '무슨 역할'인지 드러나지 않으므로, 표준 induction-head 프로브로 recall을 분리 측정한다. 방법은 실텍스트 반복 induction 분해다(synthetic 분포 교체 없음). 각 실 passage A(길이 L)를 2번 반복한 seq=[A][A]에서 토큰당 NLL bits를 local_bits(첫 등장 0..L, recall 불필요)와 recall_bits(둘째 등장 L..2L, in-context recall 가능)로 분리하고 induction_gain=local_bits−recall_bits(>0이면 recall 작동)를 측정한다. 4조건(origin/prune-high/prune-low/prune-random, k=min disjoint count-matched, Stage2 KV-state v-zeroing 재사용)×3 seed로 실행한다.

구현은 스펙대로 완료되었고 CPU 스모크 테스트(classifier count-match/disjoint/agreement=1.0, [A][A] 2L 분할, planted 서명에서 induction_gain>0, verdict 서명, 헤드룸 게이트 발화)를 통과했다. 격리 워크트리에서만 작업했으며 원본 main 트리는 무변경이고 커밋(8af7b54, 8파일 932 insertions)까지 완료되었다(push 안 함). 실제 GPU real run은 아직 실행하지 않았다.

판정을 GO-WITH-FIXES로 두는 이유는 코드가 실행 가능하고 인프라 규칙을 준수하지만, 실험 결론(고랭크=recall 유닛 → 논문 oversaturation 반증)의 신뢰도를 직접 위협하는 major 결함 3건(판정 마진 부재, recall-특이성 판정의 산술적 교란, 서사 프레이밍 오귀속)이 남아 있어 real run 전에 반드시 손봐야 하기 때문이다.

## 실험 스펙 요약

- **가설 H(recall-role)**: pure gdn2-1.3B에서 HIGH-rank head는 in-context recall(기억) 유닛이다. 따라서 high-rank head의 KV-state를 pruning하면 induction_gain이 특이적으로 붕괴하고, LOW-rank/random pruning은 그렇지 않다. 서명은 Δrecall(high)≫Δlocal(high)(recall-특이적, PPL 전반 저하가 아니라 recall 구간에 집중) ∧ Δrecall(high)≫Δrecall(low/random)이다.
- **DV(주지표)**: 조건별 recall_bits, local_bits, induction_gain. 손상 Δrecall=recall_bits(cond)−recall_bits(origin), Δlocal=local_bits(cond)−local_bits(origin). 모두 실텍스트(wikitext/github/arxiv) 토큰당 bits.
- **NULL/역전**: Δrecall(high)이 Δlocal(high)와 유사하거나 low/random과 유사하면 high-rank가 recall-특이적이 아님 → 한계(LIMITATION)로 기록하며 코드 실패가 아니다.
- **헤드룸 게이트**: origin induction_gain>0.3 bits(사전등록 임계)여야 측정 가능, 미만이면 UNTESTABLE_HEADROOM.
- **핵심 설정(PIN 승계)**:
  - [PIN-1] model=gdn2-1.3B, 100B 체크포인트 ONLY(`/home/sohyung/models/gdn2_1.3B_100b.pth`, sha256[:16]=4b03319f, 약 17.4GB), 10B HARD-REJECT. Config.from_name("gdn2_1.3B")+strict=False+bf16+fused_recurrent. n_layer=18, mixer num_heads==num_v_heads==16 → 18×16=288 heads(config n_head=18은 attention 필드로 mixer와 무관), Stage2 "124/288=43.06%"와 정합. 첫 real run provenance에 num_heads/num_v_heads/총head수(288) 명시 로깅.
  - [PIN-2] head rank metric=threshold_rank eps=1e-4(논문 Eq.6), θ_R=0.5, head_classifier.classify로 k=min disjoint bottom-k(low)/top-k(high) count-match, 교차도메인 일치 약 0.97 sanity gate. 순수-rank 분류(≠JRNP Saturation Score, α 미공개 DECLARED DEVIATION).
  - [PIN-4] 프루닝=Stage2 head_mask.HeadMasker 재사용(v_conv1d head value 채널 0 → S_h=0, true KV-state pruning). 4조건, k IDENTICAL across high/low/random.
  - [PIN-5] data=data_cache/ 실텍스트 3도메인 + --require-real-data(synthetic fallback→INVALID hard-fail). induction 시퀀스는 실텍스트 2배 반복(분포 교체 아님), passage 길이 L=1024→2L=2048(block_size 4096 이내) FROZEN, ≥16 passages/domain.
  - [PIN-6] seed≥3(0,1,2), 전 프레임워크 seed pin, seed당 1 sbatch job + afterok 의존성 aggregate로 mean±std.
  - NLL=analysis.token_nll_bits 재사용(재구현 금지), aggregate_seeds 패턴 재사용, YYMMDD_ 파일 접두, incremental flush+resume.
- **target 인프라**: greenbeard. GPU gpu01(RTX PRO 6000 Blackwell 96GB) 또는 SLURM partition main --gres=gpu:rtx6000:1. teacher-forced forward만(autoregressive decode 없음 → Stage2 NIAH 지배 비용 제거되어 Stage2보다 빠름). per-seed 예산 6h(예상 15~20분). HF_HOME=/data2/sohyung/hf_home(root disk 회피), BLAS threads=4.

## 잔존 블로커

없음. real run을 막는 하드 블로커는 없다. 다만 아래 발견 사항의 major 3건은 real run 전에 해결할 것을 권고한다(결론 신뢰도 직결).

## 발견 사항

### major

| reviewer | title | file | suggestedFix |
|---|---|---|---|
| code-correctness | 판정 서명이 스펙의 '≫'(much-greater)가 아니라 사실상 '>'(margin=1e-9)로 구현되어 노이즈 수준 차이도 RECALL_ROLE_SUPPORTED로 통과 | stage4_recall_role.py:73,158-163; aggregate_seeds.py:86-90 | 사전등록 effect-size margin(예: Δ 차이 ≥0.3 bits 또는 origin induction_gain의 일정 비율)과 seed간 부호 일관성/유의성 체크(예: mean−std>0, 또는 3 seed 모두 동일 방향)를 판정 조건에 추가. margin 값은 실행 전 사전등록으로 고정. |
| literature | 핵심 서사가 원논문 주장 방향을 뒤집어 서술 — '반증' 프레이밍의 대전제(paper: 고랭크=recall/retrieval)가 실제 논문(low-rank=retrieval)과 정반대 | aggregate_seeds.py:116-126, stage4_recall_role.py:8-14 | 주석·README·verdict 문구에서 논문 주장을 정확히 표기: 논문=HIGH-rank=oversaturated/redundant(prunable), LOW-rank=retrieval-indispensable. 본 실험 명제=HIGH-rank=recall unit → 논문의 'HIGH-rank=prunable junk'를 role 차원에서 반증. 'recall 유닛'을 논문 예측으로 귀속시키는 표현만 제거. |
| alignment | 핵심 교란: 'recall-특이성' 판정(Δrecall>Δlocal)이 recall 유닛과 일반 load-bearing 헤드를 구분하지 못할 수 있음 — recall/local의 비대칭 베이스라인 때문 | (verdict 로직) stage4_recall_role.py | 판정 (a)를 절대 Δ 비교에서 headroom-정규화 비교로 보강(예 recall_collapse_frac = Δrecall / origin_induction_gain). (b) high vs low/random 대조가 이 실험의 진짜 특이성 축임을 README/verdict에 명시하고 (a)는 보조/필요조건으로 격하. origin의 local_bits/recall_bits 절대값과 비대칭 headroom을 aggregate 최상위에 노출. |

### minor

| reviewer | title | file | suggestedFix |
|---|---|---|---|
| code-correctness | 'Stage2 origin PPL 12.6 재현' sanity가 실제로 검증되지 않음 — 이 프로브는 그 PPL을 계산하지 않는다 | stage4_recall_role.py:114-127; induction_probe.py:100-124 | origin condition에서 local_bits(첫 등장, 비반복 실텍스트)를 Stage2 origin bits/token(log2(12.6)≈3.65)과 대조하는 sanity 로그/느슨한 range gate 추가(불일치 시 WARN). 최소한 report에 origin local PPL=2^local_bits 기록. |
| code-correctness | 조건별 bits가 NaN이면 delta가 NaN→'NaN>x'가 조용히 False가 되어 데이터 파손이 NULL_OR_REVERSAL로 위장됨 | stage4_recall_role.py:149-163 | compute_verdict 진입부에서 origin/high/low/random의 local_bits·recall_bits 유한성 assert, 또는 비유한 시 status='INVALID_METRICS'로 별도 분기(NULL과 구분). |
| code-repro | 재현 sanity gate가 코드에 자동화되어 있지 않음 — Stage2 origin PPL(12.6±0.4) 재현 여부는 수동 대조에 의존 | stage4_recall_role.py:71,214-217 | origin 조건에서 macro bits→exp로 통합 PPL을 부수 계산해 report에 남기고, Stage2 origin PPL 대비 |Δ| 임계 초과 시 WARN 플래그. 계산 비용 거의 0(기존 per-token bits 재사용). |
| code-repro | 의존성 고정: sbatch venv는 핀되나 requirements.lock이 Stage4에 명시 참조/복사되지 않음 | run_recall_role.sbatch:19 | README에 '../260725_stage1_rank_stratification/requirements.lock 기준 venv'라고 명시하거나 Stage4에 심볼릭/복사본 배치. runtime_versions에 fla-core/triton 버전 추가 권장. |
| literature | 논문의 pruning-asymmetry 예측을 low/random 대조군에 귀속시킨 서술이 논문과 어긋남 — 논문은 LOW-rank pruning이 파국(retrieval 붕괴)이라 예측 | stage4_recall_role.py:16-24 | README/verdict에 대조군 예측 출처를 구분 표기: 논문 예측(low=recall critical)과 Stage2 관측(low≈random redundant)이 상충하며 본 실험은 Stage2 관측을 작업가설로 승계함을 명시. |
| literature | induction 프로브가 원류 Olsson 방법론과 두 지점에서 DECLARED DEVIATION — 실텍스트 사용 및 score 정의; 표준 관행 범위 내이나 명시 필요 | induction_probe.py:1-25 | README에 Olsson 원정의(random-token, 500th−50th) 대비 본 변형(real-text [A][A], local−recall)을 명시. headroom 게이트 실패 시 '실텍스트 첫-copy 예측가능성으로 local_bits가 낮아 gain 압축' 가능성을 한계로 사전기재. 가능하면 random-token control 1개 도메인 추가로 편향 정량화. |
| literature | 헤드룸 임계 0.3 bits는 논문 근거 없음 — 실험자 사전등록값(스펙이 이미 인지); 문헌상 표준 induction gain 값과 대조 불가 | stage4_recall_role.py:72, aggregate_seeds.py:23 | 결과 보고 시 0.3 bits가 문헌 앵커 없는 내부 사전등록값임을 caveat로 유지. origin induction_gain 실측 분포를 먼저 보고 임계 타당성 사후 점검. |
| alignment | random 대조가 seed당 1 draw뿐 — high vs random 특이성이 단일 표본 대조라 통계적으로 약함 | stage4_recall_role.py:93 | seed당 random draw를 여러 개(예 5~10)로 늘려 Δrecall(random) 분포를 얻고, high가 상위 꼬리를 넘는지(예 high > mean+2std, 또는 empirical p) 판정. aggregate에서 Δrecall_high와 Δrecall_random의 mean±std를 비교하고 겹침 여부를 verdict에 반영. k=43%가 큰 경우 random이 high를 우연히 포함하는 문제를 README caveat로 명시. |
| alignment | per-seed verdict 마진(1e-9)과 aggregate 마진(0)이 사실상 무마진 — '≫'(훨씬 큼)이 아니라 '>'(단순 큼)만 검정 | stage4_recall_role.py:73 | '≫'를 실질 마진으로 구현(예 Δrecall(high) − max(Δrecall_low, Δrecall_random) > c·std 또는 절대 0.1 bits 사전등록). 또는 seed간 부호 일치(2/3 또는 3/3 majority) 게이트 추가. |
| alignment | induction 시퀀스가 실제 로더의 '문서 경계 무시 concat 후 절단' 청크라 passage A가 응집된 단일 문단이 아님 — 프로브 해석 caveat | (data_stage1 로더) | probe_meta의 도메인별 induction 결과를 aggregate에서 도메인별 origin induction_gain으로 노출하고 헤드룸 게이트를 도메인별로도 점검(특정 도메인만 gain<0.3이면 flag/제외). README에 'passage A는 concat-절단 슬라이스이며 표준 induction 관행상 허용'임을 명시. |
| resources | 경미한 낭비: report JSON에 conditions와 conditions_full 이중 저장 (영향 미미) | stage4_recall_role.py:267-271, induction_probe.py:117-124 | 선택: report에서 trimmed 'conditions' 제거하고 conditions_full 하나만 유지(집계는 이미 스칼라만 읽음). 기능 영향 없음. |

### 통과·정합 확인 항목 (조치 불요, 참고용)

- **code-smoke**: CPU 스모크 통과(k=16 agreement=1.000, induction_macro local_bits=4.596/recall_bits=0.030/gain=4.565>0, verdict=RECALL_ROLE_SUPPORTED, headroom UNTESTABLE_HEADROOM 정상 발화, dRecall(high=3.5)≫dLocal(high=0.1) 및 ≫dRecall(low=0.3/random=0.4)). import 체인/재사용 모듈 인터페이스 전부 정합(analysis/loader_gdn2/data_stage1/head_classifier/head_mask/common). aggregate 파이프라인 합성 3-seed 통과. sbatch 3종 bash -n + `sbatch --test-only` 통과(Job to start on gpu01 partition main). 격리 워크트리 클린, results/ gitignore, main 트리 무변경.
- **code-repro**: 시드 고정 전 프레임워크 완비(global RNG + data-shuffle + random-mask draw 재현 가능). 설정·커맨드라인 기록(args/git_head/runtime_versions/ckpt_provenance/head_provenance) 전부 저장. --resume + 조건별 incremental flush로 중단 복구. YYMMDD_ 명명 컨벤션 준수. seed 0/1/2 독립 job + afterok aggregate로 main-table-ready 요건 충족. 고정길이 passage 로더 계약 보장(2L-1 assertion·n_passages≥16 항상 성립). reset_shared_cache로 조건 간 recurrent-state 오염 제거(Stage2 검증 fix 동일).
- **literature 정합**: Rank_eff threshold ε=1e-4 논문과 정확 일치. Qwen3-Next=48L post-trained NIAH 93.8/46.9/90.6 및 KV 38.9% 감축 Table 1 정합(TARGET/non-pass caveat 올바름). JRNP Saturation Score Eq.14에 α 존재·값 미공개 확인(DECLARED DEVIATION 정당). token_nll_bits 인덱싱(길이 2L−1, seam bits[L−1] 제외) 경계 오프셋 정확.
- **resources**: 인프라 규칙 준수(partition main + --gres=gpu:rtx6000:1, 로그인 노드 직접 CUDA 금지 주석, HF_HOME=/data2, BLAS threads=4, aggregate는 GPU 미요청). OOM 위험 없음(상주 <5GB vs 96GB). per-seed <20분(Stage2 실측 대비, NIAH decode 제거). 데이터 캐시 충분해 무음 tiling-pad 미발동.

## 자동 수정 내역

없음(자동 수정 미적용).

## 문헌 대조 결과

- **대조 원논문 arXiv:2602.02195 "State Rank Dynamics in Linear Attention LLMs"** (PDF: `/home/sohyung/sohyung's_brain/state rank dynamics in linear attention LLMs.pdf`, HTML: https://arxiv.org/html/2602.02195). 실제 논문 §head pruning 확인 결과: 논문이 pruning하며 'redundant'로 부르는 것은 **HIGH-rank head**이고(NIAH 93.8→90.6, negligible degradation), retrieval/reasoning에 indispensable한 것은 **LOW-rank head**다(pruning 시 93.8→46.9 catastrophic collapse; "low-rank heads are indispensable for model reasoning, whereas high-rank heads exhibit significant redundancy"). 따라서 '논문=고랭크=recall/retrieval 유닛'이라는 keyLiterature/일부 서술의 프레이밍은 방향이 틀렸다. 다만 본 실험의 실제 검정 명제('HIGH-rank=recall 유닛')는 논문(HIGH-rank=redundant junk)과 반대이므로 '반증' 관계 자체는 성립하며, aggregate_seeds.py:120('refutes the paper's oversaturation(high-rank=prunable junk) reading')는 정확하다. 문제는 verdict 코드 로직(방향 무관)이 아니라 'recall 유닛'을 논문 예측으로 귀속시킨 서술뿐이다(major, 서사 정확도).
- **pruning-asymmetry 대조군 예측**: 논문(https://arxiv.org/html/2602.02195)은 LOW-rank pruning이 파국(NIAH 93.8→46.9, retrieval 붕괴)이라 예측한다. 그러나 사내 Stage2(STAGE2_RESULTS.md:19, low-rank prune macro PPL 39.4 vs random 34.4 거의 동급; :25-34)는 low≈random redundant로 나와 논문 예측을 역전시켰다. 본 Stage4의 대조군 해석('low/random이 recall을 덜 손상')은 논문이 아니라 Stage2 경험 결과 승계임을 verdict에서 명시해야 한다(minor).
- **induction 프로브 원류 Olsson et al. 2022 "In-context Learning and Induction Heads"** (https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) 및 **Elhage et al. 2021 "A Mathematical Framework for Transformer Circuits"** (induction-head 개념 정의). Olsson 원정의는 (1) RANDOM 토큰 반복, (2) ICL score=500th token loss − 50th token 평균 loss(단일 반복 내 위치차)다. 본 실험은 (1) 실텍스트 2배 반복 [A][A], (2) score=local_bits−recall_bits로 재정의한 DECLARED DEVIATION이다. 실텍스트 반복은 mech-interp 통용 변형이나, 첫 copy에 자연어 n-gram 예측가능성이 있어 local_bits가 낮아지면 induction_gain을 과소평가할 수 있고 recall_bits에 induction 외 요인이 섞일 수 있다 — headroom 게이트 통과/실패에 영향 가능한 측정 편향(minor, declared).
- **헤드룸 임계 0.3 bits**: 논문(ε=1e-4만 존재, bits 임계 없음)과 Olsson(500th−50th 상대 지표, 고정 bits 임계 없음) 어디에도 외부 앵커가 없는 내부 사전등록값이다. UNTESTABLE_HEADROOM 발생 시 '문헌 표준 미달'이 아니라 '자체 임계 미달'로 해석해야 한다(minor, 코드가 이미 [PREREG] 표기).

## 실행 전 체크리스트

1. **[major 우선] 판정 마진 수정**: RECALL_SPECIFIC_MARGIN(1e-9) 및 aggregate의 순수 '>'를 실질 effect-size margin(사전등록 절대 bits 또는 origin gain 비율)으로 교체하고, seed간 부호 일관성/유의성(mean−std>0 또는 majority) 게이트 추가. margin 값은 real run 전 사전등록으로 고정.
2. **[major] recall-특이성 판정 보강**: (a) Δrecall>Δlocal을 headroom-정규화(recall_collapse_frac=Δrecall/origin_induction_gain)로 보강하고 (a)를 보조 조건으로 격하, (b) high vs low/random 대조를 진짜 특이성 축으로 명시. origin local_bits/recall_bits 절대값과 비대칭 headroom을 aggregate 최상위에 노출.
3. **[major] 서사 프레이밍 정정**: README/주석/verdict에서 '논문=고랭크=recall 유닛'을 제거하고 '논문=HIGH-rank=oversaturated/redundant(prunable), LOW-rank=retrieval-indispensable; 본 실험=HIGH-rank=recall unit → 논문 반증'으로 정확히 표기.
4. **PPL sanity gate 추가**: origin 조건에서 통합 PPL(2^macro_bits)을 부수 계산해 report에 남기고 Stage2 origin PPL(12.6±0.4) 대비 |Δ| 초과 시 WARN. origin local PPL도 log2(12.6)≈3.65 bits 근방 대조 로그.
5. **NaN 방어**: compute_verdict 진입부에 4조건 local/recall_bits 유한성 assert 또는 INVALID_METRICS 분기(NULL과 구분).
6. **인프라 규칙(greenbeard) 준수**:
   - GPU real run은 반드시 SLURM으로 제출한다 — 로그인 노드에서 직접 CUDA 실행 금지. `sbatch run_recall_role.sbatch` / `bash submit_recall_role_seeds.sh`.
   - partition main, `--gres=gpu:rtx6000:1`(또는 gpu01 RTX PRO 6000 Blackwell 96GB). aggregate 잡은 GPU 미요청(CPU 전용) 유지.
   - `export GDN2_CKPT_PATH=/home/sohyung/models/gdn2_1.3B_100b.pth`(100B ONLY, 10B HARD-REJECT).
   - HF_HOME=/data2/sohyung/hf_home(root disk 회피), OMP/OPENBLAS/MKL_NUM_THREADS=4(SVD oversubscription 방지).
   - `--require-real-data` 유지(synthetic fallback→INVALID hard-fail).
7. **단일 seed 먼저**: real run은 seed 1개(`sbatch run_recall_role.sbatch 0`)를 먼저 돌려 (a) origin induction_gain>0.3 헤드룸 통과, (b) origin bits→PPL이 Stage2 12.6 근방 재현, (c) provenance num_heads=16/num_v_heads=16/총 288 로깅을 확인한 뒤 3-seed(0/1/2) 배치 + afterok aggregate 제출.
8. **결과 정합**: 같은 로더·분류·마스킹 재사용이므로 origin PPL이 Stage2와 재현되어야 sanity — 재현 실패 시 로더/데이터 drift 의심. main-table-ready(pilot 아님): 3 seed mean±std, per-seed sbatch, 의존성 aggregate 필수.
