# 실험 리뷰 보고서: Stage 3 — GDN2 고랭크 헤드의 진짜 용량(genuine capacity) 검정

작성일: 2026-07-27
슬러그: gdn2-rank-genuine-capacity
브랜치: sh_exp/gdn2-rank-genuine-capacity
판정: **GO-WITH-FIXES**

## 실험 요약

실제 사전학습된 gdn2-1.3B(100B 체크포인트)와 실제 텍스트(data_cache/)를 사용하여, 논문 arXiv:2602.02195의 인과 사슬 "full rank ⟹ oversaturated ⟹ prunable"을 반박하는 Stage 3 실험이다. 핵심은 saturation의 두 정의를 분리하는 데 있다: (A) 선형대수적 정의 = full rank, (B) 기능적 정의 = 용량 초과로 정보를 파괴하는 간섭. Stage 2는 이미 역전 현상(고랭크 pruning은 파국적: PPL 12.6→854, 약 68배; 저랭크 ≈ random)을 보였고, Stage 3는 고랭크 헤드가 그 랭크 용량을 실제로 "사용"하는지(genuine) 아니면 saturated junk를 담고 있는지를 각 헤드 순환상태 S_h의 스펙트럼 내용(spectral content)을 겨냥한 사후(post-hoc) 수술로 검정한다. 세 가지 개입(prune-fraction 스윕 / SVD top-r 절단 / 스펙트럼-보존 노이즈 치환)을 high / low / random 헤드 그룹에 적용하며, 3 seed, seed당 sbatch + 의존성 aggregate로 구성된 main-table-ready 실험이다.

판정 근거를 요약하면, 핵심 설계(세그먼트 상태 캐리, 스펙트럼-노이즈 불변식, 3렌즈 커널 정합, 문헌 4종 수치 대조)가 모두 소스 검증을 통과했고 CPU 스모크가 전부 정상 통과했으나, int-1 random 그룹의 count-match 결함(major)과 config n_head=18 vs 프로즈 "16 heads" 불일치 확인(major)이 실행 전 반드시 정리되어야 하므로 GO-WITH-FIXES이다.

## 실험 스펙 요약

### 가설 / DV

- **H (G3)**: pure 사전학습 gdn2-1.3B에서 고랭크 헤드는 상태의 스펙트럼 용량을 채우는 정보를 담고 있다(정의-A full rank가 실제로 사용됨). 따라서 고랭크 S_h의 스펙트럼 CONTENT를 훼손하면(SVD top-r 절단 및 스펙트럼-보존 특이벡터 랜덤화) 저랭크·count-matched random 헤드에 동일 조작을 가할 때보다 언어모델링 성능이 훨씬 크게 저하된다. 저랭크 헤드는 이미 저차원이라 거의 둔감하다.
- **DV(주지표)**: origin 대비 macro PPL delta(wikitext/github/arxiv 실텍스트에서 토큰당 비트의 2^mean), 개입 dose(prune fraction; retained-rank ratio r; ladder step origin→top-r→spectrum-noise→zero)의 함수, 헤드 그룹별(high/low/random).
- **DV(보조, headroom-gated)**: S-NIAH RULER-multikey 검색 정확도. origin ≥ 0.30일 때만 보고, 아니면 UNTESTABLE_FLOOR로 표기(Stage 2 origin=0.18). 따라서 PPL이 PRIMARY DV이다.
- **효과 서명**: (int-2) r 감소에 따른 PPL 기울기가 high는 가파르고 low는 평평; (int-3) 스펙트럼-노이즈 PPL이 high에서 origin보다 훨씬 큼(랭크·에너지 보존에도 content가 중요), junk는 origin과 유사. 대조 지표 = delta(high) − delta(low), 양쪽 대 delta(random), 3-seed mean±std.
- **NULL/역전 해석**: 고랭크가 top-r/spectrum-noise에 평평하면 고랭크 랭크가 진짜 idle/redundant라는 뜻으로, 코드 실패가 아니라 한계(PREREG)로 기록한다.

### 핵심 설정 (PIN, 세션 간 CONSTANT)

- **[PIN-1]** model=gdn2-1.3B, Config.from_name("gdn2_1.3B"), load_state_dict(strict=False), bf16, mode="fused_recurrent"; checkpoint=100B `/home/sohyung/models/gdn2_1.3B_100b.pth`(17.4GB, sha256[:16]=4b03319f; 95B 수용, 10B HARD-REJECT); 18 layers × 16 heads = 288 heads, num_v_heads==num_heads==16 (no GVA). ※아래 major 항목 참조.
- **[PIN-2]** head rank metric = threshold_rank eps=1e-4, Rank_eff=Σ I(σ_i>eps·σ_1)(논문 Eq.6), cap d=min(dk,dv); θ_R=0.5(정규화 랭크); head_classifier.classify로 k=min(#low,#high) DISJOINT count-matched bottom-k/top-k(isdisjoint assert); 교차도메인 일치 ~0.97. 순수-랭크 분류는 JRNP Eq.14의 DECLARED DEVIATION(α 미공개).
- **[PIN-5]** data = data_cache/ 실텍스트 3도메인(wikitext-103 / codeparrot-clean / ccdv-arxiv), ≥16 seqs × up to 2048 tok, --require-real-data(synthetic fallback → INVALID).
- **[PIN-6]** seed≥3 (0,1,2), PYTHONHASHSEED/random/numpy/torch/cuda 전부 pin, seed당 1 sbatch job, 의존성 job으로 mean±std aggregate.
- **개입 그리드(PREREG 동결)**: (1) prune-fraction {0, 0.10, 0.20, 0.30, 0.389(paper KV), 0.43(our Stage2), 0.50, 0.60}, KV-state v-zeroing; (2) SVD top-r 유지비율 r/cap ∈ {1.0, 0.75, 0.50, 0.25, 0.125, 0}; (3) 스펙트럼-보존 노이즈 S_noise = U_rand diag(σ_orig) V_rand^T (Haar-random 직교 U,V; 특이값 동일 → threshold_rank+nuclear_norm+energy 불변, 특이벡터 랜덤화), ladder origin→top-r(r=0.5)→spectrum-noise→zero. segment_len=256 FROZEN.

### target 인프라

greenbeard SLURM. RTX PRO 6000 Blackwell 96GB(gpu01). sbatch 전용, 로그인 노드 직접 CUDA 금지. venv `/home/sohyung/sh_gdn2_venv`(torch 2.13+cu130). HF_HOME=/data2, TRITON_CACHE_DIR 지정, OMP/MKL=4. seed당 6h 예산(예상 75–123분, 3–5배 여유).

## 잔존 블로커

없음. 잔존 블로커 목록은 비어 있으며, 실행 자체를 막는 항목은 존재하지 않는다. 다만 아래 major 발견 2건은 실행 전 반드시 정리(수정 또는 확인)해야 결과가 role 효과로 해석 가능하다.

## 발견 사항

자동 수정은 수행되지 않았다(자동 수정 내역: null). 아래는 심각도별 정리이며, blocker는 없다.

### Major (실행 전 반드시 처리)

| reviewer | title | file | suggestedFix |
|---|---|---|---|
| code-correctness | int-1 random 그룹이 count-matched가 아님: frac×288 헤드를 뽑아 high/low의 frac×k(~123) 대비 약 2.3배 헤드 수로 대조군을 교란 | stage3_mechanism.py:165-168 | random arm도 round(frac*k) 크기(high/low와 동일 n)를 all_heads(또는 all_heads − high∪low)에서 시드된 순열로 추출. PREREGISTRATION.md:54를 "random arm은 각 frac에서 high/low와 동일한 헤드 수를 마스킹"으로 재기술. whole-pool 스윕이 의도였다면 별도 count-matched random arm을 추가하여 0.389/0.43 마크에서의 role 비교를 깨끗하게 함. (주: int-1은 보고 전용이며 g3_seed=topr_dissoc AND spec_dissoc는 int-2/int-3만으로 구동되어 주판정을 뒤집지 않음 → major, blocker 아님) |
| literature | config의 n_head=18과 스펙의 "16 heads(num_heads==num_v_heads==16)" 주장 간 확인 필요 | /home/sohyung/long-gdn/dsc/lit_gpt/config.py:118-135 | 모델 빌더(gdn2_1.3B → GatedDeltaNet2)에서 실제 self.num_heads를 런타임 assert로 로깅해 16 vs 18 확정. head_classifier 산출 헤드 총수(288 vs 324)를 Stage1/2 로그와 대조. Stage1/2가 동일 로더로 288을 산출했다면 정합이나, 첫 real run provenance에 num_heads/num_v_heads/head 총수를 명시 기록 |

### Minor — code-correctness

| reviewer | title | file | suggestedFix |
|---|---|---|---|
| code-correctness | 스펙트럼-노이즈 rank/nuclear-norm 불변식이 selftest의 CPU float64에서만 검증되고, 실제 주입되는 live bf16 상태에서는 미검증 — bf16 round-trip이 스펙트럼을 교란할 수 있음 | state_surgery.py:149-156 | S_noise를 bf16으로 캐스팅한 뒤 bf16 텐서에서 threshold_rank+nuclear_norm 재측정(그룹·조건당 1회, 저렴)해 PPL과 함께 로깅; 또는 불변식이 float-precision임을 명시하고 캐스팅 후 rank/nuclear delta를 리포트 JSON에 기록 |
| code-correctness | 다중세그먼트 surgery-off "origin drift vs single-shot"이 진단으로만 보고되고 하드 게이트가 없음 — 실제 커널에서 큰 drift가 나면 모든 int-2/int-3 delta를 조용히 편향 | stage3_mechanism.py:235-268 | compute_verdict/aggregate에 multiseg_origin_drift_vs_single이 사전등록된 소량 허용치(예: single-shot macro PPL의 >1%)를 넘으면 seed를 실패/플래그 처리하는 하드 게이트 추가 |
| code-correctness | deterministic-kernel / CUBLAS_WORKSPACE_CONFIG pinning 없음; seed는 고정되나 triton fused_recurrent 경로가 결정적으로 강제되지 않음 | stage3_mechanism.py:109-120 | PPL은 teacher-forced(샘플링 없음)이므로 커널 float-associativity 수준 결정성만 있고 cross-seed mean±std가 보고 불확실성임을 문서화; 비트재현이 필요하면 sbatch에 CUBLAS_WORKSPACE_CONFIG=:4096:8 export |

### Minor — code-smoke

| reviewer | title | file | suggestedFix |
|---|---|---|---|
| code-smoke | 모든 CPU 스모크 통과: 4개 파일 py_compile OK, state_surgery 불변식·stage3 드라이버·probe 모두 정상 실행 | stage3_mechanism.py | (조치 불필요) |
| code-smoke | sbatch 스크립트 문법·리소스 지시자·의존성 배선 모두 정상 (sbatch --test-only 통과) | run_stage3.sbatch | (조치 불필요) |
| code-smoke | 재사용 모듈·common·data_cache 임포트 경로가 워크트리 내에서 전부 해소됨 | stage3_mechanism.py | (조치 불필요) |
| code-smoke | 판정이 세그먼트化 drift에 오염되지 않도록 그룹별 자기-origin 대비 delta로 설계됨 — 저자 caveat이 코드상 올바르게 격리 | stage3_mechanism.py | (조치 불필요; 첫 real run에서 segment_control.json 육안 확인) |
| code-smoke | 스모크 JSON 덤프에 int-1 결과 누락(int2/int3만 저장) — 값은 verdict에 반영되나 아티팩트에는 미기록 | stage3_mechanism.py:703 | 스모크 json.dump에 'int1_prune_fraction': int1 키 추가하여 실주행 스키마와 일치(기능 영향 없음) |

### Minor — code-repro

| reviewer | title | file | suggestedFix |
|---|---|---|---|
| code-repro | 시드 고정이 python/numpy/torch/cuda/hash 전 프레임워크에 완전하고, 스펙트럼-노이즈 RNG까지 (seed,layer,head)로 유도되어 재현 가능 | stage3_mechanism.py:109-120; run_stage3.sbatch:42 | 완전 비트재현이 필요하면 CUBLAS_WORKSPACE_CONFIG=:4096:8 + torch.use_deterministic_algorithms(True, warn_only=True); mean±std 워크플로엔 필수 아님 |
| code-repro | 설정/커맨드라인/프로버넌스 기록 완비 — args, git HEAD, 런타임 버전, ckpt realpath/size/sha, segment_len, grid 전부 직렬화 | stage3_mechanism.py:516-533; loader_gdn2.py:219-249 | 선택적으로 전체파일 sha256 추가 시 무결성 보증 완전(로드 비용, optional) |
| code-repro | 체크포인트 저장/재개(resume)가 stage 단위 incremental flush로 견고 — 완료 stage는 JSON에서 로드 | stage3_mechanism.py:411-432 | int-2/int-3를 (group,dose) 조건 단위로 flush하면 6h 초과 중단 시 재실행 비용 절감(선택) |
| code-repro | 결과 파일 명명이 YYMMDD_ 접두사 컨벤션 준수(stage3_report_<YYMMDD>_<token_tag>_seed<N>_<githead>.json) | stage3_mechanism.py:529-533; aggregate_seeds3.py:33-44 | 없음(컨벤션 준수) |
| code-repro | 다중 시드 실행 완전 지원 — seed 0/1/2 독립 sbatch + afterok 의존성 aggregate로 mean±std, main-table 요건 충족 | submit_stage3_seeds.sh:8-14; aggregate_seeds3.py:76-120 | 없음 |
| code-repro | 의존성 고정이 부분적 — 런타임 버전은 리포트에 기록되나 sbatch가 pip 버전 pin 없이 설치(scikit-learn) | run_stage3.sbatch:29-32 | sklearn/wonderwords/nltk를 버전 핀 설치하거나 venv에 사전 설치(optional, PRIMARY DV 무영향) |
| code-repro | PPL 평가 데이터 경로가 --require-real-data를 명시적으로 재확인하지 않음(classify가 먼저 abort하므로 실질 무해, Stage2 동일) | stage3_mechanism.py:382-384; head_classifier.py:76-82 | --require-real-data일 때 data_meta 로드 직후 `assert not any(m.get('is_fallback') ...)` 추가로 이중 방어 |
| code-repro | 스모크 테스트가 venv에서 통과 — 재현성 하네스(S=1 control, spectrum 불변량, top-r 감소, disjoint groups)가 실행으로 검증 | stage3_mechanism.py:643-706 | 없음; 실모델 첫 런에서 segment_control.json의 multiseg drift가 ~0인지 육안 확인 권장 |

### Minor — alignment

| reviewer | title | file | suggestedFix |
|---|---|---|---|
| alignment | 세그먼트 상태 캐리 메커니즘이 실제 GDN2 커널·fla Cache 소스와 정합하여 가설 검정 가능 — 핵심 설계 건전 | state_surgery.py:197-251 | 추가 조치 불필요; 첫 real run의 segment_control.json에서 multiseg_origin_drift_vs_single ≈ 0(<1e-2·PPL) 반드시 육안 확인 |
| alignment | PURE GDN(gdn2_per_layer=1, 18층 전부 GatedDeltaNet2, attention/RoPE 0층)이라 세그먼트 간 위치오프셋 무관 — 구현자 주장 소스 검증됨 | /home/sohyung/long-gdn/dsc/lit_gpt/model.py:220 | 조치 불필요(검증 통과 기록) |
| alignment | 체크포인트·시드·데이터·그리드가 이전 스테이지와 CONSTANT 유지, main-table-ready 규모 — 계획 외 설정 변경 없음 | PREREGISTRATION.md | 조치 불필요 |
| alignment | 프로즈의 "16 heads" 주장이 config n_head=18과 표면 불일치(코드는 런타임 헤드수 사용하므로 정확성엔 무해) | 260726_stage2_pruning_asymmetry/head_mask.py:16 | 첫 real run에서 masker.num_heads와 classification n_heads_total(=layers*heads)을 로그 확인, 불일치 시 보고서 문구만 정정(코드 변경 불요) |
| alignment | int-2 slope 회귀가 r=0(완전 zero-out) 끝점을 포함해 "rank 사용 여부"와 "순수 pruning" 손상을 부분 혼입 | stage3_mechanism.py:294-300 | 보조 지표로 r=0 제외 slope도 함께 계산·보고해 순수 top-r 용량-사용 신호를 교차확인; 게이트 정의는 PREREG 유지, post-hoc 추가는 PREREG에 날짜부 근거 기재 |

### Minor — resources

| reviewer | title | file | suggestedFix |
|---|---|---|---|
| resources | 자원 구성이 인프라 규칙 모두 준수 — sbatch 전용, /data2 캐시, BLAS 캡, 100B 체크포인트 | run_stage3.sbatch | 현행 유지 |
| resources | OOM 위험 없음 — GPU/CPU 메모리 모두 넓은 여유 | state_surgery.py:241 | 없음; 첫 런에서 nvidia-smi 피크만 육안 확인 권장 |
| resources | 예상 실행 시간이 6h 예산 대비 3–5배 여유 — 시간 초과 위험 낮음 | run_stage3.sbatch | 없음 |
| resources | --resume 복구 단위가 intervention 전체(조건별 아님) — 중단 시 최대 한 intervention 손실 | stage3_mechanism.py:410 | run_int2/run_int3 내부 루프에서 (group,dose)마다 증분 기록·건너뛰기로 복구 입도 개선(우선순위 낮음, 런타임 짧음) |
| resources | S=1 control이 SegmentedSurgeon을 segment_len=1e9로 재-인스턴스화 + 별도 단발 forward — origin PPL 중복 계산 | stage3_mechanism.py:250 | int-2 ratio=1.0 / int-3 step=origin의 macro_ppl을 그룹 무관 단일값 1회 계산 후 복사(세그먼트 origin ~6회 재계산을 1회로 축소, 절감 소폭, 선택) |

## 자동 수정 내역

없음. 이번 리뷰에서 자동 적용된 수정은 없다(자동 수정: null). 위 발견 사항의 suggestedFix는 실행자가 수동으로 반영해야 한다.

## 문헌 대조 결과

literature reviewer의 검증 결과는 다음과 같다.

- **논문 핵심 수치·설정이 arXiv:2602.02195와 정확히 일치 — 검증 통과.** 논문 HTML을 직접 대조한 결과: (1) Eq.6 threshold rank Rank_eff=Σ I(σ_i>ε·σ_1), ε=10^-4("aligns with standard conventions in numerical linear algebra libraries e.g. PyTorch, NumPy")가 rank_metrics.py:61 및 PIN-2와 byte-for-byte 일치; (2) Eq.14 JRNP Saturation Score S_h=α(r̄/d)+(1-α)(n̄/max n̄)에서 α는 논문 미공개 → "pure-rank classification은 DECLARED DEVIATION"이 사실로 확인; (3) KV-cache 38.9% reduction이 abstract+§5.3에 verbatim 존재; (4) Table 1의 Qwen3-Next-Instruct NIAH 93.8/90.6/46.9, Thinking 90.6/62.5/43.8 모두 정확 일치. 출처: https://arxiv.org/html/2602.02195v1 (Eq.6, §5.2 Eq.14, §5.3, Table 1).
- **GDN2 readout 방향 표기 오류(minor).** vendored 커널 docstring이 상태 S∈R^{d_k×d_v}, 읽기 연산을 'o = S^T q'로 정의하나(/home/sohyung/long-gdn/dsc/lit_gpt/gdn2_ops/fused_recurrent_gdn2.py:18,23), 스펙 summary/codeChangeSummary, PREREGISTRATION.md(라인 5,62), state_surgery.py 상단 docstring(라인 5)은 모두 'o_t = S_t q_t'로 표기. 다만 수술(SVD top-r, spectrum-noise, zero)은 S 자체에 가해지고 top-r 절단과 U,V 양쪽 Haar 랜덤화가 transpose 대칭이므로 개입 수학은 동일 — 정확성 버그 아님. 정정 권고: 'o=S^T q (state S∈R^{d_k×d_v}, default layout [K,V])'로 표기 수정, 개입 결과 무영향 명시. 출처: 커널 소스 fused_recurrent_gdn2.py:23; GDN 원논문 readout 일반형 https://arxiv.org/abs/2605.22791(방향은 상태 layout 규약 의존).
- **kernel API(initial_state/output_final_state, state shape [N,HV,K,V]) 및 segmented-carry 설계가 vendored 구현과 정합 — 검증 통과.** initial_state([N,HV,K,V] fp32, 라인 432), output_final_state(라인 295,410), final_state 반환 [N,HV,K,V](라인 444) 확인; gdn2.py forward의 initial_state/output_final_state 노출과 SegmentedSurgeon의 "final_state를 surgery 후 다음 segment의 initial_state로 주입" 설계가 native 커널 semantics와 정확히 부합. 출처: /home/sohyung/long-gdn/dsc/lit_gpt/gdn2_ops/fused_recurrent_gdn2.py, gdn2.py.
- **Haar-random orthogonal 생성(Mezzadri 2007 sign-correction)이 표준 관행과 정확히 일치 — 검증 통과.** state_surgery.py:75-82 _haar_orthogonal이 Q,R=qr(Gaussian); d=sign(diag(R)); return Q*d로 구현되어 Mezzadri 2007의 sign-correction(L_ii=Sign(R_ii))과 일치, naive QR이 Haar가 아니라는 함정을 올바르게 회피. 특이값 보존(rank/nuclear/energy 불변) + 특이벡터 랜덤화(content 파괴)는 표준 spectrally-matched surrogate 관행. 출처: https://arxiv.org/pdf/math-ph/0609050 (Mezzadri 2007).
- **하이퍼파라미터가 표준 관행 범위(minor).** prune-fraction {0,...,0.389(paper),0.43(ours),...,0.60}, SVD r/cap {1.0,...,0}, ε=1e-4, θ_R=0.5, seeds 0/1/2 모두 문헌/스펙 관행 범위. segment_len=256은 FROZEN 상수로 preregister되나 grid에 단일값만 존재. 권고(선택): S=1 control 외에 segment_len∈{128,256,512} 중 최소 1개 sensitivity point를 첫 real run에서 확인해 해석의 segment_len 강건성 제시.

## 실행 전 체크리스트

- [ ] **[major]** int-1 random arm을 count-matched(round(frac*k), all_heads 또는 all_heads−high∪low에서 시드 추출)로 수정하고 PREREGISTRATION.md:54 문구를 정정. 미수정 시 int-1의 high/low 대 random 비교와 0.389/0.43 마크 표는 role 효과로 해석 불가(주판정은 불변).
- [ ] **[major]** 모델 빌더에서 실제 self.num_heads를 런타임 assert로 로깅하여 16 vs 18 확정. head_classifier 산출 헤드 총수(288 vs 324)를 Stage1/2 로그와 대조하고, 불일치 시 프로즈만 정정. num_heads/num_v_heads/head 총수를 첫 real run provenance에 명시 기록.
- [ ] 인프라 규칙(greenbeard, MEMORY run-via-slurm 준수): SLURM sbatch ONLY(`--partition main --gres=gpu:rtx6000:1`, `--time 6:00:00`), 로그인 노드 직접 CUDA 금지(cuInit=100). `submit_stage3_seeds.sh`가 seed 0/1/2 독립 sbatch --parsable + afterok aggregate로 배선됨.
- [ ] venv `/home/sohyung/sh_gdn2_venv`(torch 2.13+cu130) 활성화; export GDN2_CKPT_PATH=/home/sohyung/models/gdn2_1.3B_100b.pth(100B, 17.4GB 실재), HF_HOME=/data2/sohyung/hf_home(루트 디스크 금지), TRITON_CACHE_DIR, OMP/MKL=4.
- [ ] --require-real-data 전달 확인(synthetic fallback → INVALID); classify가 먼저 abort하므로 실질 방어되나 PPL 경로 assert 추가는 권장.
- [ ] 실행 전 head_classifier를 동일 seed로 재실행하여 교차도메인 일치 ≈ 0.97 및 Stage 2와 동일한 high/low 헤드 집합 확인(Stage 3 그룹이 설명 대상인 Stage 2 pruning-asymmetry와 일치).
- [ ] 첫 real run 직후 `results/stage3_100b_seed0/segment_control.json`에서 reproduces_origin=true 및 multiseg_origin_drift_vs_single ≈ 0(<1e-2·PPL) 육안 확인 — 값이 크면 int-2/int-3 delta 오염(fake 번들의 0.13 drift는 예상 아티팩트, 실 커널에서는 ~0 기대).
- [ ] PREREG CAVEAT verbatim 기록: 논문의 93.8/90.6/46.9/38.9%는 Qwen3-Next(48-layer, post-trained) TARGET lines이지 18-layer pure gdn2의 pass 기준이 아니며, 미달은 일반화 한계로 기록(코드 실패 아님). 개입은 사후 상태 수술(재학습 아님). NIAH origin=0.18 floor → PPL이 PRIMARY DV(UNTESTABLE_FLOOR gate origin<0.30). S-NIAH는 RULER-multikey(≠ 논문 single-needle) → within-run asymmetry만 signal. 순수-랭크 분류는 JRNP Eq.14의 DECLARED DEVIATION(α 미공개).
- [ ] G3 판정 규칙 확인: g3_seed = topr_dissoc AND spec_dissoc(int-2 slope high>low AND int-3 spectrum delta high>low, 그룹 자기-origin 대비 delta로 baseline shift 상쇄). NULL/역전은 PREREG된 한계(rank idle)로 기록하고 코드 실패로 취급하지 않음.
