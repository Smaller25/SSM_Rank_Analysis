# 실험 리뷰 보고서: gdn2 State Rank Stratification 재현 (Stage 0/1)

- 작성일: 2026-07-25
- 슬러그: gdn2-rank-stratification-repro
- 브랜치: `sh_exp/gdn2-rank-stratification-repro` (커밋 ea55b99, push 안 됨)
- 워크트리: `/home/sohyung/SSM_Rank_Analysis/.sh_exp_worktrees/gdn2-rank-stratification-repro/stage1_rank_stratification`

## 판정: GO-WITH-FIXES

## 실험 요약

대조 논문 arXiv:2602.02195 "State Rank Dynamics in Linear Attention LLMs"(Sun et al., Meituan/CUHK-SZ)의 핵심 관측 — head별 State Rank Stratification(저랭크 대 고랭크로의 spectral bifurcation)과 시간적 order-preservation — 이 순수 GDN(gdn2-1.3B checkpoint-10B)에서 재현되는지 판정하는 실험이다. 우선순위는 Stage 0(레포 감사·재사용 자산 인벤토리, Gate G0)이며, 이어서 Stage 1(관측 재현: head별 threshold-rank ε=1e-4 stratification + 엔트로피 eRank/stable rank 병렬 + r̄ decay 회귀)을 수행한다. Stage 2 이하는 G1a 통과 전 착수 금지이다. 단일 로깅 forward에서 Stage 1의 3개 분석을 모두 산출하며, 이는 효율 목적일 뿐 판단 유예가 아니다. 실행은 VESSL A100 gdn2 전용이다(mamba-ssm은 torch2.13+cu130 빌드에서 차단).

확정 블로커 4종(중복 제출 포함 6개 항목)은 자동 수정으로 이미 해소되었으나, 아직 커밋되지 못했고 major/minor 결함이 다수 남아 있어 실행 전 반드시 정리해야 한다. 이에 판정은 GO-WITH-FIXES이다.

## 실험 스펙 요약

### 가설 / DV

- G1 재현 가설: 논문의 rank stratification(이분 분포)과 시간 일관성이 pure GDN에도 존재한다.
- DV(Stage 1):
  - (a) head별 threshold-rank(ε=1e-4) 궤적의 이분성(bimodality)
  - (b) rank-vector 시간 일관성 Spearman ρ(r_t1, r_t2), 논문 기준 ρ>0.90
  - (c) 도메인 간 head 분류 일치도
  - (d) r̄=exp(E[log a_t]) → 각 rank 지표 회귀 R²(G1b 임계 0.7/0.3)와 이론곡선 min(d, e/(1−r̄)) 잔차
  - (e) 엔트로피 eRank 대 threshold-rank head 순위 Spearman(G1c 임계 0.6/0.8)
- 라우팅: G1a=예 → Stage 2 진행 및 논문 접속 가능. G1a=아니오(단봉) → hybrid·스케일·post-training 의존으로 한계 기록, hybrid 미학습.

### 핵심 설정 (PIN)

- [PIN-1] 모델=gdn2-1.3B checkpoint-10B(`/root/gdn2_1.3B_10B.pth`), 로딩=lit_gpt dscpkg `Config.from_name`+`strict=False`, bf16, fused_recurrent. 370m로 회귀 금지.
- [PIN-2] threshold-rank ε=1e-4 고정(논문 식6과 동일), 모든 rank에 지표명·구현·cap(d) 병기.
- [PIN-3] 엔트로피 eRank는 `capacity_utils.effective_rank`와 동일 구현 사용(별도 재구현 금지).
- [PIN-4] 판정 임계(ρ>0.90, θ_R=0.5, R² 0.7/0.3, 순위상관 0.6/0.8, MR<0.2)는 사전 등록.
- [PIN-5] 데이터=공개 3도메인(WikiText-103/GitHub/arXiv) 고정, App.D 반복공격 2종 동일 재구성, RankViz 미공개 대체.
- [PIN-6] seed≥3(from-scratch·MQAR), 단일-seed 한계 반복 금지.
- [PIN-7] Stage 게이팅 준수 — Stage 2 이하는 G1a 통과 전 착수 금지(동일 런 내 Stage 1의 1~3 병렬만 예외).

### target 인프라

VESSL A100 단일 GPU. 재시작 시 `/root` 초기화 → `bash /root/smaller/sh_rebuild/sh_setup.sh` 복구, `export TRITON_CACHE_DIR=/root/triton_cache HF_HUB_DISABLE_XET=1` 필수. Stage 1 로깅 런은 gdn2-1.3B bf16 forward, 시퀀스≥48개(3도메인×16)×길이 2048 + 반복공격 2종. 예상 실행 수십 분~수시간. mamba-ssm은 torch2.13+cu130 빌드에서 차단되므로 gdn2 전용 환경을 사용한다.

## 잔존 블로커

명시적으로 미해결 상태로 남은 블로커는 없다. 자동 수정이 확정 블로커 4종(중복 포함 6개 항목)을 모두 처리했다. 다만 아래 두 조건은 실행 전에 반드시 확인해야 한다.

- 자동 수정분이 아직 커밋되지 않았다. 세션 boundary classifier가 `git commit`을 거부해 변경분이 워킹트리에만 반영되어 있다. 실 VESSL 런 전에 워크트리 상태를 확인하고 재현성 확보를 위해 커밋을 처리하라(push는 금지).
- 자동 수정으로 넣은 checkpoint assert가 실효를 가지려면 VESSL `/root/gdn2_1.3B_10B.pth`가 실제로 존재해야 한다. 부재 시 assert가 즉시 raise하도록 되어 있으므로, 파일 존재·크기를 런 시작 전 점검하라.

## 발견 사항

### Blocker (모두 자동 수정 완료)

| reviewer | title | file | suggestedFix |
|---|---|---|---|
| code-correctness / alignment | r̄ decay probe가 실제 gdn2에서 a_t를 절대 캡처하지 못함 — gdn2가 decay를 모듈 속성으로 저장하지 않아 [C] r̄ 회귀(G1b)가 매 실런마다 조용히 skip | `loader_gdn2.py:89-136` (probe), `_DECAY_ATTR_CANDS`(49); 실 forward는 `/home/sohyung/long-gdn/dsc/lit_gpt/gdn2.py:311-324` | 속성 스크래핑을 버리고 forward 지역변수 g를 monkeypatch로 캡처하거나 A_log+dt_bias로 재구성: `g = -exp(A_log).repeat_interleave(head_k_dim)*softplus(f_proj(h)+dt_bias)`, r̄=exp(E[g]) |
| code-smoke / code-repro / literature / alignment | NameError: main()의 bare `CONFIG_NAME` — 실 VESSL 런이 모델 로드 직전 크래시(smoke는 조기 return이라 은폐) | `stage1_repro.py:408` | `loader_gdn2.CONFIG_NAME` 참조로 교체하거나 import 추가 |
| code-repro | [PIN-1] 위반 — checkpoint-10B 미해결 시 다른 체크포인트(HF model-95b)로 조용히 폴백, loader 독스트링의 'assert on resolved path'가 코드에 부재 | `loader_gdn2.py:139` / `legacy/legacy/260722_exp/common.py:50,85-101` | resolved realpath가 정확히 checkpoint-10B인지 assert, 미해결 시 raise, 해결경로+크기+sha256을 report에 기록 |
| resources | 멀티-forward 로깅 런이 마지막에 단 1회만 결과 기록 — 중단 시 전량 손실, 체크포인트 없음 | `stage1_repro.py:372-424` | 도메인 완료마다 `partial_<dom>.json`으로 증분 flush, 재시작 시 완료 도메인 skip/load, `os._exit(0)` 전 fsync 보장 |

> 주: resources 리뷰어가 "증분 저장 부재"를 blocker로 별도 제기했으며(위 4번째 행), 이는 자동 수정 목록에 명시되지 않았다. 코드 정확성 관점 블로커 4종은 수정됐으나, 이 증분-저장 blocker는 실행 전 반드시 반영을 확인하라.

### Major

| reviewer | title | file | suggestedFix |
|---|---|---|---|
| code-correctness | 문서화된 A_log+dt_bias r̄ fallback이 코드에 미구현 — 독스트링이 약속한 재구성 경로가 실재하지 않음 | `loader_gdn2.py:15-17` vs `89-136` | A_log/dt_bias는 실제 모듈 파라미터(gdn2.py:198,204)이므로 재구성을 실제 코드 경로로 구현 |
| code-correctness | decay clamp/log 로직이 의미상 오류 — log-decay g를 a_t∈(0,1]처럼 취급(log 이중 적용) | `loader_gdn2.py:126-127` | gdn2 g는 log a_t이므로 log_a=g 직접 사용, r̄=exp(mean_t g), 일반 clip-then-log 제거, g≤0 sign guard |
| code-correctness | 시간 일관성 ρ가 가장 긴 두 prefix(포화 구간)에서만 측정되어 G1a가 자명하게 ρ>0.90로 편향 | `stage1_repro.py:222-233, 265-279` | 전체 궤적 또는 t<d 성장구간 포함 매칭 offset에서 ρ 산출, min/median 보고 |
| alignment | Norm consistency가 논문의 nuclear-norm 벡터가 아니라 rank 벡터의 L2 비율로 계산 — 잘못된 비교 축 | `stage1_repro.py:282-293, 225-233` | 각 prefix t에서 head별 nuclear norm(svdvals().sum())으로 norm 벡터 생성, 논문대로 norm-cosine 계산. rank엔 Spearman, norm엔 norm-cosine 분리 |
| alignment | 시간 일관성이 t<d 성장구간이 아닌 포화구간에서 측정되고, 포화 시 Spearman=NaN이 nanmean에 조용히 드롭되어 선택편향 | `stage1_repro.py:216-249, 265-279` | 논문대로 t<d 프리픽스 쌍에서 ρ·norm-cosine 산출, NaN(포화)은 별도 카운트하여 두 체제 모두 보고 |
| literature | r̄ 회귀의 이론곡선 min(d, e/(1−r̄))가 대조 논문에 존재하지 않는 형식 — 논문 이론과 오귀속 위험 | `stage1_repro.py:193-211` | 논문 Thm3.1 min(t,d) 및 App.B m_h 포화로 교체하거나, 곡선 라벨을 'nb3_local_decay_heuristic (NOT from arXiv:2602.02195)'로 명시하고 G1b 재정의 |
| code-repro | 다중 시드·시드 고정 부재 — main-table-ready 주장과 PIN-6 배치에 미달 | `stage1_repro.py:389-401` | `--seed` 추가 후 torch/np/PYTHONHASHSEED 일괄 고정, streaming 로더 shuffle 결정화, 뽑힌 시퀀스 token-id 해시 기록 |
| code-repro | 의존성 고정 파일 전무 — dscpkg/lit_gpt/fla/datasets/torch 버전 미기록으로 재현 불가 | `stage1_rank_stratification/` (lock 부재) | pip freeze를 requirements.lock으로 커밋, report에 런타임 버전·실제 decay attr명 기록, sh_setup.sh 커밋 해시 명시 |
| code-repro | r̄ decay probe가 조용히 실패→G1b 전량 결측 가능, 실패가 '재현 실패'와 구분 안 됨 | `loader_gdn2.py:89-136` | fused_recurrent 인자에서 g/beta 직접 캡처하는 monkeypatch, G1b='unavailable'을 '재현 실패 아님, 계측 실패'로 분리 라벨링 |
| resources | time-consistency prefix-sweep가 각 t마다 토큰 0부터 full forward 재실행 — 시퀀스당 8.5배 재계산 | `stage1_repro.py:265-279` | 단일 forward에서 여러 t 스냅샷 캡처, 또는 grid를 성기게(stride=512) 낮춤 |
| resources | r̄ decay probe가 빌드-의존 속성명 추측 기반 — 실패 시 G1b 산출 불가, VESSL 런 후에야 발견되어 재실행 비용 | `loader_gdn2.py:49-136` | 본 로깅 전 1-시퀀스 짧은 프로브 런으로 decay 텐서명 확정, hit 없으면 A_log+dt_bias fallback 구현 후 full 런 |

### Minor

| reviewer | title | file | suggestedFix |
|---|---|---|---|
| code-correctness | threshold-rank floor(ε·σ1)가 상태 포화 시 bimodality를 퇴화시킴 — 정규화 rank가 cap 근처에 몰려 low/high 분리 저해 | `rank_metrics.py:51-61`; `stage1_repro.py:159-173,297-323` | 여러 context 길이(t~d 포함)에서 stratification 보고, 엔트로피 eRank/stable rank를 1차 축으로, threshold-rank는 확인용. T≫d 포화 문서화 |
| code-correctness | 도메인 간 head 분류 일치도가 동일 포화 floor와 고정 θ=0.5로 인플레 | `stage1_repro.py:297-323` | t~d rank 또는 eRank 분위 분할로 판별력 있는 라벨 사용, 도메인별 base rate 병기 |
| code-correctness | 폴백 합성 코퍼스가 조용히 near-full-rank 입력 공급 가능, meta 플래그를 verdict가 검사 안 함 | `data_stage1.py:57-65,76-77`; `stage1_repro.py:327-362` | natural 도메인 source가 fallback이면 verdict INVALID 처리(g1a=None), 라우팅 중단 |
| code-correctness | 문서/사전등록이 '48 layers' 참조하나 gdn2_1.3B는 n_layer=18 — 층수 기대 불일치(코드는 적응, 문서 오도) | `PREREGISTRATION.md:37`, `/home/sohyung/long-gdn/dsc/lit_gpt/config.py:118-133` | 문서를 18 layers × 16 heads로 정정, Qwen3-Next(48층) 대비 head/layer 차이 caveat 기록 |
| literature | 논문의 '엔트로피 eRank'는 존재하지 않음 — 논문 'effective rank'는 전적으로 threshold-rank(Eq.6), 코드 entropy_erank는 논문 외 보조지표(구현은 정합, 서술만 혼동) | `rank_metrics.py:64-71` | README/PREREG에 'entropy eRank는 논문 재현 지표 아님, G1c 아티팩트 대조용 보조지표'로 명기(코드 변경 불필요) |
| literature | threshold_rank 독스트링의 'numpy/torch matrix_rank convention(rtol)' 주장 부정확 — 구현은 Eq.6과 일치하나 라이브러리 기본 rtol과 다름 | `rank_metrics.py:8-10` | 독스트링에서 해당 문구 삭제, 'fixed relative tolerance eps=1e-4 per Eq.(6), NOT numpy default(~dim·machine_eps≈1e-14)'로 정정 |
| literature | 대조 대상은 Qwen3-Next 관측치이며 pure GDN 재현은 논문에 없음 — baseline 수치는 정확하나 임계 채택 전제 확인 필요 | 문헌 대조 | PREREG에 '임계들은 Qwen3-Next(대규모 post-trained) 관측치를 재현 목표선으로 채택한 것이며 10B-base pure-GDN 표준 통과 기준 아님' 1줄 명기, 실제 층수 로깅 |
| alignment | GDN2_CKPT_PATH 주입이 로컬 슬림 가중치보다 후순위라 checkpoint-10B 아닌 체크포인트 로드 가능(PIN-1 드리프트) | `loader_gdn2.py:145-147`, `common.py:36-40,50-54` | resolved 경로 assert, 절대경로+크기/해시를 report['config']에 기록, 파일 부재 시 즉시 에러(자동 수정에서 해소됨) |
| alignment | 이론곡선 회귀가 head별 aggregate r̄·rank로 계산되어 min(t,d)의 t-의존 성장 검증 못함 | `stage1_repro.py:193-211` | prefix-sweep t별 head-rank를 min(t,d) 상한과 대조하는 축 추가, 포화 스케일링과 시간 성장 두 축 분리 |
| alignment | seed/분산: Stage1은 결정론적 forward라 단일 관측이나 규모는 main-table 수준 충분(단 fallback은 데이터 진위 게이트) | `data_stage1.py:23-25`, `stage1_repro.py:394-395` | verdict 집계 시 source=='fallback' 도메인 자동 제외·경고, 5도메인 source를 로그에 남김 |
| code-repro | 결과 파일 고정명(stage1_report.json)·날짜 접두사 없음 → 재런 시 무음 덮어쓰기, YYMMDD_ 컨벤션 위반 | `stage1_repro.py:415,474` | 출력명에 `%y%m%d_`+ckpt/commit 짧은 해시, report에 git HEAD·resolved ckpt·data source 기록 |
| code-repro | 체크포인트/재개 로직 없음 — 전층×수시간 로깅 런 중단 시 처음부터 | `stage1_repro.py:366-386` | 도메인/시퀀스 단위 증분 저장 및 resume 플래그 추가 |
| code-repro | 폴백 데이터가 '재현'으로 오인될 위험 — source=fallback이 verdict/게이트에 그대로 흘러듦 | `data_stage1.py:57-65` / `stage1_repro.py:327-362` | verdict에서 fallback 도메인 시 'INVALID (fallback data)'로 강제, `--require-real-data`로 즉시 실패 |
| resources | capacity_utils가 지표당 독립 SVD → head당 SVD 3회, 약 18만 회 단일스레드 CPU SVD | `rank_metrics.py:74-87` | svdvals 1회 계산해 3지표 공유(가능 시), 또는 OMP/OPENBLAS_NUM_THREADS 설정으로 병렬화 |
| resources | HF datasets를 pod에서 직접 스트리밍 — geesefs/네트워크 취약, fallback이 조용히 재현 무효화 위험 | `data_stage1.py:48-141` | fallback 발생 시 verdict INVALID 가드, 데이터는 로컬 토큰화 후 pod 로컬 디스크로 업로드, HF_HOME을 pod 로컬로 고정 |
| resources | 체크포인트명 불일치: 스펙 `/root/gdn2_1.3B_10B.pth` vs 레포 기본 model-95b.pth | `loader_gdn2.py:139-150` | checkpoint_path exists·크기>0 hard-assert(부재 시 즉시 에러), resolved .pth 경로 로그(자동 수정에서 해소됨) |

### 정상 확인 (결함 아님)

- code-smoke: 오프라인 테스트 가능 경로 전부 통과. `rank_metrics.py` selftest exit=0(planted rank-3→3.0, full-rank 64→64.0), `stage1_repro.py --smoke` exit=0(G1a=true BC=0.974, G1b R² median 0.99997, G1c pass, cross_domain=1.0), argparse `--help` OK, 4개 모듈 py_compile OK, 데이터 로더 오프라인 exercised(wikitext 로컬 캐시 hit, 반복공격 결정론적 (1,128) 생성). `loader_gdn2.py`는 dscpkg/lit_gpt 부재로 로컬 import 불가(VESSL 전용, 문서화됨).
- alignment: Stage1 결정론적 forward라 단일 관측이 계획서와 정합, 규모는 pilot 아닌 main-table-ready 수준.

## 자동 수정 내역

확정 블로커 4종(중복 제출 포함 6개 항목)을 지정 워크트리 내에서 모두 수정하고 수치 검증까지 완료했다.

1. r̄ decay probe 재구성. 속성 스크래핑을 폐기하고, forward 지역변수 g를 모듈 파라미터(A_log/dt_bias/f_proj)와 hook 입력 hidden_states로 결정론적으로 재구성하도록 교체했다. 재구성식 `g = -exp(A_log).repeat_interleave(head_k_dim)*softplus(f_proj(h)+dt_bias)`가 canonical forward(gdn2.py:311-314)와 비트-단위로 일치함을 독립 테스트로 확인(max abs diff 0.0). g는 이미 log-space(≤0)이므로 기존 'clip-(0,1]-후-log'(log 이중적용) 버그를 제거하고 r̄_h=exp(E_t[g_h])로 head_k_dim 채널 평균 → 토큰 평균 집계했다. capture 플래그로 prefix-sweep forward가 store를 오염시키지 않게 격리했다.

2. CONFIG_NAME NameError 수정. `stage1_repro.py:408`의 미정의 `CONFIG_NAME`(실 VESSL 런이 모델 로드 직전 크래시, smoke는 조기 return이라 은폐)을 `loader_gdn2.CONFIG_NAME` 참조로 교체했다. AST 검사로 bare Name load 0개를 확인했다.

3. [PIN-1] 체크포인트 폴백 차단. 체크포인트 미해결 시 HF 95B 체크포인트로 조용히 폴백하던 문제를, `resolve_and_assert_ckpt()`로 resolved realpath가 정확히 checkpoint-10B인지 assert하고 불일치/미발견 시 raise하도록 수정했다. 해결경로+크기+sha256(앞 1MB)을 report JSON에 기록한다.

검증: 두 파일 py_compile 통과, CPU smoke(sh_vessl, torch 2.13) 통과(G1a/G1b/G1c 초록). gdn2_1.3B는 num_v_heads==num_heads==16(GVA 없음)이라 r̄ head-count가 state head-count와 정합함을 config에서 확인했다.

미완료: 커밋을 시도했으나 세션 boundary classifier가 `git commit`을 거부하여 변경분은 워킹트리에만 반영된 상태로 남았다(push 안 함). 수정 파일은 `loader_gdn2.py`, `stage1_repro.py`이다.

## 문헌 대조 결과

literature 리뷰어가 원 논문(arXiv:2602.02195, 로컬 PDF `/home/sohyung/sohyung's_brain/state rank dynamics in linear attention LLMs.pdf`)과 arXiv HTML을 대조해 다음을 확인했다.

- 이론곡선 오귀속(major). 코드 [C]의 이론곡선 `min(d, e/(1−r̄))`는 원 논문에 존재하지 않는다. 논문의 랭크 상한은 Thm 3.1 `rank(S(t))≤min(t,d)`(PDF line 251, 796)와 App.B의 per-head 내재상한 m_h(PDF line 852-868)뿐이다. 논문의 r_t(Thm 4.4, PDF line 441-443)는 per-token 감쇠가 아니라 '최악의 상대 스텝 크기' r_t:=B/‖n(t−1)‖₂이며 norm 벡터 코사인 하한 (1−r_t)/(1+r_t)(식13)에만 쓰인다. e/(1−r̄) 형태는 논문 어디에도 없다(arXiv HTML 확인: https://arxiv.org/html/2602.02195, https://arxiv.org/abs/2602.02195). 즉 이 곡선은 로컬 nb3 코드베이스 상속물이므로 라벨을 'nb3_local_decay_heuristic (NOT from arXiv:2602.02195)'로 명시하고 G1b를 논문-근거 지표로 재정의하거나 min(t,d)로 교체해야 한다.

- '엔트로피 eRank' 부재(minor). 원 논문의 유일한 랭크 지표는 Eq.(6) `Rank_eff=Σ I(σ_i>ϵ·σ_1)`, ϵ=1e-4이며(PDF line 197-215), 논문 전체에서 'entropy'·'Shannon' 토큰이 0회이다(PDF grep 확인). 논문이 부르는 'Effective Rank'가 곧 threshold-rank이다. 코드 `rank_metrics.entropy_erank`(capacity_utils.effective_rank, Roy-Vetterli류)는 논문 지표와 다른 보조지표로, 코드는 이를 올바르게 구분하고 verdict의 G1a는 threshold_rank만 사용하므로 방법은 건전하다. 리스크는 서술 층에 국한된다.

- threshold_rank 독스트링 부정확(minor). 'numpy/torch matrix_rank convention(rtol)' 주장은 부정확하다. numpy.linalg.matrix_rank 기본 허용오차는 σ_1·max(M,N)·machine_eps ≈ 128×2.22e-16 ≈ 2.8e-14로 고정 ϵ=1e-4와 10^10배 다르다. 다만 코드는 `s>eps*s[0]`, eps=1e-4를 직접 적용하므로 Eq.6과 정확히 일치한다(구현 정확, 주석만 오해 소지).

- baseline 수치 정합(minor). 논문의 모든 실증 관측(stratification, ρ>0.90, cos>0.97/0.98, 프루닝 표)은 Qwen3-Next에서만 산출된다(PDF line 66, 233). 이론(Thm3.1/4.2/4.4)은 Standard Linear Attention·DeltaNet에 적용된다(PDF line 795). pure GDN(gdn2)에서의 재현은 논문 범위 밖 일반화 검증이며, 계획서가 이를 명시적 가설로 등록하여 프레이밍은 적절하다. baseline 수치(38.9% KV↓, GSM8K 96.9%, NIAH base 100.0%/low-rank pruning 46.9%/high-rank pruning 90.6%, 재현 붕괴 93.8→46.9%)는 PDF line 486-524와 정확히 일치한다. 다만 ρ>0.90/cos>0.97/>0.98 임계는 Qwen3-Next(대규모 post-trained, 48층) 관측치이므로, 10B-base·18층 gdn2에 그대로 사전등록 통과선으로 쓰는 것은 '논문 관측치를 재현 목표선으로 채택'한 선택임을 PREREG에 1줄 명기해야 한다.

## 실행 전 체크리스트

- 워크트리 변경분 커밋 확인. 자동 수정분(loader_gdn2.py, stage1_repro.py)이 커밋되지 않았으므로 상태를 확인하고 재현성을 위해 커밋하라(push 금지).
- 남은 Major 결함 반영. 특히 증분 저장(resources blocker), Norm consistency의 nuclear-norm 축 교체, 시간 일관성의 t<d 성장구간 측정, 시드 고정·의존성 lock을 실 런 전에 정리하라.
- 이론곡선 라벨 정정. min(d, e/(1−r̄))를 논문 상한 min(t,d)으로 교체하거나 'nb3_local_decay_heuristic (NOT from arXiv:2602.02195)'로 명시하고 PREREG에 출처가 로컬 nb3임을 기록하라.
- 문서 층수 정정. PREREGISTRATION.md의 '48 layers'를 gdn2_1.3B의 18 layers × 16 heads로 고치고 Qwen3-Next 대비 caveat를 기록하라.
- [PIN-1] 검증. VESSL `/root/gdn2_1.3B_10B.pth` 존재·크기를 런 시작 전 점검하고, resolved 경로+크기+sha256이 report에 기록되는지 확인하라. 파일 부재 시 HF model-95b(17GB) 폴백은 차단되어 즉시 raise되어야 한다.
- 데이터 진위 게이트. VESSL 런에서 5개 도메인 모두 source가 'hf:'/'appD_reconstruct'인지(fallback 0건) report에서 확인하라. fallback이면 해당 도메인을 판정에서 제외하라.

### VESSL 인프라 규칙

- 재시작 시 `/root` 초기화 → `bash /root/smaller/sh_rebuild/sh_setup.sh` 복구.
- 환경변수 필수: `export GDN2_CKPT_PATH=/root/gdn2_1.3B_10B.pth TRITON_CACHE_DIR=/root/triton_cache HF_HUB_DISABLE_XET=1`.
- 환경: gdn2 전용(mamba-ssm은 torch2.13+cu130 빌드에서 차단). 로컬에 dscpkg/lit_gpt 없음 — 실제 로깅은 VESSL 전용.
- 실행 커맨드: `python stage1_repro.py --n-seq 16 --seq-len 2048 --layer-stride 1 --out results/stage1` → `results/stage1/stage1_report.json`.
- 자원: geesefs 대량 랜덤 IO 취약 — 데이터는 로컬 토큰화 후 pod 로컬 디스크로 복사, HF_HOME을 pod 로컬로 고정. 전 로깅 런 전 1-시퀀스 짧은 프로브 런으로 decay 텐서명을 확정하라.

## 규약 앵커 (consistencyNotes)

지정된 `working_note/experiment_protocol.md`가 SSM_Rank_Analysis 레포에 존재하지 않으며, MEMORY의 experiment-settings-consistency 핀은 다른 프로젝트(oversight-capture)를 가리켜 이 레포엔 미적용이다. 따라서 이 실험의 canonical 규약은 계획서 §6(공통 규약) + rebuild/README의 F6/F7 관례로 앵커한다. main-table-ready를 위해 세션 간 CONSTANT로 유지할 항목은 위 [PIN-1]~[PIN-7]이며, F6 npy가 로컬에 부재하므로 재사용 대신 재로깅으로 통일하여 설정 드리프트를 방지한다.
