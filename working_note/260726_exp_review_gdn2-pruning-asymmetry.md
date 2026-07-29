# Stage 2 실험 리뷰: GDN2 Pruning Asymmetry (저랭크 vs 고랭크 헤드 절제)

- 작성일: 2026-07-26
- slug: gdn2-pruning-asymmetry
- 브랜치: `sh_exp/gdn2-pruning-asymmetry` (커밋됨, push 안 됨)
- 워크트리: `/home/sohyung/SSM_Rank_Analysis/.sh_exp_worktrees/gdn2-pruning-asymmetry/260726_stage2_pruning_asymmetry`
- 대상 인프라: greenbeard (SLURM sbatch)

## 판정: GO-WITH-FIXES

세 개의 blocker(캐시 오염, fresh-vs-resume 분기, NIAH KV 캐시 부재로 인한 6h 벽 위험)는 Stage 2 코드 경로 내부에서 자동 수정되어 잔존 블로커는 없다. 다만 다중 시드 집계 스크립트 부재, NIAH 런타임 의존성 미고정, 저랭크/고랭크 헤드 집합 겹침 위험, origin NIAH floor 헤드룸 게이트 부재 등 major 발견들이 남아 있어, 전체 SLURM 실행 전에 반드시 해결(또는 사전등록된 방식으로 명시)해야 한다.

## 실험 스펙 요약

### 가설 / DV
- 가설 H (G2): 저랭크 헤드 그룹을 마스킹하면 S-NIAH 검색 정확도와 도메인별 PPL이 동일 크기의 고랭크 헤드 그룹 마스킹보다 실질적으로 더 크게 열화하며, 이 저-vs-고 격차가 동일 개수 랜덤 마스킹 대조군을 초과한다(손상이 헤드 '개수'가 아니라 '어느 헤드=저랭크'에 특정됨).
- 논문 방향 예측(arXiv:2602.02195): 저랭크 절제는 NIAH를 붕괴(93.8→46.9), 고랭크 절제는 거의 불변(93.8→90.6).
- 주 DV: needle 깊이/위치별 S-NIAH 검색 정확도(needle 값 정확 방출), 조건별.
- 부 DV: 도메인별 PPL(wikitext/github/arxiv), 조건별.
- 효과 지표: delta(low-rank mask) − delta(high-rank mask), 그리고 둘 다 vs delta(random mask), ≥3 시드 mean±std.
- Stage 1 뉘앙스: 고랭크 헤드 == 저감쇠(고보존) 헤드(G1b R²≈0.90). 따라서 '저랭크 절제가 검색을 해친다'는 것은 빠른감쇠/저메모리 헤드가 needle 검색의 load-bearing이라는, 다소 반직관적 예측이다.

### 핵심 설정 (PIN)
- [PIN-1] model=gdn2-1.3B, `Config.from_name("gdn2_1.3B")`, `load_state_dict(strict=False)`, bf16, mode=`fused_recurrent`. 체크포인트=100B paper-matched `/home/sohyung/models/gdn2_1.3B_100b.pth` (17.4GB, sha256[:16]=4b03319f; 95B 준-등가 허용, 10B는 `resolve_and_assert_ckpt`가 하드 리젝). 아키텍처 18층 × 16헤드 = 288헤드, num_v_heads==num_heads==16 (GVA 없음).
- [PIN-2] 헤드 랭크 지표 = threshold-rank eps=1e-4, Rank_eff = Σ I(σ_i > eps·σ_1) (Eq.6), cap d=min(dk,dv). 분류 임계 θ_R=0.5 (정규화 랭크 rank/cap), Stage 1 `cross_domain_agreement()`와 동일.
- 헤드 분류는 Stage 1과 동일한 per-head 최종 상태(loader_gdn2 .states, data_cache/ 자연 3도메인 aggregate)에서 산출. Stage 1 cross-domain agreement는 0.971이었음. 마스킹 헤드 수 k는 high/low/random 세 조건에서 동일해야 함.
- 마스킹 = gdn2.py ~391행, o_norm 직후·rearrange/o_proj 이전의 mixer 출력 o(b,t,h,d)를 per-head zeroing (forward hook, (layer_idx, head_idx) 키). origin = 빈 마스크.
- 데이터 = data_cache/ 실제 텍스트(wikitext-103/codeparrot-clean-valid/ccdv-arxiv), `--require-real-data`(폴백 시 INVALID).
- S-NIAH = niah_ruler.make_mk_niah(RULER multikey) + 신규 검색 스코어링(greedy ~128토큰 생성 후 needle 값 매칭). PPL = analysis.token_nll_bits.
- [PIN-6] seed≥3 (0,1,2), torch/numpy/PYTHONHASHSEED 고정, aggregate mean±std 보고. MATH/GSM8K 제외(base 모델 floor, plan §2 이탈).

### 게이트 G2
저랭크 절제 손실이 고랭크 절제 손실보다 훨씬 크고 랜덤 대조를 초과하면 pure GDN에서 기능적 asymmetry가 존재 → Stage 3 판정 적용. 재현 실패 시 hybrid/post-training 특이 가능성을 기록하고 framing 재결정.

### 사전등록 caveat (verbatim)
논문의 93.8/46.9/90.6/38.9% 수치는 Qwen3-Next(48층, post-trained) 관측치를 재현 TARGET 라인으로 채택한 것이지, 18층 gdn2의 pass 기준이 아니다. 미달 시 = 기록해야 할 일반화 한계이지 코드 실패가 아니다.

## 잔존 블로커

없음. 아래 '자동 수정 내역'의 세 blocker가 모두 해소되었다. 단, 자동 수정 커밋이 auto-mode 분류기에 의해 차단되어 변경분은 워킹트리에 적용된 채(스테이지 해제) 남아 있으므로, **사용자가 직접 커밋해야 한다.**

## 발견 사항

### Blocker (자동 수정 완료)

| reviewer | title | file | suggestedFix |
|---|---|---|---|
| code-correctness | 분류 단계 `bundle.states()` 이후 `SHARED["cache"]`가 None으로 리셋되지 않아 오염된 recurrent-state가 모든 NIAH/PPL forward를 오염 → 전 조건 DV 무효화 | common.py:62,69,108-109; stage2_pruning.py:144,90-93 | logits 평가 전 shared cache를 None으로 리셋(Bundle.logits 진입부 또는 states finally). 오염된 기존 결과는 무효 → 재실행 |
| code-correctness | fresh-run vs `--resume`가 서로 다른 logits 산출 — 캐시 오염이 in-process 분류 실행 시에만 발동해 resume 플래그에 따라 결과가 조용히 분기(재현성/일관성 붕괴) | stage2_pruning.py:139-146; run_stage2.sbatch:40 | 근본 캐시 리셋을 고쳐 states 실행 여부와 무관하게 logits가 항상 clean. --resume에 버그 은폐 의존 금지 |
| resources | NIAH greedy_generate에 KV 캐시 없음: 매 디코드 스텝마다 ~2048토큰 전체 re-forward → ~124배 낭비, 3시드 직렬 실행 시 6h 벽 초과 위험 | niah_retrieval.py:40-68 | fla/gdn2 recurrent 상태 캐시로 증분 디코드, 또는 (a) gen 토큰 128→~32 축소 (b) 시드별 독립 sbatch job 제출. 단일 조건으로 재-타이밍 후 --time 보정 |

### Major

| reviewer | title | file | suggestedFix |
|---|---|---|---|
| code-repro / alignment | 다중 시드 교차 집계(mean±std) 스크립트 부재 — G2 게이트가 '≥3시드 aggregate'로 정의되나 시드별 판정만 산출, 최종 판정을 수기 의존 | stage2_pruning.py:101-129,268; run_stage2.sbatch:32-42 | seed0/1/2 리포트를 읽어 조건별 NIAH acc·macro PPL과 delta_low/high/random을 mean±std로 집계하고 최종 G2를 산출하는 `aggregate_seeds.py`를 추가, sbatch 말미에서 호출 |
| code-repro | NIAH 런타임 의존성(wonderwords, nltk punkt_tab, baber/paul_graham_essays)이 requirements.lock에 미고정, sbatch에서 무버전 ad-hoc 설치 → 주 DV가 비재현적 | run_stage2.sbatch:22-23; niah_ruler.py:62-65 | wonderwords==/nltk==/datasets==5.0.0/punkt_tab 버전을 lock에 고정하고 실제 설치 버전을 provenance로 기록. 캐시 워밍 후 sbatch에 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 설정 |
| literature | 마스킹이 헤드 OUTPUT(o_norm readout)을 zeroing할 뿐 recurrent KV-state를 줄이지 않음 — 논문의 실제 pruning(KV-state 감소, '38.9% KV 절감')과 상이 | head_mask.py:62-75 | (a) 이를 OUTPUT-ablation asymmetry probe로 명시하고 KV-38.9% 라인을 out-of-scope로 제외, 또는 (b) per-head recurrent_state를 실제로 zeroing하는 state-prune을 추가 |
| literature | 헤드 분류가 순수 rank threshold(θ_R=0.5)만 사용 — 논문의 pruning 타깃은 rank AND nuclear norm 결합 Saturation Score(Eq.14, weight α) | head_classifier.py:105-113 | 순수-rank 분할을 JRNP Eq.14로부터의 의도적 이탈로 명시. nuclear_norm이 이미 rank_metrics에 있으므로 Saturation-Score 변형(α sweep)을 robustness arm으로 선택 추가 |
| alignment | low_heads(bottom-k)와 high_heads(top-k)가 겹칠 수 있어 저랭크 vs 고랭크 대조축 오염 위험 — low_frac>0.5 시 order[:k]와 order[-k:]가 중복, driver의 `assert len==k`는 겹침을 못 잡음 | head_classifier.py:110-113 | count-match 후 `assert set(low).isdisjoint(set(high))` 추가. 겹치면 k를 min(...,n//2)로 축소하거나 median/분위수 분할로 정확히 disjoint. 리포트에 low_frac/high_frac 및 low∩high 명시. Stage1 리뷰 경고(T>>d 포화 시 분리 퇴화)와 정합 |
| alignment | origin NIAH가 base 모델 floor에 걸리면 asymmetry 측정 불가 — 헤드룸 게이트 부재, 바닥 노이즈에서 우연히 low>high가 나오면 오판정 | stage2_pruning.py:101-129 | compute_verdict에 origin NIAH 헤드룸 게이트 추가(예: origin<0.3이면 'UNTESTABLE_FLOOR' 보류). PPL을 폴백 주 판정으로 승격하는 경로 사전등록. seed0 origin 스모크로 헤드룸 선확인 |

### Minor

| reviewer | title | file | suggestedFix |
|---|---|---|---|
| code-correctness | NIAH 프롬프트 길이는 add_special_tokens=default(True)로 측정되나 생성은 False로 토큰화 → 깊이 off-by-a-few 및 seq_len 초과 여지 | niah_retrieval.py:48,94-95; niah_ruler.py:150,167 | 양 호출부의 add_special_tokens를 일치(권장 False)시켜 budget 길이와 평가 길이 일치 |
| code-correctness | needle-depth 국소화가 query key의 첫 re.search 매치에 바인딩 — needle 이외 occurrence(질문 tail/haystack 충돌)에 오결합 가능(부 DV만 영향) | niah_retrieval.py:91-96 | 전체 needle 문장 문자열로 위치 탐색하고 질문 tail을 제외한 context 영역 내에서만 검색 |
| code-repro | head_mask.py가 common을 import하나 260722_exp를 sys.path에 명시 추가 안 함 — driver 경유로만 동작, 단독 import 시 ModuleNotFoundError | head_mask.py:22-25,44 | head_mask.py의 sys.path 부트스트랩에 260722_exp 경로를 명시 추가해 자기완결성 확보 |
| code-repro | 체크포인트 해석이 --ckpt 인자보다 GDN2_CKPT_PATH/파일탐색에 override될 수 있어 provenance 혼동 여지(10B는 하드 리젝, provenance는 기록됨) | loader_gdn2.py:225-231; stage2_pruning.py:262,274 | resolve 후 prov['realpath']가 args.ckpt realpath와 일치(또는 token_tag가 100b/95b)하는지 driver에서 assert, 불일치 시 abort |
| literature | eval이 RULER multikey(num_needle_k=4)이나 논문 Table 1 NIAH는 single-needle gkamradt — 절대 정확도 스케일이 93.8/46.9/90.6과 직접 비교 불가(내부 asymmetry는 유효) | niah_retrieval.py:99-100 | S-NIAH가 RULER-multikey(더 어렵고 스케일 상이)임을 명시하고, 절대값이 아닌 WITHIN-run low-vs-high-vs-random asymmetry만 G2 신호임을 기재 |
| literature | Table 1 target 라인(93.8/90.6/46.9, KV 38.9%)이 코드에 정확히 전사됨(Qwen3-Next-Instruct row). Thinking row(90.6→62.5→43.8)에선 고랭크 절제도 하락 | (확인) | 선택: Thinking-row caveat 기재 — 고랭크 절제가 항상 무시가능한 것은 아님, base 모델 결과가 두 row 사이여도 논문과 정합 |
| alignment | S-NIAH haystack(paul_graham_essays)이 --require-real-data 보호 밖 — 오프라인 노드에서 로드 실패 시 시드 중단(GPU 시간 낭비) | niah_ruler.py (get_haystack essay 분기) | driver 시작 시 get_haystack('essay') 1회 pre-flight 체크, 실패 시 GPU 점유 전 즉시 종료 또는 NIAH skip·PPL 계속 분기. HF_HUB_OFFLINE·캐시 경로 로그 |
| resources | HF_HOME=/data2/sohyung/hf_home가 sbatch에 미export — .bashrc 의존(비로그인 sbatch는 미source 가능), essay가 ROOT-disk 캐시에도 워밍됨(인프라 규칙 위반) | run_stage2.sbatch:12 | sbatch에 `export HF_HOME=/data2/sohyung/hf_home`(및 HF_DATASETS_CACHE) 추가, /home/sohyung/.cache의 중복 paul_graham 사본 제거 |
| resources | 결과 디렉토리가 /data2가 아닌 root-disk 워크트리 내부(현재 JSON만이라 허용, 규칙상 주의) | run_stage2.sbatch:34 | JSON 출력은 현행 유지. per-token logits/state tensor로 확장 시 --out을 /data2로 변경. GPU 메모리는 96GB에서 여유(OOM 없음) |

### 긍정 확인 (강점)

| reviewer | 확인 내용 |
|---|---|
| code-smoke | CPU 스모크 종단 통과(EXIT=0): classify k=16, agreement=1.000, count-match clean, niah_asymmetry=True, PPL finite. 5개 모듈 전부 byte-compile, 모든 Stage1/260722 API 심볼 resolve. |
| code-smoke | 마스킹 훅 사이트가 스펙 위치와 정확 일치(gdn2.py:391 o_norm 출력 (b,t,h,d), rearrange/o_proj 이전). layer_idx는 common.load_model이 설정. |
| code-smoke | sbatch가 `--test-only`로 검증 통과(Job to start on gpu01 in partition main), shell syntax OK, 로그 디렉토리 존재. 로그인 노드 직접 CUDA 미실행. |
| literature | threshold_rank eps=1e-4가 논문 Eq.(6)과 byte-for-byte 일치. cap d=min(dk,dv) 정규화가 논문 saturation framing과 정합. |
| literature | 논문에 random-pruning 대조가 없음 — 하네스의 count-matched 랜덤 arm은 방법론적으로 더 강한(우월한) 원저 대조. 'which heads vs how many'를 분리. |
| alignment | 시드 고정(PYTHONHASHSEED/random/numpy/torch/cuda), 조건별 incremental flush + --resume, args/git_head/runtime_versions/ckpt_provenance/timestamp/prereg_caveat 전부 기록. 명명 `stage2_report_<YYMMDD>_<token_tag>_seed<N>_<githead>.json` (YYMMDD 접두 준수). |

## 자동 수정 내역

세 blocker 모두 `260726_stage2_pruning_asymmetry` 경로 안에서만 수정해 해결했다.

- 캐시 오염(blocker 1) + fresh-vs-resume 분기(blocker 2): 분류 단계 `bundle.states()`가 남긴 오염된 recurrent-state 캐시를 매 logits 평가 전에 None으로 리셋. `common.py`/`loader_gdn2.py`(허용 경로 밖)는 손대지 않고, Stage 2 코드에서 `bundle.base.shared`를 통해 리셋하는 방식으로 우회. 이로써 NIAH/PPL 오염과 resume 분기를 동시에 제거.
- NIAH 6h 벽 위험(blocker 3): 시드당 독립 sbatch job(신규 `submit_stage2_seeds.sh`) + gen_tokens 128→48로 완화.
- CONSTANT(PIN-1/2/5/6: 모델/체크포인트/rank-metric/도메인/시드+k-count-match)는 변경하지 않음.
- CPU smoke 및 캐시-리셋 헬퍼 단위 테스트 통과.
- 변경 파일(절대경로): `/home/sohyung/SSM_Rank_Analysis/.sh_exp_worktrees/gdn2-pruning-asymmetry/260726_stage2_pruning_asymmetry/{stage2_pruning.py, niah_retrieval.py, run_stage2.sbatch}` + 신규 `submit_stage2_seeds.sh`.
- 주의: 사용자 지시대로 커밋을 시도했으나 auto-mode 분류기가 커밋을 차단 → 변경분은 워킹트리에 적용된 채(스테이지 해제) 남아 있음. **사용자가 직접 커밋 필요.**

## 문헌 대조 결과

primary 대조 논문: arXiv:2602.02195 "State Rank Dynamics in Linear Attention LLMs" (Sun et al., Meituan/CUHK-SZ) §5 pruning + Table 1. 로컬 PDF `/home/sohyung/sohyung's_brain/state rank dynamics in linear attention LLMs.pdf`, HTML https://arxiv.org/html/2602.02195v1.

- rank 정의 일치: `rank_metrics.threshold_rank` = Σ_i I(σ_i > eps·σ_1), eps=1e-4가 논문 Eq.(6) Rank_eff와 byte-for-byte 일치(PDF page 2, §2.3). 논문도 "PyTorch/NumPy 표준 관례와 정렬"이라 명시. classification substrate는 faithful.
- pruning 연산 상이(major): 논문 §5.3은 High-Rank 헤드 제거가 "KV-state memory footprint와 recurrent update computation의 직접 감소"를 낳고 "38.9% KV-cache overhead 감소"를 보고(PDF page 5). 하네스는 o_norm 출력만 zeroing(head_mask.py:62-75)하므로 KV/메모리 절감이 0 → KV-38.9% 라인은 이 코드 경로로 재현 불가. OUTPUT-ablation은 '이 헤드가 load-bearing인가'의 defensible proxy이나 논문의 pruning 연산은 아님.
- 선택 기준 상이(major): 논문 §5.2 JRNP는 Saturation Score S_h = α·(r̄_h/d) + (1−α)·(n̄_h/max_j n̄_j) (Eq.14)로 rank AND nuclear norm 결합. 구현은 norm 항을 완전히 제거하고 순수 rank로 분할. α는 논문 미공개(WebFetch로 확인)라 정확 JRNP는 재현 불가지만, 순수-rank 분할은 명명된 단순화로서 '논문과 일치'가 아니라 '이탈'로 기재해야 함.
- eval 벤치 상이(minor): 논문 Table 1 NIAH는 single-needle gkamradt NIAH(PDF page 5), 하네스는 RULER multikey(num_needle_k=4). 절대 정확도 스케일이 다르므로 93.8% 등과 절대값 비교 금지. 내부 low-vs-high-vs-random asymmetry는 유효.
- target 라인 전사 정확(minor): stage2_pruning.py:270-273의 origin 93.8/high 90.6/low 46.9/KV 38.9%가 Qwen3-Next-Instruct row와 일치(PDF page 5, Table 1). 48층도 확인(PDF page 3). Thinking row(90.6→62.5→43.8)에선 고랭크 절제도 상당한 하락 — 'high-rank barely moves' framing은 Instruct-specific. 방향성 asymmetry(low >> high)는 양 row 모두 성립.
- random 대조 부재 확인(minor, 강점): WebFetch로 논문에 random-pruning baseline 없음 확인. 하네스의 count-matched 랜덤 arm은 논문이 남긴 gap을 닫는 원저·정당한 대조.

eval harness lineage: NVIDIA RULER S-NIAH / niah_multikey_1 (Apache-2.0, lm-evaluation-harness), legacy/260722_exp/niah_ruler.py에 verbatim 복제. GatedDeltaNet2 아키텍처: arXiv:2605.22791. 그라운드-트루스 마스킹 사이트: `/home/sohyung/long-gdn/dsc/lit_gpt/gdn2.py` forward.

## 실행 전 체크리스트

인프라 규칙 (greenbeard):
- [ ] SLURM sbatch로만 제출(`--partition main --gres=gpu:rtx6000:1`). 로그인 노드 직접 CUDA 금지(cuInit=100). gpu01 = RTX PRO 6000 Blackwell 96GB.
- [ ] venv `/home/sohyung/sh_gdn2_venv` (torch 2.13+cu130). mamba-ssm 불필요.
- [ ] `GDN2_CKPT_PATH=/home/sohyung/models/gdn2_1.3B_100b.pth`(100b) export, `resolve_and_assert_ckpt`로 해석. 10B 리젝 확인.
- [ ] `TRITON_CACHE_DIR=/home/sohyung/.triton_cache`, `HF_HUB_DISABLE_XET=1`. OMP/OPENBLAS/MKL 스레드 4로 cap.
- [ ] `--time 6:00:00`. 시드별 독립 job(자동 수정된 submit_stage2_seeds.sh)으로 각 시드에 신선한 6h 예산.
- [ ] `export HF_HOME=/data2/sohyung/hf_home`를 sbatch에 추가(현재 미export, major). root-disk 중복 essay 캐시 제거.
- [ ] 캐시 워밍 후 `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1` 설정, get_haystack('essay') pre-flight 체크.

실행 전 코드 수정 (반드시):
- [ ] 자동 수정된 세 blocker 변경분을 사용자가 직접 **커밋**(auto-mode가 차단함).
- [ ] `aggregate_seeds.py` 추가: seed0/1/2 리포트를 mean±std로 집계, delta_low/high/random 시드평균, 최종 G2 산출. sbatch 말미 호출.
- [ ] NIAH 런타임 의존성(wonderwords/nltk/datasets/punkt_tab) 버전 lock 고정 + provenance 기록.
- [ ] `head_classifier.py`에 `assert set(low).isdisjoint(set(high))` 및 k>0/최소 그룹크기 게이트 추가. low_frac/high_frac/low∩high를 리포트에 기록.
- [ ] `compute_verdict`에 origin NIAH 헤드룸 게이트('UNTESTABLE_FLOOR') + PPL 폴백 사전등록. seed0 origin 스모크로 헤드룸 선확인.

리포트/재현성:
- [ ] provenance(resolved ckpt realpath/size/sha, git HEAD, runtime_versions에 wonderwords/nltk 추가) 기록.
- [ ] 사전등록 caveat verbatim echo: 93.8/46.9/90.6/38.9%는 Qwen3-Next(48층, post-trained) TARGET 라인이지 18층 gdn2의 pass 기준이 아님. 미달 = 일반화 한계 기록.
- [ ] 마스킹이 OUTPUT-ablation(KV pruning 아님)임을 명시, KV-38.9% 라인 out-of-scope 또는 state-prune arm 추가. 순수-rank 분할이 JRNP Eq.14로부터의 이탈임을 기재. S-NIAH가 RULER-multikey임을 명시.
