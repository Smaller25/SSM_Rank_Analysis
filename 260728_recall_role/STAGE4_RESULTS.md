# Stage 4 — high-rank head는 in-context RECALL 유닛인가 (induction probe, pure GDN)

대조 논문 **arXiv:2602.02195**은 high-rank head를 "oversaturated ⟹ prunable(잉여)"로 읽고, 역할
차원에선 **low-rank가 retrieval-indispensable**하다고 함의한다. Stage 4는 pure GDN(gdn2-1.3B,
100B paper-matched)에서 **역할(role)** 차원만 떼어내 검증한다: induction(반복 복사) probe로
high/low/random head를 각각 KV-state pruning 하고, **recall 구간(둘째 복사)** 의 induction_gain이
어느 조건에서 붕괴하는지를 본다. hybrid는 이번 범위에서 제외(pure만).

## 세팅
- 모델 gdn2-1.3B model-100b.pth(해시 6cb3072), 로컬 SLURM(rtx6000), **3 seed(0/1/2)**, 실 pretrained.
- probe = Olsson(2022) 스타일 induction. 실 텍스트 passage(len 1024)를 [A][A]로 이어 붙여 seq 2048.
  - `local_bits` = 첫 복사(문맥 없음) 비트, `recall_bits` = 둘째 복사(앞을 베끼면 낮아짐) 비트.
  - `induction_gain = local_bits − recall_bits` (클수록 in-context 복사 능력 강함).
- 4조건: origin / high-rank prune / low-rank prune / random prune. **동수(124/288 head, KV 43.1% 절감)**,
  low·high는 disjoint, count-matched.
- 판정(사전등록):
  - **headroom 게이트**: origin induction_gain > 0.30 bits(다수 seed) 아니면 UNTESTABLE.
  - **(a) 필요**: Δrecall(high) > Δlocal(high) + 0.10 bits (high pruning이 local보다 recall을 더 해침).
  - **(b) 주지표**: Δrecall(high) > Δrecall(low)+0.10 AND > Δrecall(random)+0.10 (high-specific).
  - **seed 부호 일관**: 각 seed에서 high가 두 대조군을 모두 이겨야(다수결).

## 결과 (3-seed mean±std, 보수적)

| 조건 | induction_gain | Δrecall vs origin | recall_collapse_frac |
|---|---|---|---|
| origin | **2.380 ± 0.022** | — | — |
| **high prune** | **−0.006 ± 0.002** | 7.529 ± 0.025 | **3.16×** |
| low prune | 3.042 ± 0.048 | 0.926 ± 0.053 | 0.39× |
| random prune | 1.353 ± 0.178 | 2.453 ± 0.201 | 1.03× |

- **high pruning이 in-context recall을 소멸**시킨다: induction_gain 2.38 → **−0.006**(사실상 0). 즉
  high-rank head를 지우면 모델이 방금 본 passage를 더 이상 베끼지 못한다.
- **low pruning은 recall을 온전히 유지**(gain 3.04, 오히려 소폭↑ — low-rank는 recall에 거의 무관).
  random은 중간(1.35). → recall 붕괴는 high-rank에 **집중**.
- recall-specific: Δrecall(high)=7.53 ≫ Δlocal(high)=5.14 (margin 0.10 초과) — high pruning은 local
  일반 손상을 넘어 **recall을 초과 손상**.
- collapse_frac(=Δrecall/origin_gain): high **3.16×** vs low 0.39× vs random 1.03×.
- **3/3 seed 모두** high가 low·random을 동시에 이김(부호 일관 100%), std 매우 작음.

## 판정: G4 = YES — RECALL_ROLE_SUPPORTED
pure GDN의 **high-rank head는 in-context RECALL(induction) 유닛**이다. 역할 차원에서 논문의
"high-rank=prunable 잉여 / low-rank=retrieval-indispensable" 함의와 **정반대**다. Stage 2(high-rank =
load-bearing) · Stage 3(high-rank full rank = genuine capacity)에 이어, **그 부하의 정체가
retrieval(recall)임**을 역할 수준에서 규명한다.

## 한계 (정직)
- **hybrid 미포함**(pure만). 논문 대상 Qwen3-Next는 하이브리드라, attention이 recall을 대신 져서
  SSM high-rank가 역할 차원에서도 잉여였을 가능성은 별도 future work(매칭 hybrid ckpt 부재).
- probe는 실 텍스트 반복-복사 induction 한 종류. RULER-multikey 등 다른 recall 과제로 일반화는 미검증.
- 개입은 사후 KV-state pruning(head value→0). 단일 체크포인트, 3 seed.
- 논문 수치(93.8/46.9/90.6/38.9%)는 48층 post-trained target이지 18층 gdn2 pass 기준 아님.

## 재현
- 코드 = `stage4_recall_role.py`(probe+classify+verdict), `induction_probe.py`, `aggregate_seeds.py`.
- 결과 = `results/recall_role_100b_seed{0,1,2}/stage4_report*.json` + `results/aggregate_stage4.json`.
- 실행 = `sbatch run_recall_role.sbatch <seed>` ×3 → `run_aggregate.sbatch`(afterok). seed 0 sanity 선행.
- go/no-go 리뷰 = `working_note/260728_exp_review_recall-role-induction.md`.
