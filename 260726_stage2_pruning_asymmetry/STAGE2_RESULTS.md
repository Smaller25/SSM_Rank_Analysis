# Stage 2 결과 — KV-state pruning 비대칭 (gdn2-1.3B 100B, 3 seed) (2026-07-27)

논문 arXiv:2602.02195 §5의 pruning 비대칭(low-rank pruning이 high-rank보다 훨씬 해로움)을 pure
GDN(gdn2-1.3B 100B)에서 검증. **진짜 KV-state pruning**(head value→0, S_h=0, KV 메모리 제거) 4조건
(origin/high/low/random), 동수 마스킹, 3 seed(0/1/2).

## 세팅
- KV-state pruning: head의 value 채널(v_conv1d 출력)을 0으로 → 그 head의 recurrent state S_h=0
  → KV state에서 제거. **124/288 head 마스킹 = KV 43.06% 절감** (논문 38.9%와 비교 가능).
- head 분류: per-head threshold-rank(Eq.6, ε=1e-4), k=min(#low,#high)로 disjoint 보장.
- NIAH = RULER-multikey(needle 값 방출), PPL = wikitext/github/arxiv(data_cache 실텍스트). seed 3개.
- 아키텍처 사실: GDN2는 head 독립·o_norm(0)=0 → state-prune 정확도 == output-ablation, 차이는 메모리뿐.

## 결과 (3 seed mean±std)
| 조건 | NIAH acc | macro PPL |
|---|---|---|
| origin | 0.183±0.047 | 12.6±0.4 |
| high-rank prune | 0.000 | **854.3±101** |
| low-rank prune | 0.017±0.024 | 39.4±3.0 |
| random prune | 0.000 | 34.4±1.7 |

- NIAH origin=0.18 < 0.30 → floor(UNTESTABLE) → primary DV = PPL.
- PPL 상승(vs origin): high **+841.7±100.9** / low +26.8±2.6 / random +21.8±1.7.

## 판정: G2 = NO (논문과 정반대)
- high-rank pruning이 파국적(PPL 12.6→854, ~32×) → **high-rank head = load-bearing**.
- low-rank pruning은 random과 거의 동급(+26.8 vs +21.8) → **low-rank head ≈ redundant**.
- 논문 예측(low 붕괴/high 불변)의 **역전**. pure GDN에선 high-rank(=저-decay, 고-retention;
  Stage 1 G1b R²~0.9) head가 메모리를 지고 있어 제거하면 무너짐.

## 해석
논문의 "high-rank prunable"은 Qwen3-Next가 **하이브리드(attention 층)**라 attention이 부하를 지고
SSM high-rank head가 잉여였을 가능성. pure GDN엔 attention이 없어 high-rank SSM head가 곧 메모리
→ critical. 논문 비대칭은 **하이브리드/post-training 특이적**, pure GDN엔 일반화 안 됨(역전).

## 한계
- NIAH floored → 신호는 PPL. output-ablation-등가 state-prune(메모리 43% 절감 리포트).
- RULER-multikey(≠논문 single-needle), 순수-rank 분류(≠JRNP Saturation Score, α 미공개).
- 논문 수치는 Qwen3-Next(48층 post-trained) target, pure 18층 gdn2의 pass 기준 아님(PREREG).

## 재현
`bash submit_stage2_seeds.sh` (SLURM gpu01) → seed별 results/stage2_100b_seed{0,1,2}/ →
`aggregate_seeds.py` → results/aggregate_stage2.json.
