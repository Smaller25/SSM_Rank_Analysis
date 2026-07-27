# Stage 3 결과 — high-rank head는 oversaturated가 아니다 (genuine capacity) (2026-07-27)

논문 arXiv:2602.02195의 인과 사슬 "full rank ⟹ oversaturated ⟹ prunable"을 실 gdn2-1.3B(100B)+실텍스트에서
반박. saturation 두 정의 분리: (A)선형대수=full rank, (B)기능=간섭이 정보 파괴. 3 seed. NIAH floored→PPL primary.

## int-1 — prune-fraction sweep (count-matched, macro PPL 3-seed mean)
| KV frac | high | low | random |
|---|---|---|---|
| 0.00 | 12.6 | 12.6 | 12.6 |
| 0.10 | 47.4 | 14.4 | 15.1 |
| 0.20 | 140.6 | 16.9 | 19.8 |
| 0.30 | 381.7 | 23.1 | 25.9 |
| **0.389 (paper KV)** | **655.8** | 34.9 | 33.1 |
| 0.43 (our Stage2) | 861.9 | 40.3 | 47.5 |
| 0.50 | 1332.8 | 53.2 | 62.3 |
| 0.60 | 2192.1 | 83.8 | 148.8 |
→ 모든 압축률에서 high-rank pruning이 파국적, low ≈ random(거의 disposable). 논문 38.9% 지점: high 655.8 vs low 34.9.

## int-2 — SVD top-r 절단 기울기 (rank 줄일 때 PPL 상승률, 3-seed mean±std)
high **5.86±0.48**, low **0.033±0.007**, random 1.14±0.20. high ≫ low (~176×), high−random=4.72.
→ high-rank state의 full rank가 실제로 쓰임(절단하면 급락); low-rank는 이미 저차원(평평).

## int-3 — 스펙트럼-보존 noise (rank/에너지 보존, 내용만 파괴; PPL delta 3-seed mean±std)
high **16.6±0.8**, low **0.12±0.03**, random 5.47±0.06. high ≫ low (~141×), high−random=11.14.
→ rank/특이값을 그대로 두고 내용만 부수면 high-rank가 파국 → 저장된 **정보(내용)**가 중요, 단순 rank/에너지 아님.

## 판정: G3 = YES
high-rank head는 rank 용량을 **실제로 사용**(top-r 절단·spectrum-noise 둘 다에 저랭크/랜덤보다 훨씬 민감).
→ **full rank = genuine capacity, oversaturated 아님.** 논문의 "high-rank ⟹ oversaturated ⟹ prunable" 비약을
pure GDN에서 반박. Stage 2(고랭크 pruning 파국)와 합쳐, **정의-A(full rank) ≠ 정의-B(기능적 포화).**

## 3-Stage 서사
- Stage 1: rank stratification 재현, stratification = decay stratification (high-rank=저-decay=고-retention).
- Stage 2: 고랭크 pruning 파국(논문 역전) → high-rank = load-bearing.
- Stage 3: 고랭크가 full rank를 실제 사용 → NOT oversaturated. eRank/full-rank는 오해의 소지 있는 pruning 신호.

## 한계
NIAH floored→PPL primary. 순수-rank 분류(≠JRNP Saturation Score, α미공개). 사후 state surgery.
hybrid(swa_gdn2) 비교 = 매칭 체크포인트 부재로 future work. 논문 수치는 Qwen3-Next(48층 post-trained) target.
