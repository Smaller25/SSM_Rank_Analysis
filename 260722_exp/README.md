# 260722_exp — multi-state 생성 방법론 비교 (GT 정렬)

**목적**: GT 라벨이 있는 데이터셋을 넣어, linear-attention에서 **multi-state를 만드는 기존 방법론의 문제**를
드러내고 **state-수준 신호를 더한 dynamic**과 비교한다. 모델 backbone = gdn2-1.3B.

## 비교하는 방법론 (경계/청킹 기준)
- **control (content-agnostic)**: `fixed` (고정 길이), `log` (logarithmic schedule — memory caching에 언급만 되고
  기존 미구현)
- **기존 dynamic (DLA)**: `DLA_frobenius` — RMS-norm한 상태의 **상대 프로베니우스 변화율**
  `I_t = ||RMS(S_t) − RMS(S_{t-1})||_F / (||RMS(S_{t-1})||_F + ε)`, threshold로 새 state/합치기 이진결정.
- **논문 baseline info score**: `surprisal` (per-token NLL bits).
- **추가 state-signal 후보(제안)**: `epiplexity`, `erank`, `state_entropy`(상태 원소분포 엔트로피),
  그리고 **저장된 k-v 개수 추정군** `nuclear`(핵norm)·`participation`(참여비)·`numrank`(수치랭크).

## 데이터셋 (GT = 정답 경계 토큰위치)
| | 내용 | GT |
|---|---|---|
| D1 | 한 문장 반복(동질) | 없음 → 포화 캘리브레이션 |
| D2 | 같은 구조·새 사실(수도 64) | 각 사실 경계(과밀) |
| D3 | 토픽 전환 Math→Music→Code | 2 전환점 |
| D4 | 동일내용·언어전환 En→Zh→Ko | 2 언어경계 |
| D5 | MQAR(pair=N) | 질의 시작 |
| D6 | 수제 다중 needle | needle 위치 |
| **D7** | **RULER MK-NIAH (lm-eval-harness와 동일)** | **needle 삽입 위치** |

D7은 `niah_ruler.py`가 lm-eval `prepare_niah.py`(niah_multikey_1: essay haystack, words→numbers, 4 keys,
seed 42)를 **그대로 복제**해 생성하고 needle 토큰위치를 GT로 뽑음. RULER는 본래 long-context라 D7은 길다
(`--niah-len`, 기본 2048).

## 평가
- **count-matched**: 모든 방법에 |GT|개 경계 부여 → **위치 F1** + 무작위 null 대비 z (신호 우열의 공정 비교).
- **natural-count**: threshold(z>1.5)에서 몇 개 자르나 → 과/과소분할 경향.

## 실행 (VESSL A100)
```bash
cd <볼륨>/260722_exp          # boundary_experiment 옆에 두면 lit_gpt/가중치 자동 인식
pip install "flash-linear-attention==0.5.1" "transformers==5.12.1" lightning lightning-utilities \
            einops scipy huggingface_hub matplotlib
# D7(MK-NIAH) 의존 (lm-eval-harness와 동일 생성):
pip install wonderwords nltk datasets
python -c "import nltk; nltk.download('punkt_tab')"

python run.py                 # D2~D7
python run.py --datasets D7 --niah-len 2048 --plot
```
- 가중치: 옆 `boundary_experiment/gdn2-1.3b-weights.pth`(슬림 bf16) 자동 사용. 없으면 `GDN2_CKPT_PATH` 지정.
- HF 다운로드/스크래치는 로컬(`~/hf_local`,`~/.gdn2_scratch`)로 감(볼륨 geesefs 손상 회피).
- state 궤적은 prefix 재실행(`--stride`, 기본 8). SVD 비용 크면 `--layer-stride 2`.

## 파일
- `common.py` 모델 로더·state 추출  · `data.py` D1~D7  · `niah_ruler.py` RULER MK-NIAH 복제
- `analysis.py` 신호·청커·평가  · `run.py` 오케스트레이션
