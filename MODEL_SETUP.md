# 모델 셋업 꿀팁 (linear/SSM recurrent LM 로딩·상태추출)

이 repo에서 여러 update-rule 모델을 로드하고 **recurrent state를 뽑아 rank 분석**하며 겪은 실전 gotcha 모음.
공통 환경: 클러스터 GPU는 **SLURM으로만** 실행(직접 CUDA 금지), env `sh_infocap`(torch 2.12+cu130, fla 0.5.1, transformers 5.12.1, mamba-ssm, lightning), 캐시/체크포인트 전부 `/data2`(루트 디스크 꽉 참).

## 공통 gotcha
- **SLURM 필수**: 노드에서 `CUDA_VISIBLE_DEVICES`가 비어 있음 → `--gres=gpu:rtx6000:1`로 할당. 직접 `python`은 GPU 못 봄.
- **캐시 경로 전부 `/data2`로**: `HF_HOME`, `TMPDIR`, `XDG_CACHE_HOME`, `TRITON_CACHE_DIR=/data2/sohyung/triton_cache` (Triton이 커널 컴파일 캐시를 씀 — 안 잡으면 루트/홈에 써서 문제).
- **transformers ≥ 5**: 많은 fla/커스텀 모델이 `_tied_weights_keys`를 **list**로 두는데 tr5는 dict/None 기대 → 로드 전에 `ModelCls._tied_weights_keys = {}` 패치.
- **Triton 커널은 GPU 전용**: fla의 `RMSNorm`·mixer 커널은 CPU에서 `"0 active drivers"`로 죽음 → **CPU 스모크 불가**, 검증은 GPU(SLURM)에서.
- **recurrent state 추출 공통 패턴**: `out = model(ids, use_cache=True); cache[i]["recurrent_state"]`. fla `Cache`(`fla.models.utils.Cache`)를 쓰고, `get/update_layer_cache`는 `module.layer_idx`로 인덱싱.

## 1. `state-spaces/mamba2-370m` (SSD) — mamba_ssm
```python
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.utils.generation import InferenceParams
m = MambaLMHeadModel.from_pretrained("state-spaces/mamba2-370m", device="cuda", dtype=torch.float32).eval()
# state: InferenceParams로 forward 후 key_value_memory_dict
ip = InferenceParams(max_seqlen=T, max_batch_size=B); m(ids, inference_params=ip)
S = ip.key_value_memory_dict[layer][1][0]     # (nheads=24, headdim=64, d_state=128)
```
- tokenizer는 `EleutherAI/gpt-neox-20b`. HF 표준 아님(AutoModel X) → mamba_ssm 필요.

## 2. `linear-moe-hub/{MoM-,}Gated-Deltanet-*` — fla + **weight adapter**
이 org 체크포인트(2025)는 **전이기 arch**라 현재 fla와 두 곳이 다름: **fused SwiGLU MLP**(`gate_proj = 2×intermediate`, `up_proj` 없음) + **`attn.D` skip term**. 그대로는 로드 실패.
```python
from fla.models.mom import MomConfig, MomForCausalLM         # MoM
# from fla.models.gated_deltanet import GatedDeltaNetConfig, GatedDeltaNetForCausalLM  # 순수(단일상태)
MdlCls._tied_weights_keys = {}
cfg = CfgCls.from_pretrained(hf); sd = load_file(hf_hub_download(hf, "model.safetensors"))
new = {}
for k, v in sd.items():
    if k.endswith("attn.D"):                      # 출력 skip → recurrent state에 안 들어가므로 버림
        continue
    if k.endswith("mlp.gate_proj.weight") and v.shape[0] % 2 == 0:
        h = v.shape[0] // 2
        new[k] = v[:h]; new[k.replace("gate_proj", "up_proj")] = v[h:]   # gate-half-first
    else:
        new[k] = v
model = MdlCls(cfg); model.load_state_dict(new, strict=False)
```
- **split 순서 검증**: `gate_first=True`가 맞음 — LM loss **3.11**(정) vs **14.05**(뒤집으면). 반드시 loss로 확인.
- **MoM은 rank 분석에 부적합**: mixture-of-memories(num_memories=4, topk=2) → `recurrent_state`가 list `[primary, shared]`, 유휴 메모리가 0행렬(eRank·participation-ratio 폭발 유발) + state 크기 부풀림. **단일상태 비교엔 순수 GatedDeltaNet이나 아래 gdn2를 쓸 것.**

## 3. `gyung/gdn2-370m-fineweb-edu-100b` (Gated DeltaNet-2, 단일상태) — **lit_gpt raw .pth**
HF 포맷 아님. repo엔 `checkpoint-{1..6}B-model-ckpt.pth`만(config·tokenizer·safetensors 없음). 학습 코드가 로컬에: `/home/sohyung/long-gdn/dsc/lit_gpt`.
```python
import sys; sys.path.insert(0, "/home/sohyung/long-gdn/dsc")
from lit_gpt.config import Config; from lit_gpt.model import GPT; from lit_gpt.gdn2 import GatedDeltaNet2
cfg = Config.from_name("gdn2_370M")               # 16층, 8헤드, head_k=head_v=128, vocab 32000
m = GPT(cfg).to("cuda").to(torch.bfloat16).eval()
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
m.load_state_dict(ck["model"] if "model" in ck else ck, strict=False)   # 0 missing/unexpected
```
- **`pip install lightning`** 필요(`lit_gpt.utils`가 import). fla 0.5.1이 dsc의 `get_layer_cache` 등 import 만족(추가 fla 불필요).
- tokenizer: `TinyLlama/TinyLlama_v1.1` (vocab 32000). dtype **bf16**.
- **상태추출이 까다로움**: 기본 `GPT.forward`는 gdn2를 `use_cache` 없이 호출 → 상태 저장 안 함. 두 가지 꼭 해야 함:
  1. 각 GDN-2 모듈에 **`layer_idx` 수동 세팅**(lit_gpt이 안 함) — 안 하면 `"requires layer_idx when past_key_values is provided"`.
  2. `GatedDeltaNet2.forward`를 **monkeypatch**해 shared `fla.models.utils.Cache` + `use_cache=True` 주입 → forward 후 `cache[i]["recurrent_state"]`.
- state shape: `(1, 16, 128, 128)`. LM loss 2.23(정상 로드 확인). layer0 eRank ≈ 9/128 (≈7%).

## 4. `fla-hub/{delta_net,gla,retnet}-*` (1.3B/100B) — fla 네이티브
```python
import fla
from fla.models.delta_net import DeltaNetConfig, DeltaNetForCausalLM   # gla/retnet도 동형
DeltaNetForCausalLM._tied_weights_keys = {}
m = DeltaNetForCausalLM.from_pretrained(hf, torch_dtype=torch.bfloat16).to("cuda").eval()   # 0 missing/unexpected
```
- **`HF_HUB_DISABLE_XET=1` 필수**: 이 repo들은 HF **xet** 스토리지라 인증 없이 다운로드하면 `401 Unauthorized` (CAS server). 끄면 일반 HTTPS로 받아짐.
- **bf16 권장**: DeltaNet chunk 커널은 **fp32 거부**(`ChunkDeltaRuleFunction does not support float32`). 다른 것도 bf16이 안전.
- state: `cache[i]["recurrent_state"]`, `use_gate=False`(DeltaNet=no-decay 확인). RetNet은 고정 decay라 eRank가 극단적으로 낮음(≈1/256, 거의 rank-1).
- 주의: **1.3B라 우리 370M(mamba2/gdn2)과 크기 confound** → eRank는 정규화(/max rank)로 비교.

## 5. fla `layers` from-scratch (tiny 모델 학습용)
`fla.layers`에 전 update rule이 nn.Module로 있음: `DeltaNet, GatedDeltaNet, GatedLinearAttention, MultiScaleRetention(RetNet), LinearAttention(순수 no-decay), Mamba2` — 시그니처 `(hidden_size, num_heads, layer_idx, ...)`, forward `(hidden_states, past_key_values=, use_cache=)`.
- **bf16 autocast로 학습**(DeltaNet fp32 거부). `torch.autocast("cuda", dtype=torch.bfloat16)`, 파라미터는 fp32 유지(AdamW 안정).
- **Mamba2 제약**: `num_heads*head_dim == expand*hidden_size` (예: hidden 256, expand 2 → num_heads 8, head_dim 64).
- `LinearAttention`은 short-conv 없음(가장 순수). GLA/RetNet은 `use_short_conv=True` 옵션 있음(MQAR recall에 크게 도움 = Zoology).
- **주의**: from-scratch는 학습 품질이 confound(hyperparam·steps) → 아키텍처 비교엔 pretrained가 더 깨끗. (이번에 GatedDeltaNet만 recall 1.0, 나머진 저학습/크래시.)

## 부록: recall / eRank 측정
- **MQAR(in-context)**: 시퀀스 `[k1 v1 … kN vN | k1…kN]`, query 위치 logits를 value id 범위로 argmax → 정답 비교. id는 reserved 범위(key_off=1000, val 5000+, <32000)로 모델 무관.
- **eRank** = `exp(정규화 특이값 스펙트럼의 Shannon 엔트로피)` (`capacity_utils.effective_rank`). state 행렬당(헤드/메모리별) 계산 후 평균, `/min(d1,d2)`로 정규화.
- **algebraic rank ≠ eRank**: 무작위/학습 가중치는 a.s. full algebraic rank지만 eRank는 스펙트럼 집중으로 훨씬 낮음 (`theory/random_matrix_full_rank.md`).
