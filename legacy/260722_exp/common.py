"""공통: 환경 설정 + gdn2-1.3B 로더 + logits/state 추출.

원 노트북(260720_boundary_vs_state_saturation.ipynb)의 §0 로더를 그대로 재사용.
경로는 env-agnostic: HF 다운로드/스크래치는 로컬(~), 큰 가중치는 볼륨의 슬림 파일(cwd 또는 boundary_experiment).
"""
import os, sys, math

_ON_GB = os.path.isdir("/data2/sohyung")
# gcc/triton 중간파일·HF 다운로드는 반드시 로컬 디스크로 (VESSL geesefs 볼륨에 쓰면 손상됨)
SCRATCH = os.environ.get("GDN2_SCRATCH") or ("/data2/sohyung/tmp" if _ON_GB
                                             else os.path.expanduser("~/.gdn2_scratch"))
os.makedirs(SCRATCH, exist_ok=True)
for _k in ("TMPDIR", "TEMP", "TMP"):
    os.environ[_k] = SCRATCH
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(SCRATCH, ".triton"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = "/data2/sohyung/hf_cache" if _ON_GB else os.path.expanduser("~/hf_local")

import numpy as np  # noqa: E402
import torch        # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
CONFIG_NAME = "gdn2_1.3B"
TOKENIZER = os.environ.get("GDN2_TOKENIZER", "TinyLlama/TinyLlama_v1.1")
GDN2_HF, GDN2_CKPT_FILE = "LLM-OS-Models2/gdn2-1.3b-paper-matched", "model-95b.pth"
WEIGHTS_NAME = "gdn2-1.3b-weights.pth"          # 슬림 bf16 가중치 (399 model tensors)

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIT_CANDS = [os.environ.get("LIT_GPT_PATH", ""), os.getcwd(), _HERE,
              os.path.join(_HERE, "..", "boundary_experiment"),
              os.path.join(os.getcwd(), "..", "boundary_experiment"),
              "/root/smaller/boundary_experiment", "/root/dscpkg", "/home/sohyung/long-gdn/dsc"]
_W_CANDS = [os.environ.get("GDN2_CKPT_PATH", ""),
            os.path.join(os.getcwd(), WEIGHTS_NAME), os.path.join(_HERE, WEIGHTS_NAME),
            os.path.join(_HERE, "..", "boundary_experiment", WEIGHTS_NAME),
            "/root/smaller/boundary_experiment/" + WEIGHTS_NAME,
            "/data2/sohyung/" + WEIGHTS_NAME]


def _find_lit():
    for c in _LIT_CANDS:
        if c and os.path.isdir(os.path.join(c, "lit_gpt")):
            return os.path.abspath(c)
    raise RuntimeError("lit_gpt(dscpkg) 못 찾음 — LIT_GPT_PATH 지정하세요.")


def _find_weights():
    for c in _W_CANDS:
        if c and os.path.exists(c) and os.path.getsize(c) > 0:
            return c
    return None


class Bundle:
    def __init__(self, model, shared, n_layer):
        self.model, self.shared, self.n_layer = model, shared, n_layer

    @torch.no_grad()
    def logits(self, ids):
        return self.model(ids.to(DEVICE)).float()

    @torch.no_grad()
    def states(self, ids):
        """recurrent state per layer: {layer: tensor(heads, dk, dv)} (CPU float)."""
        from fla.models.utils import Cache
        self.shared["cache"] = Cache()
        self.model(ids.to(DEVICE))
        st = {}
        for i in range(self.n_layer):
            try:
                rs = self.shared["cache"][i]["recurrent_state"]
            except Exception:
                continue
            elems = rs if isinstance(rs, (list, tuple)) else [rs]
            mats = [e.detach().float().cpu().reshape(-1, e.shape[-2], e.shape[-1])
                    for e in elems if torch.is_tensor(e) and e.dim() >= 2]
            if mats:
                st[i] = torch.cat(mats, 0)
        return st


def load_model():
    sys.path.insert(0, _find_lit())
    from lit_gpt.config import Config
    from lit_gpt.model import GPT
    from lit_gpt.gdn2 import GatedDeltaNet2
    cfg = Config.from_name(CONFIG_NAME)
    m = GPT(cfg).to(DEVICE).to(DTYPE).eval()

    w = _find_weights()
    if w is not None:
        print("[ckpt] local slim:", w)
        ck = torch.load(w, map_location="cpu", weights_only=False)
    else:
        from huggingface_hub import hf_hub_download
        print("[ckpt] HF download (17GB) ...")
        ck = torch.load(hf_hub_download(GDN2_HF, GDN2_CKPT_FILE), map_location="cpu", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    miss, unexp = m.load_state_dict(sd, strict=False)
    print(f"[load] missing={len(miss)} unexpected={len(unexp)} n_layer={cfg.n_layer}")

    SHARED = {"cache": None}
    _orig = GatedDeltaNet2.forward
    def patched(self, hidden_states, attention_mask=None, past_key_values=None, use_cache=False, **kw):
        pkv = SHARED["cache"] if SHARED["cache"] is not None else past_key_values
        uc = True if SHARED["cache"] is not None else use_cache
        return _orig(self, hidden_states, attention_mask=attention_mask,
                     past_key_values=pkv, use_cache=uc, **kw)
    GatedDeltaNet2.forward = patched
    for i, mm in enumerate([x for x in m.modules() if isinstance(x, GatedDeltaNet2)]):
        mm.layer_idx = i
        mm.mode = "fused_recurrent"
    return Bundle(m, SHARED, cfg.n_layer)


def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TOKENIZER)
