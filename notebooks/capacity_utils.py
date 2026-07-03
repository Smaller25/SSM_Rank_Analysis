"""Shared utilities for the information-capacity notebooks.

Canonical home for the model loaders, recurrent-state extraction, and the
information-theoretic signals (S1 eRank, S2 predictive entropy, S3 in-context
epiplexity). Notebook 1 (`information_capacity_signals.ipynb`) defines these
inline; notebook 2 onward import them from here.

Run on a CUDA box (mamba-ssm for mamba2-370m, flash-linear-attention for the MoM
Gated-DeltaNet). See the parent README / SLURM script for the environment.
"""
import math
import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS = {
    "mamba2-370m": {"hf": "state-spaces/mamba2-370m", "kind": "mamba2"},
    "gdn-340m":    {"hf": "linear-moe-hub/MoM-Gated-Deltanet-340M", "kind": "gdn"},
}


# ----------------------------------------------------------------------------- signals
def effective_rank(matrix):
    """exp(Shannon entropy of the normalized singular-value spectrum)."""
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.cpu().numpy()
    s = np.linalg.svd(matrix, compute_uv=False)
    s = s / (s.sum() + 1e-12)
    return float(np.exp(-np.sum(s * np.log(s + 1e-12))))


# ----------------------------------------------------------------------------- models
class ModelBundle:
    def __init__(self, name, model, tokenizer, kind):
        self.name, self.model, self.tokenizer, self.kind = name, model, tokenizer, kind

    @torch.no_grad()
    def logits(self, input_ids):
        input_ids = input_ids.to(DEVICE)
        out = self.model(input_ids)
        return out.logits if hasattr(out, "logits") else out[0]

    @torch.no_grad()
    def states(self, input_ids):
        input_ids = input_ids.to(DEVICE)
        if self.kind == "mamba2":
            return _mamba2_states(self.model, input_ids)
        if self.kind == "gdn":
            return _gdn_states(self.model, input_ids)
        raise ValueError(self.kind)


def _mamba2_states(model, input_ids):
    from mamba_ssm.utils.generation import InferenceParams
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    ip = InferenceParams(max_seqlen=input_ids.shape[1], max_batch_size=input_ids.shape[0])
    with torch.no_grad():
        _ = model(input_ids, inference_params=ip)
    states = {}
    for layer_idx, (conv_state, ssm_state) in ip.key_value_memory_dict.items():
        states[layer_idx] = ssm_state[0].detach().cpu().float()   # (nheads, headdim, d_state)
    return states


def _gdn_states(model, input_ids):
    """MoM Gated-DeltaNet: recurrent_state is a list [primary, (shared?)]; return per-layer
    (num_memories*num_heads[+shared], head_k, head_v) matrices (assumes batch size 1)."""
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    assert input_ids.shape[0] == 1, "_gdn_states assumes batch size 1"
    with torch.no_grad():
        out = model(input_ids, use_cache=True)
    cache = out.past_key_values
    n_layers = getattr(model.config, "num_hidden_layers", None) or len(cache)
    states = {}
    for i in range(n_layers):
        try:
            rs = cache[i]["recurrent_state"]
        except Exception:
            continue
        elems = rs if isinstance(rs, (list, tuple)) else [rs]
        mats = []
        for el in elems:
            if el is None or not torch.is_tensor(el) or el.dim() < 2:
                continue
            t = el.detach().cpu().float()
            mats.append(t.reshape(-1, t.shape[-2], t.shape[-1]))
        if mats:
            states[i] = torch.cat(mats, dim=0)
    if not states:
        raise RuntimeError("No recurrent_state found in cache")
    return states


def load_bundle(name):
    spec = MODELS[name]
    if spec["kind"] == "mamba2":
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        model = MambaLMHeadModel.from_pretrained(spec["hf"], device=DEVICE, dtype=torch.float32).eval()
        return ModelBundle(name, model, tok, "mamba2")
    if spec["kind"] == "gdn":
        # linear-moe-hub checkpoints use a fused SwiGLU MLP + attn.D; adapt onto current fla.
        import fla  # noqa: F401
        from safetensors.torch import load_file
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer
        from fla.models.mom import MomConfig, MomForCausalLM
        MomForCausalLM._tied_weights_keys = {}
        cfg = MomConfig.from_pretrained(spec["hf"])
        ckpt = load_file(hf_hub_download(spec["hf"], "model.safetensors"))
        sd = {}
        for k, v in ckpt.items():
            if k.endswith("attn.D"):
                continue
            if k.endswith("mlp.gate_proj.weight") and v.shape[0] % 2 == 0:
                half = v.shape[0] // 2
                sd[k] = v[:half]
                sd[k.replace("gate_proj", "up_proj")] = v[half:]
            else:
                sd[k] = v
        model = MomForCausalLM(cfg)
        model.load_state_dict(sd, strict=False)
        tok = AutoTokenizer.from_pretrained(spec["hf"])
        model = model.to(DEVICE).eval()
        return ModelBundle(name, model, tok, "gdn")
    raise ValueError(name)


# ----------------------------------------------------------------------------- S1-S3
def S1_erank(bundle, input_ids):
    """Mean per-matrix effective rank of the final recurrent state, averaged over layers."""
    states = bundle.states(input_ids)
    per_layer = []
    for st in states.values():
        if st.dim() == 2:
            per_layer.append(effective_rank(st))
        else:
            per_layer.append(np.mean([effective_rank(st[h]) for h in range(st.shape[0])]))
    return {"erank_mean": float(np.mean(per_layer)),
            "erank_per_layer": np.asarray(per_layer)}


@torch.no_grad()
def S2_pred_entropy(bundle, input_ids):
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    logits = bundle.logits(input_ids)
    logp = torch.log_softmax(logits.float(), dim=-1)
    p = logp.exp()
    ent = -(p * logp).sum(-1)
    tgt = input_ids[:, 1:].to(logp.device)
    nll = -logp[:, :-1, :].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return {"pred_entropy_mean": float(ent.mean()),
            "bits_per_token": float(nll.mean() / math.log(2)),
            "pred_entropy_curve": ent[0].cpu().numpy()}


@torch.no_grad()
def S3_epiplexity(bundle, input_ids, tail_frac=0.2):
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    logits = bundle.logits(input_ids)
    logp = torch.log_softmax(logits.float(), dim=-1)
    tgt = input_ids[:, 1:].to(logp.device)
    nll = -logp[:, :-1, :].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[0]
    nll_bits = nll.cpu().numpy() / math.log(2)
    k = max(1, int(len(nll_bits) * tail_frac))
    asymptote = float(nll_bits[-k:].mean())
    excess = np.clip(nll_bits - asymptote, 0, None)
    return {"epiplexity_bits": float(excess.sum()),
            "epiplexity_per_token": float(excess.mean()),
            "asymptote_bits": asymptote,
            "nll_curve_bits": nll_bits}
