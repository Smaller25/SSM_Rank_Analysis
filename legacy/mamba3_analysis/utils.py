import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.utils.generation import InferenceParams
from scipy.stats import ttest_rel, ttest_ind
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm
import json
import os
import warnings


def effective_rank(matrix):
    """
    Compute effective rank using Shannon entropy.
    
    Args:
        matrix: (headdim, d_state) numpy array or tensor
    
    Returns:
        float: effective rank
    """
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.cpu().numpy()
    
    s = np.linalg.svd(matrix, compute_uv=False)
    s = s / (s.sum() + 1e-12)
    entropy = -np.sum(s * np.log(s + 1e-12))
    rank = np.exp(entropy)
    
    return rank

def get_ssm_states(model, input_ids):
    """
    InferenceParams를 통해 각 layer의 ssm_state 추출.
    
    Args:
        model: Mamba model
        input_ids: (batch, seq_len) tensor
    
    Returns:
        states: {layer_idx: Tensor(nheads, headdim, d_state)}
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    
    inference_params = InferenceParams(
        max_seqlen=input_ids.shape[1],
        max_batch_size=input_ids.shape[0]
    )
    
    with torch.no_grad():
        _ = model(input_ids, inference_params=inference_params)
    
    states = {}
    for layer_idx, (conv_state, ssm_state) in inference_params.key_value_memory_dict.items():
        # ssm_state: (batch, nheads, headdim, d_state) → squeeze batch
        states[layer_idx] = ssm_state.squeeze(0).cpu().float()
    
    return states

def get_A_disc(model):
    """
    Extract discrete-time decay parameters A_disc from all SSM layers.
    Mamba2 uses A_log, so A_disc = exp(A_log).
    
    Returns:
        np.array: (n_layers, n_heads) array of A_disc values
    """
    A_disc_all = []
    
    for layer in model.backbone.layers:
        if hasattr(layer, 'mixer') and hasattr(layer.mixer, 'A_log'):
            # A_log shape: (n_heads,)
            A_log = layer.mixer.A_log.detach()
            A_disc = torch.exp(A_log).cpu().numpy()
            A_disc_all.append(A_disc)
        elif hasattr(layer, 'mixer') and hasattr(layer.mixer, 'A'):
            # Fallback for Mamba1 or other variants
            A = layer.mixer.A.detach().cpu().numpy()
            if A.ndim == 2:
                A = A.mean(axis=1)
            A_disc_all.append(A)
    
    return np.array(A_disc_all)  # (n_layers, n_heads)

def classify_heads(A_disc, slow_threshold=0.99, fast_threshold=0.50):
    """
    Classify heads into Type A (slow), B (medium), C (fast).
    
    Args:
        A_disc: (n_layers, n_heads) array
    
    Returns:
        dict: {'A': mask, 'B': mask, 'C': mask}
    """
    type_A = A_disc >= slow_threshold
    type_C = A_disc <= fast_threshold
    type_B = ~(type_A | type_C)
    
    return {
        'A': type_A,
        'B': type_B,
        'C': type_C
    }

def extract_state(model, tokenizer, text, device='cuda'):
    """
    Extract state for a given text (for injection experiments).
    
    Returns:
        dict: {layer_idx: state_tensor (nheads, headdim, d_state)}
    """
    input_ids = tokenizer(text, return_tensors='pt')['input_ids'].to(device)
    return get_ssm_states(model, input_ids)


from mamba_ssm.utils.generation import decode

def inject_state_and_generate(model, tokenizer, query, state_dict, max_new_tokens=50, device='cuda'):
    query_ids = tokenizer(query, return_tensors='pt')['input_ids'].to(device)
    q_len = query_ids.shape[1]

    inference_params = InferenceParams(
        max_seqlen=q_len + max_new_tokens,
        max_batch_size=1
    )

    with torch.no_grad():
        out = model(query_ids, inference_params=inference_params)

    for layer_idx, ssm_state in state_dict.items():
        conv_state, _ = inference_params.key_value_memory_dict[layer_idx]
        inference_params.key_value_memory_dict[layer_idx] = (
            conv_state, ssm_state.unsqueeze(0).to(device)
        )

    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = torch.cat([query_ids, next_token], dim=1)
    inference_params.seqlen_offset = q_len

    with torch.no_grad():
        for _ in range(max_new_tokens - 1):
            if next_token.item() == tokenizer.eos_token_id:
                break
            out = model(next_token, inference_params=inference_params)
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            inference_params.seqlen_offset += 1

    answer = tokenizer.decode(generated[0, q_len:], skip_special_tokens=True)
    
    # 생성 후 최종 state 반환
    final_state = {k: v[1].squeeze(0).cpu()
                   for k, v in inference_params.key_value_memory_dict.items()}
    
    return answer, final_state
    
# def inject_state_and_generate(model, tokenizer, query, state_dict, max_new_tokens=50, device='cuda'):
#     """
#     Generate text using injected state.
    
#     Args:
#         model: Mamba model
#         tokenizer: tokenizer
#         query: query string
#         state_dict: {layer_idx: state_tensor (nheads, headdim, d_state)}
#         max_new_tokens: generation length
    
#     Returns:
#         str: generated text
#     """
#     query_ids = tokenizer(query, return_tensors='pt')['input_ids'].to(device)
#     batch_size = query_ids.shape[0]
    
#     # Create inference params
#     inference_params = InferenceParams(
#         max_seqlen=query_ids.shape[1] + max_new_tokens,
#         max_batch_size=batch_size
#     )
    
#     # Inject states as (conv_state=None, ssm_state)
#     for layer_idx, ssm_state in state_dict.items():
#         # ssm_state needs batch dimension: (1, nheads, headdim, d_state)
#         ssm_state_batched = ssm_state.unsqueeze(0).to(device)
#         # conv_state can be None
#         inference_params.key_value_memory_dict[layer_idx] = (None, ssm_state_batched)
    
#     # Generate
#     with torch.no_grad():
#         output_ids = model.generate(
#             query_ids,
#             max_length=query_ids.shape[1] + max_new_tokens,
#             inference_params=inference_params,
#             eos_token_id=tokenizer.eos_token_id,
#             pad_token_id=tokenizer.pad_token_id,
#             return_dict_in_generate=False
#         )
    
#     return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def compute_state_mse(state1, state2):
    """
    Compute MSE between two state dictionaries.
    
    Returns:
        dict: {layer_idx: mse_value}
        float: mean MSE across all layers
    """
    mse_per_layer = {}
    
    for layer_idx in state1.keys():
        if layer_idx in state2:
            s1 = state1[layer_idx].float()
            s2 = state2[layer_idx].float()
            mse = torch.mean((s1 - s2) ** 2).item()
            mse_per_layer[layer_idx] = mse
    
    mean_mse = np.mean(list(mse_per_layer.values())) if mse_per_layer else 0.0
    
    return mse_per_layer, mean_mse
    
def compute_state_cosine(state1, state2):
    """
    Compute cosine similarity between two state dictionaries.
    
    Returns:
        dict: {layer_idx: cosine_sim}
        float: mean cosine similarity
    """
    cos_per_layer = {}
    
    for layer_idx in state1.keys():
        if layer_idx in state2:
            s1 = state1[layer_idx].flatten().cpu().numpy()
            s2 = state2[layer_idx].flatten().cpu().numpy()
            cos = cosine_similarity(s1.reshape(1, -1), s2.reshape(1, -1))[0, 0]
            cos_per_layer[layer_idx] = cos
    
    mean_cos = np.mean(list(cos_per_layer.values())) if cos_per_layer else 0.0
    
    return cos_per_layer, mean_cos

def find_saturation_point(rank_trajectory, threshold_ratio=0.95):
    """
    Find T* where rank reaches 95% of maximum.
    
    Args:
        rank_trajectory: array of ranks at different T
        threshold_ratio: ratio of max rank to consider saturated
    
    Returns:
        int: T* (index in T_RANGE)
    """
    max_rank = np.max(rank_trajectory)
    threshold = threshold_ratio * max_rank
    
    for idx, rank in enumerate(rank_trajectory):
        if rank >= threshold:
            return idx
    
    return len(rank_trajectory) - 1
    
def check_keywords(text, keywords):
    """
    Check if any keyword is present in text (case-insensitive).
    
    Returns:
        bool: True if at least one keyword found
    """
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)

def clear_memory():
    """Clear CUDA cache and run garbage collection."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

print("✓ All utility functions defined successfully!")