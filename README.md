# SSM Rank & State Analysis

Analysis of the SSM hidden state in Mamba-2 (forward-compatible for Mamba-3):
effective rank, head heterogeneity, semantic separability of states, state
injection, and document-position effects.

## Contents

### `notebooks/` — standalone Mamba-2 experiments
| Notebook | Question |
|---|---|
| `mamba2_effective_rank.ipynb` | Effective rank of the SSM hidden state `h_t` (heads × headdim × d_state) vs. context length `T`; per-head saturation point `T*`. |
| `exp1_state_analysis.ipynb` | Head heterogeneity across model scale (130m → 2.7b): `A_disc` vs. effective rank, Type A/B/C head classification, slow-head rank-saturation curves. |
| `exp2_state_similarity.ipynb` | Do same-topic documents yield similar Type-A head states? (intra- vs. inter-topic cosine similarity → can the state be used as a retrieval key?) |
| `exp3_state_injection.ipynb` | Does injecting a saved document state make the model behave as if it had read the document? (Oracle vs. Inject vs. No-context vs. Wrong-state.) |

### `mamba3_analysis/` — consolidated comprehensive suite
Unified notebook (`mamba3_comprehensive_analysis*.ipynb`) re-running the experiments
above plus state-linearity tests and a document-position-effect study, with
helper modules (`utils.py`, `exp4_position_effect.py`). See
[`mamba3_analysis/README.md`](mamba3_analysis/README.md) for full details.
Precomputed Exp1/Exp2 outputs are in `mamba3_analysis/results/` (`.npz`).

### `figures/`
Selected rank / per-head-change / accuracy plots.

## Setup

```bash
pip install torch transformers datasets mamba-ssm causal-conv1d
pip install matplotlib seaborn scipy scikit-learn tqdm
```

Primary model: `state-spaces/mamba2-370m` (swap in larger checkpoints or Mamba-3 when available).

## License

MIT
