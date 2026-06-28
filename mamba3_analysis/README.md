# Mamba3 State Analysis - Comprehensive Suite

A unified Jupyter notebook for analyzing Mamba2 state-space models, forward-compatible for Mamba3.

## Overview

This project consolidates 4 experiments into a single comprehensive notebook:

1. **Experiment 1**: Head Heterogeneity & Effective Rank Analysis
2. **Experiment 2**: State Similarity for Semantic Separability
3. **Experiment 3**: State Injection Validation (Enhanced with MSE)
4. **Experiment 3-2**: State Linearity Tests (NEW)
5. **Experiment 4**: Document Position Effect on QA (NEW)

## Model Selection

**Primary Model**: `state-spaces/mamba2-370m`

Mamba3 not yet publicly released (paper: March 2026, ICLR 2026). The notebook is designed modularly to easily swap Mamba3 models when available.

Available Mamba2 models:
- `state-spaces/mamba2-130m`
- `state-spaces/mamba2-370m` ← **Recommended**
- `state-spaces/mamba2-780m`
- `state-spaces/mamba2-1.3b`
- `state-spaces/mamba2-2.7b`

## Setup

### Prerequisites

```bash
pip install torch transformers datasets mamba-ssm causal-conv1d
pip install matplotlib seaborn scipy scikit-learn tqdm
```

### File Structure

```
mamba3_analysis/
├── mamba3_comprehensive_analysis.ipynb  ← Main unified notebook
├── exp1_head_heterogeneity.py          ← Code modules (for reference)
├── exp2_semantic_separability.py
├── exp3_state_injection.py
├── exp3_2_state_linearity.py
├── exp4_position_effect.py
├── results/
│   ├── exp1_*.npz
│   ├── exp2_*.npz
│   ├── exp3_*.json
│   ├── exp3_2_*.json
│   ├── exp4_*.json
│   └── plots/
│       ├── exp1_*.png
│       ├── exp2_*.png
│       ├── exp3_*.png
│       ├── exp3_2_*.png
│       └── exp4_*.png
└── README.md
```

## Usage

### Option 1: Run the Complete Notebook

Open `mamba3_comprehensive_analysis.ipynb` and run all cells sequentially.

**Runtime estimates** (on mamba2-370m):
- Exp1: ~10-15 min
- Exp2: ~10 min
- Exp3: ~5 min
- Exp3-2: ~15 min
- Exp4: ~20 min
- **Total**: ~60-90 min for full suite

### Option 2: Add Experiment Code to Notebook

The main notebook contains setup and utilities (Sections 0-3). To add experiments:

1. Open `mamba3_comprehensive_analysis.ipynb`
2. Copy cells from experiment files (`exp1_*.py`, `exp2_*.py`, etc.)
3. Paste into notebook after Section 3
4. Run cells sequentially

### Option 3: Run Specific Experiments

If you only need certain experiments, run:
1. Sections 0-3 (Setup, always required)
2. Experiment 1 (if Exp4 will be run - provides T*)
3. Your desired experiments

**Note**: Experiment 4 requires Experiment 1 to measure T*.

## Experiment Details

### Experiment 1: Head Heterogeneity

**What it does**:
- Extracts A_disc parameters and classifies heads (Type A/B/C)
- Computes effective rank at T=256
- Analyzes rank trajectories (T=32 to T=1024)
- Detects saturation point (T*)

**Key outputs**:
- Heatmaps: A_disc and effective rank (Layer × Head)
- Rank trajectory plots
- T* distribution

### Experiment 2: Semantic Separability

**What it does**:
- Extracts states (Type A, Type C, All heads, Hidden) for 30 documents
- Computes intra-topic vs inter-topic cosine similarity
- Statistical significance testing

**Key outputs**:
- Similarity matrices
- Separability gap bar chart
- t-SNE visualization

### Experiment 3: State Injection

**What it does**:
- Tests RAG mechanism via 4 setups:
  - A (Oracle): doc + query → forward
  - B (Inject): doc → state → inject + query
  - C (No context): query only
  - D (Wrong context): irrelevant state + query
- **NEW**: MSE between Oracle and Injected states

**Key outputs**:
- Hit rate comparison
- MSE heatmap (Layer × QA pair)
- Full answer comparison table

### Experiment 3-2: State Linearity (NEW)

**What it does**:
- Tests 4 hypotheses:
  1. Addition: `state(doc1+doc2) ≟ state(doc1) + state(doc2)`
  2. Subtraction: Can we remove information?
  3. Weighted merge: Does α control balance?
  4. Generation quality: Are merged states coherent?

**Key outputs**:
- MSE heatmaps for each test
- Weighted merge quality vs alpha
- Generation quality assessment

### Experiment 4: Position Effect (NEW)

**What it does**:
- Tests 10 QA pairs in 3 positions (Front/Middle/Back)
- Sequence length: 2 × T* (from Exp1)
- Statistical tests for position bias

**Hypotheses**:
- H1 (Recency bias): Back > Middle > Front
- H2 (No bias): Front ≈ Middle ≈ Back

**Key outputs**:
- Accuracy bar chart with significance tests
- Per-topic heatmap
- State rank by position

## Results Interpretation

### Expected Patterns

**Exp1**:
- Type A heads (slow decay): High rank, long memory
- Type C heads (fast decay): Low rank, short memory
- T* typically around 200-300 tokens for mamba2-370m

**Exp2**:
- Type A state should have highest separability gap
- p-value < 0.001 indicates strong topic clustering

**Exp3**:
- Expected pattern: B ≈ A >> C >> D
- Low MSE (< 0.01) means injection preserves state well

**Exp3-2**:
- **Highly linear**: MSE < 0.1, cosine > 0.95
- **Partially linear**: MSE 0.1-0.5 (likely outcome)
- **Non-linear**: MSE > 1.0

**Exp4**:
- If recency bias exists: Back > Front
- If position-invariant: Front ≈ Back

## Verification

To verify against previous Mamba2 results:

```python
# Load old results
old_data = np.load('/Users/smaller225/code/state_memory/260401/multi_model_results/mamba2-370m_results.npz')

# Compare
print("Old max rank:", old_data['ranks_256'].max())
print("New max rank:", exp1_results['state-spaces/mamba2-370m']['ranks_256'].max())
```

Results should match within ±5% tolerance.

## Troubleshooting

### Out of Memory

- Use smaller model (mamba2-130m)
- Reduce T_MAX from 1024 to 512
- Clear cache between experiments: `clear_memory()`

### Slow Runtime

- Enable GPU: Check `torch.cuda.is_available()`
- Reduce N_SAMPLES from 20 to 10
- Use only one model in MODEL_LIST

### Import Errors

```bash
# If mamba_ssm not found
pip install mamba-ssm causal-conv1d --no-cache-dir

# If transformers version issues
pip install transformers>=4.30.0
```

## Future Work

When Mamba3 is released:

1. Update MODEL_LIST to include Mamba3 models
2. Check if state extraction API changed
3. Re-run all experiments for comparison
4. Expected improvements: Better linearity in Exp3-2, higher T* in Exp1

## Citation

If you use this notebook, please cite:

```
@inproceedings{mamba3,
  title={Mamba3: Scaling State Space Models for Reasoning},
  author={Gu et al.},
  booktitle={ICLR},
  year={2026}
}
```

## Contact

For questions or issues, refer to:
- Mamba GitHub: https://github.com/state-spaces/mamba
- Transformers: https://github.com/huggingface/transformers

## License

MIT License - Free to use with attribution.
