"""
Experiment 4: Document Position Effect on QA (NEW)

This module tests how document position affects retrieval accuracy.
Copy the cells below into the main notebook after Experiment 3-2.
"""

# Cell: Markdown
"""
## 8. Experiment 4: Document Position Effect

This experiment tests whether document position in context affects QA accuracy:

**Setup**:
- 10 diverse documents (one QA pair each)
- Total length = 2 × T* (from Exp1)
- 3 positions: Front, Middle, Back

**Hypothesis**:
- H1 (Recency bias): Back > Middle > Front
- H2 (No bias): Front ≈ Middle ≈ Back

**Analysis**: Accuracy, state rank, statistical tests
"""

# Cell: Code - Dependency check and setup
"""
print("="*60)
print("Experiment 4: Document Position Effect")
print("="*60)

# Check if Exp1 was run
model_name = MODEL_LIST[0]
if model_name not in exp1_results:
    print("ERROR: Experiment 1 must be run first to measure T*!")
    print("Please run Experiment 1 before continuing.")
else:
    T_star = exp1_results[model_name]['T_star_measured']
    print(f"\\nUsing T* from Exp1: {T_star:.0f} tokens")

    # Set total length
    TOTAL_LENGTH = int(2 * T_star)
    DOC_LENGTH_EXP4 = TOTAL_LENGTH // N_POSITION_DOCS

    print(f"Total sequence length: {TOTAL_LENGTH} tokens")
    print(f"Target doc length: ~{DOC_LENGTH_EXP4} tokens each")
    print(f"Number of documents: {N_POSITION_DOCS}\\n")
"""

# Cell: Code - Create position test documents
"""
# Create 10 diverse documents with QA pairs
POSITION_TEST_DOCS = [
    {
        'topic': 'AI/ML',
        'doc': 'Deep learning models use neural networks with multiple layers to learn hierarchical representations of data. Convolutional neural networks excel at computer vision tasks by detecting spatial patterns. Recurrent networks handle sequential data through memory mechanisms. Attention mechanisms allow models to focus on relevant information dynamically.',
        'question': 'What type of networks are good for computer vision?',
        'keywords': ['convolutional', 'CNN', 'convolutional neural']
    },
    {
        'topic': 'Space',
        'doc': 'The James Webb Space Telescope observes distant galaxies in infrared wavelengths. Launched in 2021, it orbits the Sun at the L2 Lagrange point. Its primary mirror spans 6.5 meters and consists of 18 hexagonal segments. The telescope can peer back to 13.5 billion years ago to see the first galaxies.',
        'question': 'What telescope observes in infrared wavelengths?',
        'keywords': ['James Webb', 'Webb', 'JWST']
    },
    {
        'topic': 'History',
        'doc': 'The Library of Alexandria was one of the largest libraries of the ancient world. Built in the 3rd century BCE in Egypt, it contained hundreds of thousands of scrolls. Scholars from across the Mediterranean came to study there. Its destruction remains one of history\\'s greatest losses of knowledge.',
        'question': 'Where was the ancient Library of Alexandria located?',
        'keywords': ['Egypt', 'Egyptian', 'Alexandria Egypt']
    },
    {
        'topic': 'Physics',
        'doc': 'Quantum entanglement is a phenomenon where particles become correlated in ways that classical physics cannot explain. When two particles are entangled, measuring one instantly affects the other regardless of distance. Einstein called this \"spooky action at a distance\" and was initially skeptical of it.',
        'question': 'What did Einstein call quantum entanglement?',
        'keywords': ['spooky action', 'spooky', 'distance']
    },
    {
        'topic': 'Geography',
        'doc': 'Lake Baikal in Siberia is the world\\'s deepest and oldest freshwater lake. It reaches depths of 1,642 meters and formed approximately 25 million years ago. The lake contains about 20% of Earth\\'s unfrozen freshwater. Over 1,700 species of plants and animals live there, many found nowhere else.',
        'question': 'What is the deepest freshwater lake in the world?',
        'keywords': ['Baikal', 'Lake Baikal']
    },
    {
        'topic': 'Literature',
        'doc': 'Gabriel García Márquez pioneered magical realism in literature with novels like \"One Hundred Years of Solitude\". This Colombian author won the Nobel Prize in Literature in 1982. His works blend fantastical elements with realistic settings seamlessly. Márquez\\'s storytelling influenced generations of Latin American writers.',
        'question': 'Who wrote \"One Hundred Years of Solitude\"?',
        'keywords': ['García Márquez', 'Márquez', 'Gabriel']
    },
    {
        'topic': 'Sports',
        'doc': 'Usain Bolt holds the world record for the 100-meter sprint at 9.58 seconds. The Jamaican sprinter won eight Olympic gold medals during his career. He dominated sprinting from 2008 to 2016 and retired undefeated in major championships. Bolt\\'s distinctive celebration pose became iconic worldwide.',
        'question': 'Who holds the 100-meter sprint world record?',
        'keywords': ['Usain Bolt', 'Bolt']
    },
    {
        'topic': 'Music',
        'doc': 'Johann Sebastian Bach composed the Brandenburg Concertos between 1718 and 1721. These six concertos showcase different combinations of instruments. Each concerto features a distinct character and virtuosic writing. The Brandenburg Concertos remain among the finest examples of Baroque orchestral music.',
        'question': 'Who composed the Brandenburg Concertos?',
        'keywords': ['Bach', 'Johann Sebastian Bach', 'J.S. Bach']
    },
    {
        'topic': 'Cuisine',
        'doc': 'Sourdough bread uses natural fermentation with wild yeast and lactic acid bacteria. The starter culture must be fed regularly with flour and water to remain active. Fermentation can take 12-48 hours depending on temperature and starter strength. The process creates the bread\\'s characteristic tangy flavor and chewy texture.',
        'question': 'What gives sourdough bread its tangy flavor?',
        'keywords': ['fermentation', 'lactic acid', 'bacteria']
    },
    {
        'topic': 'Medicine',
        'doc': 'Penicillin was discovered by Alexander Fleming in 1928 when he noticed mold killing bacteria in a petri dish. This antibiotic revolutionized medicine by enabling treatment of bacterial infections. Fleming shared the 1945 Nobel Prize in Physiology or Medicine for this discovery. Penicillin saved countless lives during World War II.',
        'question': 'Who discovered penicillin?',
        'keywords': ['Fleming', 'Alexander Fleming']
    },
]

print(f"Created {len(POSITION_TEST_DOCS)} documents with QA pairs")
print("Topics:", [d['topic'] for d in POSITION_TEST_DOCS])
"""

# Cell: Code - Model loading
"""
# Load model
print(f"\\nLoading model: {model_name}")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=device,
    torch_dtype=torch.float32,
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.eval()
n_layers = len(model.backbone.layers)
n_heads = model.backbone.layers[0].mixer.A.shape[0]

print(f"Model loaded: {n_layers} layers, {n_heads} heads")
"""

# Cell: Code - Position test execution
"""
print("\\nRunning position tests...")
print("This will test each QA pair in 3 positions: Front, Middle, Back\\n")

position_results = []

for target_idx in tqdm(range(len(POSITION_TEST_DOCS)), desc="Target docs"):
    target = POSITION_TEST_DOCS[target_idx]
    others = [POSITION_TEST_DOCS[i] for i in range(len(POSITION_TEST_DOCS)) if i != target_idx]

    # Shuffle others for randomness
    np.random.seed(42 + target_idx)  # Reproducible
    np.random.shuffle(others)

    result = {
        'target_idx': target_idx,
        'topic': target['topic'],
        'question': target['question'],
        'keywords': target['keywords'],
    }

    # Position 1: Front
    seq_front = [target['doc']] + [d['doc'] for d in others]
    combined_text_front = " ".join(seq_front)

    input_ids_front = tokenizer(
        combined_text_front,
        return_tensors='pt',
        truncation=True,
        max_length=TOTAL_LENGTH
    )['input_ids'].to(device)

    # Generate answer
    query_with_ctx_front = combined_text_front + " " + target['question']
    input_ids_query_front = tokenizer(
        query_with_ctx_front,
        return_tensors='pt',
        truncation=True,
        max_length=TOTAL_LENGTH + 50
    )['input_ids'].to(device)

    with torch.no_grad():
        output_ids_front = model.generate(
            input_ids_query_front,
            max_new_tokens=30,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False
        )

    answer_front = tokenizer.decode(output_ids_front[0], skip_special_tokens=True)
    answer_front_only = answer_front[len(query_with_ctx_front):].strip()
    hit_front = check_keywords(answer_front_only, target['keywords'])

    # Extract states for rank analysis
    states_front = get_ssm_states(model, input_ids_front)

    # Position 2: Middle (index 5)
    seq_middle = [d['doc'] for d in others[:5]] + [target['doc']] + [d['doc'] for d in others[5:]]
    combined_text_middle = " ".join(seq_middle)

    input_ids_middle = tokenizer(
        combined_text_middle,
        return_tensors='pt',
        truncation=True,
        max_length=TOTAL_LENGTH
    )['input_ids'].to(device)

    query_with_ctx_middle = combined_text_middle + " " + target['question']
    input_ids_query_middle = tokenizer(
        query_with_ctx_middle,
        return_tensors='pt',
        truncation=True,
        max_length=TOTAL_LENGTH + 50
    )['input_ids'].to(device)

    with torch.no_grad():
        output_ids_middle = model.generate(
            input_ids_query_middle,
            max_new_tokens=30,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False
        )

    answer_middle = tokenizer.decode(output_ids_middle[0], skip_special_tokens=True)
    answer_middle_only = answer_middle[len(query_with_ctx_middle):].strip()
    hit_middle = check_keywords(answer_middle_only, target['keywords'])

    states_middle = get_ssm_states(model, input_ids_middle)

    # Position 3: Back
    seq_back = [d['doc'] for d in others] + [target['doc']]
    combined_text_back = " ".join(seq_back)

    input_ids_back = tokenizer(
        combined_text_back,
        return_tensors='pt',
        truncation=True,
        max_length=TOTAL_LENGTH
    )['input_ids'].to(device)

    query_with_ctx_back = combined_text_back + " " + target['question']
    input_ids_query_back = tokenizer(
        query_with_ctx_back,
        return_tensors='pt',
        truncation=True,
        max_length=TOTAL_LENGTH + 50
    )['input_ids'].to(device)

    with torch.no_grad():
        output_ids_back = model.generate(
            input_ids_query_back,
            max_new_tokens=30,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False
        )

    answer_back = tokenizer.decode(output_ids_back[0], skip_special_tokens=True)
    answer_back_only = answer_back[len(query_with_ctx_back):].strip()
    hit_back = check_keywords(answer_back_only, target['keywords'])

    states_back = get_ssm_states(model, input_ids_back)

    # Compute effective ranks
    def compute_mean_rank(states):
        ranks = []
        for layer_idx in states:
            state = states[layer_idx][0]  # (n_heads, d_state, d_state)
            for head_idx in range(n_heads):
                head_state = state[head_idx].cpu().numpy()
                ranks.append(effective_rank(head_state))
        return np.mean(ranks)

    rank_front = compute_mean_rank(states_front)
    rank_middle = compute_mean_rank(states_middle)
    rank_back = compute_mean_rank(states_back)

    # Store results
    result.update({
        'front_hit': hit_front,
        'front_answer': answer_front_only,
        'front_rank': rank_front,
        'middle_hit': hit_middle,
        'middle_answer': answer_middle_only,
        'middle_rank': rank_middle,
        'back_hit': hit_back,
        'back_answer': answer_back_only,
        'back_rank': rank_back,
    })

    position_results.append(result)

print("\\nPosition tests complete!")
"""

# Cell: Code - Statistical analysis
"""
print("\\n" + "="*60)
print("STATISTICAL ANALYSIS")
print("="*60)

# Aggregate accuracy
accuracy_front = sum(r['front_hit'] for r in position_results) / len(position_results)
accuracy_middle = sum(r['middle_hit'] for r in position_results) / len(position_results)
accuracy_back = sum(r['back_hit'] for r in position_results) / len(position_results)

print(f"\\nAccuracy by position:")
print(f"  Front:  {accuracy_front:.2%} ({sum(r['front_hit'] for r in position_results)}/{len(position_results)})")
print(f"  Middle: {accuracy_middle:.2%} ({sum(r['middle_hit'] for r in position_results)}/{len(position_results)})")
print(f"  Back:   {accuracy_back:.2%} ({sum(r['back_hit'] for r in position_results)}/{len(position_results)})")

# Paired t-tests
front_hits = [int(r['front_hit']) for r in position_results]
middle_hits = [int(r['middle_hit']) for r in position_results]
back_hits = [int(r['back_hit']) for r in position_results]

t_stat_fm, p_val_fm = ttest_rel(front_hits, middle_hits)
t_stat_fb, p_val_fb = ttest_rel(front_hits, back_hits)
t_stat_mb, p_val_mb = ttest_rel(middle_hits, back_hits)

print(f"\\nPaired t-tests:")
print(f"  Front vs Middle: t={t_stat_fm:.3f}, p={p_val_fm:.4f}")
print(f"  Front vs Back:   t={t_stat_fb:.3f}, p={p_val_fb:.4f}")
print(f"  Middle vs Back:  t={t_stat_mb:.3f}, p={p_val_mb:.4f}")

# State rank analysis
mean_rank_front = np.mean([r['front_rank'] for r in position_results])
mean_rank_middle = np.mean([r['middle_rank'] for r in position_results])
mean_rank_back = np.mean([r['back_rank'] for r in position_results])

print(f"\\nMean effective rank:")
print(f"  Front:  {mean_rank_front:.2f}")
print(f"  Middle: {mean_rank_middle:.2f}")
print(f"  Back:   {mean_rank_back:.2f}")

# Hypothesis conclusion
print(f"\\n" + "="*60)
print("HYPOTHESIS EVALUATION")
print("="*60)

if accuracy_back > accuracy_middle > accuracy_front and p_val_fb < 0.05:
    print("✓ H1 (Recency Bias) SUPPORTED: Back > Middle > Front")
    print("  Interpretation: Recent information overwrites earlier state")
elif abs(accuracy_front - accuracy_back) < 0.1 and p_val_fb > 0.05:
    print("✓ H2 (No Positional Bias) SUPPORTED: Front ≈ Back")
    print("  Interpretation: State preserves all information equally")
else:
    print("✗ Mixed results - neither hypothesis clearly supported")
    print(f"  Accuracy difference (Front-Back): {accuracy_front - accuracy_back:+.2%}")

# Save results
exp4_results = {
    'position_results': position_results,
    'summary': {
        'accuracy_front': float(accuracy_front),
        'accuracy_middle': float(accuracy_middle),
        'accuracy_back': float(accuracy_back),
        'mean_rank_front': float(mean_rank_front),
        'mean_rank_middle': float(mean_rank_middle),
        'mean_rank_back': float(mean_rank_back),
        'ttest_front_vs_middle': {'t': float(t_stat_fm), 'p': float(p_val_fm)},
        'ttest_front_vs_back': {'t': float(t_stat_fb), 'p': float(p_val_fb)},
        'ttest_middle_vs_back': {'t': float(t_stat_mb), 'p': float(p_val_mb)},
    }
}

with open(f"{RESULTS_DIR}/exp4_position_effects.json", 'w') as f:
    # Convert numpy/bool types
    save_dict = {}
    for key, val in exp4_results.items():
        if key == 'position_results':
            save_dict[key] = []
            for r in val:
                r_copy = {}
                for k, v in r.items():
                    if isinstance(v, (bool, np.bool_)):
                        r_copy[k] = bool(v)
                    elif isinstance(v, (np.float64, np.float32)):
                        r_copy[k] = float(v)
                    elif isinstance(v, (np.int64, np.int32)):
                        r_copy[k] = int(v)
                    else:
                        r_copy[k] = v
                save_dict[key].append(r_copy)
        else:
            save_dict[key] = val

    json.dump(save_dict, f, indent=2)

print(f"\\nResults saved to {RESULTS_DIR}/exp4_position_effects.json")
"""

# Cell: Markdown
"""
### 8.6 Visualization
"""

# Cell: Code - Accuracy bar chart
"""
print("\\nGenerating visualizations...")

# Bar chart: Accuracy by position
fig, ax = plt.subplots(figsize=(10, 6))

positions = ['Front', 'Middle', 'Back']
accuracies = [accuracy_front, accuracy_middle, accuracy_back]
colors = ['steelblue', 'orange', 'salmon']

bars = ax.bar(positions, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.02, f'{acc:.1%}',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

# Add significance markers
if p_val_fb < 0.05:
    y_max = max(accuracies) + 0.1
    ax.plot([0, 2], [y_max, y_max], 'k-', linewidth=1.5)
    stars = '***' if p_val_fb < 0.001 else '**' if p_val_fb < 0.01 else '*'
    ax.text(1, y_max + 0.02, stars, ha='center', fontsize=16)

ax.set_ylabel('QA Accuracy', fontsize=12)
ax.set_title(f'Document Position Effect on QA Performance\\n{model_name}', fontsize=14)
ax.set_ylim(0, 1.1)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
save_path = f"{PLOTS_DIR}/exp4_position_accuracy.png"
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"Saved to {save_path}")
plt.show()
"""

# Cell: Code - Per-topic heatmap
"""
# Heatmap: Topics × Positions
accuracy_matrix = np.array([
    [int(r['front_hit']), int(r['middle_hit']), int(r['back_hit'])]
    for r in position_results
])

fig, ax = plt.subplots(figsize=(8, 10))

im = ax.imshow(accuracy_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['Front', 'Middle', 'Back'])
ax.set_yticks(range(len(position_results)))
ax.set_yticklabels([r['topic'] for r in position_results])
ax.set_xlabel('Position', fontsize=12)
ax.set_ylabel('Topic', fontsize=12)
ax.set_title(f'Hit/Miss by Topic and Position\\n{model_name}', fontsize=14)

# Add text annotations
for i in range(len(position_results)):
    for j in range(3):
        text = '✓' if accuracy_matrix[i, j] == 1 else '✗'
        ax.text(j, i, text, ha='center', va='center', color='black', fontsize=14, fontweight='bold')

plt.colorbar(im, ax=ax, label='Hit (1) / Miss (0)')

plt.tight_layout()
save_path = f"{PLOTS_DIR}/exp4_topic_position_heatmap.png"
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"Saved to {save_path}")
plt.show()
"""

# Cell: Code - State rank comparison
"""
# Line plot: State rank by position
fig, ax = plt.subplots(figsize=(10, 6))

rank_data = {
    'Front': [r['front_rank'] for r in position_results],
    'Middle': [r['middle_rank'] for r in position_results],
    'Back': [r['back_rank'] for r in position_results]
}

positions_plot = ['Front', 'Middle', 'Back']
mean_ranks = [np.mean(rank_data[p]) for p in positions_plot]
std_ranks = [np.std(rank_data[p]) for p in positions_plot]

ax.errorbar(positions_plot, mean_ranks, yerr=std_ranks, marker='o', markersize=10,
            linewidth=2, capsize=5, color='steelblue', label='Mean ± Std')

ax.set_xlabel('Document Position', fontsize=12)
ax.set_ylabel('Mean Effective Rank', fontsize=12)
ax.set_title(f'State Rank by Document Position\\n{model_name}', fontsize=14)
ax.grid(alpha=0.3)
ax.legend(fontsize=11)

plt.tight_layout()
save_path = f"{PLOTS_DIR}/exp4_state_rank.png"
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"Saved to {save_path}")
plt.show()
"""

# Cell: Code - Full answer table
"""
print("\\n" + "="*60)
print("FULL ANSWER COMPARISON")
print("="*60)

for r in position_results:
    print(f"\\n### {r['topic']}")
    print(f"**Question**: {r['question']}")
    print(f"**Keywords**: {r['keywords']}")
    print()
    print(f"- **Front**:  {r['front_answer'][:80]}... [{'✓' if r['front_hit'] else '✗'}]")
    print(f"- **Middle**: {r['middle_answer'][:80]}... [{'✓' if r['middle_hit'] else '✗'}]")
    print(f"- **Back**:   {r['back_answer'][:80]}... [{'✓' if r['back_hit'] else '✗'}]")
    print()

print("\\nExperiment 4 complete!")
"""

# Cell: Code - Cleanup
"""
# Clear memory
del model, tokenizer
clear_memory()
print("Model unloaded, memory cleared.")
"""
