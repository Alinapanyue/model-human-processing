# Dual Processing in Language Models: Colors Task Research Summary

**Author:** Yue Pan  
**Date:** December 16, 2025

---

## Executive Summary

This project investigates whether large language models exhibit dual processing patterns similar to human cognition when contextual information conflicts with prior knowledge. Using the logit lens technique to examine layer-by-layer decision-making, we test whether models show evidence of intuitive processing followed by deliberative correction, analogous to System 1 and System 2 thinking in humans.

**Key Findings:**
1. LLMs demonstrate dual processing signatures (Change of Mind and delayed decision times) in conflict scenarios
2. Discovered a U-shaped relationship between model size and dual processing strength in GPT-2 family
3. gpt2-large shows anomalously low dual processing effect despite being mid-sized, suggesting a capacity sweet spot
4. Architectural differences between GPT-2 and Llama families influence dual processing patterns

**Data Location:** All results are in `/data/model_output/logit_lens/` with files named `colors_experiment_*_<model>.csv`

---

## 1. Research Question & Theoretical Framework

### 1.1 Core Question

**Can language models overcome their learned priors when explicitly contradicted by context?**

Example scenario:
```
Context: "I'm looking at a banana. The banana is blue."
Prior knowledge: Bananas are yellow
Question: "What color is the banana?"

Correct answer (from context): blue
Intuitive answer (from prior): yellow
```

### 1.2 Dual Process Theory

Human cognition involves two systems:
- **System 1**: Fast, automatic, intuitive (prior knowledge)
- **System 2**: Slow, deliberative, controlled (context integration)

**Hypothesis**: Neural network layers act as a temporal proxy, with early layers corresponding to System 1 (quick, prior-driven) and late layers corresponding to System 2 (deeper reasoning, context integration).

---

## 2. Methodology

### 2.1 Logit Lens Technique

**Core Idea**: Examine model predictions at **every layer** (not just the final output) to track how decisions evolve.

**Implementation** (in `src/evaluate.py` → `_evaluate_single_item()`):

```python
# For each layer L = 0, 1, 2, ..., n_layers-1:
for layer_idx in range(n_layers):
    # 1. Get token ranks
    rank_blue = rank_of_token("blue", layer=layer_idx)
    rank_yellow = rank_of_token("yellow", layer=layer_idx)
    
    # 2. Get log probabilities
    logprob_blue = log_probability("blue", layer=layer_idx)
    logprob_yellow = log_probability("yellow", layer=layer_idx)
    
    # 3. Compute preference direction
    logprob_diff = logprob_blue - logprob_yellow
    # If logprob_diff > 0: Model prefers "blue" (correct)
    # If logprob_diff < 0: Model prefers "yellow" (intuitive/incorrect)
```

### 2.2 Key Metrics: CoM and TTD

#### **2.2.1 CoM (Change of Mind) - Magnitude of Cognitive Conflict**

**Definition**: The "distance" the model travels from its most incorrect state to its final decision.

**Formula** (computed in `analysis/notebooks/0_process_lm_data.ipynb`):

```python
min_diff = min(logprob_diff across all layers)
final_diff = logprob_diff at final layer

if min_diff < 0:  # Model was initially wrong
    CoM = min(0, final_diff) - min_diff
else:  # Model was correct from the start
    CoM = 0
```

**Interpretation**:
- **High CoM (e.g., 2.5)**: Strong initial intuition toward wrong answer, later corrected
  - Layer 5: logprob_diff = -2.0 (strongly prefers "yellow")
  - Layer 35: logprob_diff = +0.5 (corrects to "blue")
  - CoM = 0 - (-2.0) = 2.0

- **Low CoM (e.g., 0.2)**: Weak initial error, quickly resolved
  - Layer 2: logprob_diff = -0.2 (slightly prefers "yellow")
  - Layer 8: logprob_diff = +1.5 (strongly prefers "blue")
  - CoM = 0 - (-0.2) = 0.2

- **Zero CoM**: Never preferred incorrect answer
  - All layers: logprob_diff > 0 (always preferred "blue")

**What it measures**: The **strength of the intuitive response** that must be overcome.

#### **2.2.2 TTD (Time to Decision) - When the Model "Commits"**

**Definition**: The layer index where the model makes its final switch from incorrect to correct (normalized by total layers).

**Formula**:

```python
# Find the last layer where model still prefers incorrect answer
last_negative_layer = max(layer where logprob_diff < 0)

# Decision happens at the next layer
TTD = (last_negative_layer + 1) / total_layers
```

**Interpretation**:
- **High TTD (e.g., 0.8)**: Decision made late (layer 29 out of 36)
  - Model "struggled" to override intuition
  
- **Low TTD (e.g., 0.2)**: Decision made early (layer 7 out of 36)
  - Model quickly resolved the conflict

**What it measures**: How **long** (in computational depth) the model takes to commit to the correct answer.

#### **2.2.3 What CoM and TTD Represent for Colors Task**

For the colors task specifically:

| Metric | Cognitive Process | Example Trajectory |
|--------|------------------|-------------------|
| **CoM** | **Strength of prior knowledge interference** | "Yellow" is deeply entrenched → high CoM needed to overcome it |
| **TTD** | **Depth of processing required** | Simple override → low TTD; complex integration → high TTD |

**Competitor condition** (intervening facts present):
- **Prediction**: Higher CoM and TTD
- **Reason**: Intervening facts → working memory load → harder to recall "blue"

**NoCompetitor condition** (no intervening facts):
- **Prediction**: Lower CoM and TTD (but not zero, due to prior knowledge conflict)

---

## 3. Experimental Design

### 3.1 Stimuli

**30 entities** with conflicting color assignments:
- **Food items** (15): banana, lemon, tomato, strawberry, carrot, spinach, etc.
- **Animals** (15): flamingo, giraffe, elephant, tiger, frog, raven, etc.

**Structure per entity**:
```
Prefix: "I'm looking at a [ENTITY]."
Critical fact: "The [ENTITY] is blue."  ← Always "blue"
[INTERVENING FACTS]  ← Experimental manipulation
Question: "What color is the [ENTITY]?"

Correct answer: "blue" (from context)
Intuitive answer: varies by entity (e.g., "yellow" for banana)
```

### 3.2 Intervening Facts Design

**5 fact types** × **2 styles** = 10 fact variations per entity:

| Fact Type | Normal (reinforces prior) | Strange (violates prior) |
|-----------|--------------------------|-------------------------|
| **Appearance** | "The banana is long." | "The banana is square." |
| **Type** | "The banana is a plant." | "The banana is an animal." |
| **Subtype** | "The banana is a fruit." | "The banana is a mammal." |
| **Place** | "The banana grows on a tree." | "The banana grows on the moon." |
| **Size** | "The banana is bigger than an ant." | "The banana is smaller than an ant." |

**Hypothesis**:
- **Normal facts** → activate prior knowledge → stronger intuitive response → higher CoM/TTD
- **Strange facts** → weaken prior activation → weaker intuitive response → lower CoM/TTD

---

## 4. Experiments & Results

### **Experiment 1: Dose-Response of Fact Number**

#### **Setup** (implemented in `src/utils.py` → `get_conditions_for_color_experiment_1()`)

**Research Question**: Does dual processing increase linearly with the number of intervening facts?

**Conditions per entity: 11**
- **Baseline**: 0 intervening facts
- **Normal-1 to Normal-5**: 1, 2, 3, 4, 5 normal facts (in order: appearance → type → subtype → place → size)
- **Strange-1 to Strange-5**: 1, 2, 3, 4, 5 strange facts (same order)

**Total items**: 30 entities × 11 conditions = **330 test items per model**

**Models tested**:
- GPT-2 family: `gpt2` (124M), `gpt2-medium` (355M), `gpt2-large` (774M), `gpt2-xl` (1.5B)
- Llama-2 family: `Llama-2-7b-hf`, `Llama-2-13b-hf`
- Llama-3.2 family: `Llama-3.2-1B`, `Llama-3.2-3B`

#### **Hypotheses**

1. **H1**: CoM and TTD increase with number of facts (working memory load)
2. **H2**: Normal facts produce higher CoM/TTD than strange facts (prior activation)
3. **H3**: The normal-strange difference amplifies with more facts (interaction effect)

#### **Results** (from `1_experiment1.ipynb`, Cell 10-11)

**Finding 1: Minimal effect of fact number**
```
Baseline (0 facts): CoM = 1.611
1 fact:            CoM = 1.668
2 facts:           CoM = 1.682
3 facts:           CoM = 1.667
4 facts:           CoM = 1.628
5 facts:           CoM = 1.601
```
→ **No linear increase observed**; slight decrease at 5 facts

**Finding 2: No systematic normal vs. strange difference**
```
                Normal      Strange
1 fact:         1.667       1.669
2 facts:        1.681       1.682
3 facts:        1.675       1.660
4 facts:        1.639       1.618
5 facts:        1.607       1.594
```
→ Differences are minimal (< 0.05 CoM units)

**Finding 3: Model-specific patterns**

**GPT-2 family**:
- `gpt2`: Stable CoM across fact numbers (~1.7)
- `gpt2-medium`: **Decreasing** CoM with more facts (1.66 → 0.95)
- `gpt2-large`: **Anomaly** - extremely low CoM (~0.24) at all levels
- `gpt2-xl`: Slight increase (0.45 → 0.60)

**Llama family**:
- `Llama-3.2-1B`: Increasing trend (2.52 → 2.78)
- `Llama-3.2-3B`: Stable/increasing (1.92 → 2.08)
- `Llama-2-7b-hf`: Slight decrease (2.73 → 2.56)
- `Llama-2-13b-hf`: **Increasing** (1.70 → 1.99)

#### **Interpretation**

**H1 (Linear increase) REJECTED**: Most models show flat or decreasing CoM
- **Possible explanation**: Models may "saturate" after 1-2 facts; additional facts don't add interference

**H2 (Normal > Strange) REJECTED**: No consistent difference
- **Possible explanation**: Our "strange" facts may not be strange enough, or models treat all facts as equally distracting

**H3 (Interaction) NOT SUPPORTED**: No amplification effect observed

**Key discovery**: Model size effects dominate over experimental manipulations

---

### **Experiment 2: Model Family Comparison**

#### **Setup** (analyzed in `1_experiment1.ipynb`, Cell 12)

**Research Question**: Is the U-shaped capacity curve specific to GPT-2 architecture, or does it generalize to Llama?

**Model pairs at matched scales** (approximately):

| GPT-2 Model | Size | Llama Model | Size | Ratio |
|-------------|------|-------------|------|-------|
| gpt2 | 124M | Llama-3.2-1B | 1B | 1:8 |
| gpt2-medium | 355M | Llama-3.2-3B | 3B | 1:8.5 |
| gpt2-large | 774M | Llama-2-7b-hf | 7B | 1:9 |
| gpt2-xl | 1.5B | Llama-2-13b-hf | 13B | 1:8.7 |

**Test condition**: 5 normal facts (maximum interference)

#### **Results**

**Cross-family comparison**:

```
GPT-2 Family (5 normal facts):
  gpt2       (124M):  CoM = 1.612
  gpt2-medium (355M):  CoM = 0.939   ↓ (decline)
  gpt2-large  (774M):  CoM = 0.203   ↓↓ (anomaly!)
  gpt2-xl    (1.5B):   CoM = 0.564   ↑ (recovery)
  → U-SHAPED CURVE

Llama Family (5 normal facts):
  Llama-3.2-1B (1B):    CoM = 2.866
  Llama-3.2-3B (3B):    CoM = 2.148   ↓
  Llama-2-7b-hf (7B):   CoM = 2.515   ↑
  Llama-2-13b-hf (13B): CoM = 1.990   ↓
  → LESS PRONOUNCED U-SHAPE (or non-monotonic)
```

**Pair-wise comparisons** (Llama vs GPT-2 at matched scales):

```
124M vs 1B:   GPT-2 = 1.61,  Llama = 2.87  (1.78× higher)
355M vs 3B:   GPT-2 = 0.94,  Llama = 2.15  (2.29× higher)
774M vs 7B:   GPT-2 = 0.20,  Llama = 2.51  (12.4× higher!) ← Dramatic
1.5B vs 13B:  GPT-2 = 0.56,  Llama = 1.99  (3.55× higher)
```

#### **Interpretation**

**Finding 1: U-shaped curve is GPT-2 specific**
- GPT-2: Clear minimum at gpt2-large (774M)
- Llama: No clear minimum; more monotonic decline

**Finding 2: Architectural differences dominate**
- Even at "matched" parameter counts, Llama shows 2-12× higher CoM
- Suggests Llama models experience stronger intuitive responses or weaker context integration

**Finding 3: gpt2-large anomaly is robust**
- CoM = 0.203 (lowest across all models)
- **Hypothesis**: This model size hits a "sweet spot" for:
  - Sufficient capacity to recognize the task pattern
  - Not overly sensitive to prior knowledge activation
  - Efficient at shortcut learning: "just extract the explicitly stated color"

**Scientific implications**:
- Model capacity effects are **non-linear** and **architecture-dependent**
- Scaling laws for dual processing differ from scaling laws for general performance
- Optimal model size for specific cognitive tasks may not be "bigger is better"

---

### **Experiment 3: Fact Type Breakdown**

#### **Setup** (implemented in `src/utils.py` → `get_conditions_for_color_experiment_3()`)

**Research Question**: Do different semantic fact types produce different interference strengths?

**Conditions per entity: 10**
- 5 fact types × 2 styles (normal/strange)
- All single-fact conditions (isolates individual fact type effects)

**Analysis goal**: Rank fact types by interference strength

#### **Results** (from `1_experiment1.ipynb`, Cell 13)

**Fact type ranking by CoM (higher = more interference)**:

```
Fact Type       Normal      Strange     Difference
─────────────────────────────────────────────────
Subtype         1.699       1.662       +0.036
Type            1.682       1.668       +0.014
Size            1.679       1.668       +0.011
Place           1.673       1.669       +0.004
Appearance      1.667       1.669       -0.002
```

**Statistical tests**: All normal vs. strange comparisons are non-significant (p > 0.05)

#### **Interpretation**

**Finding 1: Minimal variation between fact types**
- All fact types produce CoM ≈ 1.67 (very tight range: 1.667-1.699)
- Suggests interference is **content-independent**

**Finding 2: No normal vs. strange effect**
- Differences are negligible (< 0.04)
- Even negative for "appearance" (opposite of prediction)

**Finding 3: Subtype shows slight edge (non-significant)**
- "The banana is a fruit" (0.032 higher CoM than appearance)
- But this is likely noise given tight confidence intervals

**Conclusion**: 
- Dual processing in this task is driven by **generic interference** (working memory load), not semantic content
- Any single fact produces ~equivalent distraction
- The "strangeness" manipulation was insufficient to create meaningful differences

---

## 5. Overall Conclusions

### 5.1 Main Contributions

1. **Validated logit lens as a tool for studying dual processing in LLMs**
   - Successfully captured layer-by-layer decision trajectories
   - CoM and TTD metrics show interpretable patterns

2. **Discovered non-linear scaling effects**
   - U-shaped curve in GPT-2 family challenges "bigger is better" assumption
   - gpt2-large (774M) shows anomalous behavior worth further investigation

3. **Architectural differences in dual processing**
   - Llama models show stronger dual processing signatures than GPT-2
   - Parameter count alone insufficient to predict dual processing behavior

4. **Limited effect of experimental manipulations**
   - Number of facts (1-5): No clear dose-response
   - Fact style (normal vs. strange): No significant difference
   - Fact type (appearance, type, etc.): No meaningful variation

### 5.2 Limitations

1. **Experimental manipulations may be too weak**
   - "Strange" facts not strange enough (models may ignore them)
   - Need more extreme interventions (e.g., "banana is size of a planet")

2. **Ceiling effects**
   - Even 1 fact may be sufficient to disrupt working memory
   - May need 0 vs. 1 fact as the critical comparison

3. **Limited sample size**
   - Only 30 entities (may lack power to detect small effects)
   - Could expand to 100+ entities for more robust conclusions

4. **Confounds in cross-family comparison**
   - Llama models trained on different data than GPT-2
   - Architectural differences (attention, normalization) not isolated

### 5.3 Future Directions

Three key research directions emerge from this work. First, mechanistic investigation of the gpt2-large anomaly through attention pattern analysis and circuit-level interpretability could reveal whether this model has learned shortcut strategies that bypass dual processing. Second, stronger experimental manipulations are needed, including more extreme strange facts and larger numbers of intervening facts (10-50), to test whether the null results for fact number and type reflect ceiling effects or genuine insensitivity to these manipulations. Third, cross-task validation should examine whether the U-shaped capacity curve generalizes to other dual-processing paradigms such as syllogistic reasoning and Stroop-like tasks, which would establish whether gpt2-large's efficiency is task-specific or reflects a general cognitive architecture advantage at this parameter scale.

---

## 6. Code Organization & File Locations

### 6.1 Main Experiment Scripts

| File | Purpose | Key Functions |
|------|---------|--------------|
| `src/run_experiment.py` | Main entry point for running experiments | `main()` - orchestrates model evaluation |
| `src/evaluate.py` | Core evaluation logic | `evaluate()` - runs full stimulus set, `_evaluate_single_item()` - logit lens extraction |
| `src/model.py` | Model initialization and inference | `initialize_lm()`, `rank_of_token_all_layers()`, `conditional_score_all_layers()` |
| `src/utils.py` | Experiment condition generators | `get_conditions_for_color_experiment_1()`, `get_conditions_for_color_experiment_3()` |

### 6.2 Analysis Scripts

| File | Purpose |
|------|---------|
| `analysis/notebooks/0_process_lm_data.ipynb` | Reads raw layer-wise CSVs, computes CoM/TTD metrics, saves processed data |
| `analysis/notebooks/1_experiment1.ipynb` | Main analysis notebook with Experiments 1-3 visualizations and statistics |
| `analysis/notebooks/utils.py` | Helper functions for plotting and metric definitions |

### 6.3 Data Files

**Raw outputs** (in `/data/model_output/logit_lens/`):
- `colors_experiment_1_<model>.csv` - Experiment 1 raw data (layer-wise logits)
- `colors_experiment_3_<model>.csv` - Experiment 3 raw data

**Processed outputs** (in `/data/model_output/processed/`):
- `colors_metrics_logit_lens.csv` - Aggregated metrics (one row per item with CoM, TTD, etc.)

