# Colors Task Analysis

## Overview

This directory contains analysis scripts for the "colors task" experiment, which tests how language models handle conflicts between contextual information and prior knowledge under cognitive load.

## Experiment Design

The task presents models with counter-intuitive facts (e.g., "The banana is blue") followed by a question ("What color is the banana?"). The key manipulations are:

1. **Number of intervening facts** (0, 1, or 5): Testing whether additional content between the critical fact and question causes models to forget the context and rely on prior knowledge
2. **Type of intervening facts** (normal, strange, or mixed): Testing whether facts consistent with prior knowledge ("The banana is long") create more interference than strange facts ("The banana is square")

## Models Tested

We evaluated four models spanning different scales:

- **GPT-2** (117M parameters, 12 layers)
- **GPT-2-XL** (1.5B parameters, 48 layers)
- **Llama-3.2-3B-Instruct** (3B parameters, 28 layers)
- **Llama-3.1-8B-Instruct** (8B parameters, 32 layers)

## Analysis Scripts

### `gpt2.py`
Single-model analysis of GPT-2 results. Shows:
- Overall accuracy in remembering contextual information vs. falling back to priors
- Effect of number and type of intervening facts
- Layer-wise development of the correct response

### `compare.py`
Cross-model comparison across all four models. Highlights:
- How model scale affects resistance to interference
- Performance degradation patterns under cognitive load
- Scaling thresholds for maintaining contextual information

## Key Findings

### 1. Clear Scaling Effect on Working Memory

**Performance under interference (5 intervening facts):**
- GPT-2 (117M): 50.0% accuracy (30 pp drop from baseline)
- GPT-2-XL (1.5B): 100.0% accuracy (0 pp drop)
- Llama-3.2 (3B): 96.7% accuracy (3.3 pp drop)
- Llama-3.1 (8B): 100.0% accuracy (0 pp drop)

The results show a dramatic transition: GPT-2 loses 30 percentage points when faced with 5 intervening facts, exhibiting human-like working memory limitations. In contrast, models at 1.5B+ parameters maintain near-perfect accuracy regardless of interference.

**Detailed Performance Breakdown:**
```
GPT-2 (117M):     0 facts: 80.0% → 5 facts: 50.0% (30 pp drop)
GPT-2-XL (1.5B):  0 facts: 100.0% → 5 facts: 100.0% (0 pp drop)
Llama-3.2 (3B):   0 facts: 100.0% → 5 facts: 96.7% (3.3 pp drop)
Llama-3.1 (8B):   0 facts: 100.0% → 5 facts: 100.0% (0 pp drop)
```

### 2. Scaling Threshold Between 117M and 1.5B

The critical transition occurs between GPT-2 (117M) and GPT-2-XL (1.5B). Above this threshold, models demonstrate "superhuman" working memory capacity, maintaining contextual information even under cognitive load that would overwhelm both humans and smaller models.

### 3. Fact Type Effects Emerge Only in Small Models

**With 5 intervening facts:**
- **GPT-2**: Normal facts (46.7%) vs. Strange facts (53.3%) - 6.6 pp difference
- **All larger models**: No difference between fact types (all ~97-100%)

Normal facts (consistent with prior knowledge like "The banana is long") create more interference than strange facts ("The banana is square") in GPT-2, confirming the hypothesis about prior knowledge reinforcement. However, this effect completely disappears in larger models.

**Fact Type Performance Details:**
```
GPT-2 (117M):     all_normal: 46.7%, all_strange: 53.3%, mixed: 50.0%
GPT-2-XL (1.5B):  all_normal: 100.0%, all_strange: 100.0%, mixed: 100.0%
Llama-3.2 (3B):   all_normal: 96.7%, all_strange: 96.7%, mixed: 96.7%
Llama-3.1 (8B):   all_normal: 100.0%, all_strange: 100.0%, mixed: 100.0%
```

### 4. Layer-wise Processing Differs by Scale

**Accuracy gains from first to final layer:**
- GPT-2 (12 layers): 35.0% → 67.6% (32.6 pp gain)
- GPT-2-XL (48 layers): 22.4% → 100.0% (77.6 pp gain)
- Llama-3.2 (28 layers): 6.4% → 98.3% (91.9 pp gain)
- Llama-3.1 (32 layers): 47.9% → 100.0% (52.1 pp gain)

Larger models show more dramatic improvements across layers, suggesting they have greater capacity to integrate contextual information and override initial intuitive biases.

**Layer-wise Development Patterns:**
```
GPT-2:      Layer 0: 35.0% → Layer 11: 67.6% (32.6 pp gain)
GPT-2-XL:   Layer 0: 22.4% → Layer 47: 100.0% (77.6 pp gain)
Llama-3.2:  Layer 0: 6.4% → Layer 27: 98.3% (91.9 pp gain)
Llama-3.1:  Layer 0: 47.9% → Layer 31: 100.0% (52.1 pp gain)
```

## Running the Analysis

From this directory:

```bash
# Analyze GPT-2 results
python gpt2.py

# Compare all four models
python compare.py
```

## Data Location

Results are stored in:
```
../../data/model_output/logit_lens/colors_*.csv
```

Each CSV contains layer-by-layer logit lens analysis with columns for:
- `num_intervening_facts`: 0, 1, or 5
- `fact_type_condition`: none, individual fact types, all_normal, all_strange, mixed
- `mean_logprob_response_isCorrect`: Whether the model preferred the correct (contextual) vs. intuitive (prior) answer
- `layer_idx`: Layer of the model where the prediction was extracted

## Implications

### For Understanding Model Cognition

1. **Dual-process behavior in small models**: GPT-2's performance (67.6% overall, dropping to 50% under load) mirrors human dual-process cognition, showing tension between intuitive priors and contextual reasoning.

2. **Resource limitations explain small model behavior**: The 30 pp performance drop in GPT-2 with intervening facts directly parallels human working memory constraints. The model isn't "learning incorrectly"—it's operating under resource constraints.

3. **Scale creates qualitatively different systems**: The jump from 117M to 1.5B parameters isn't just quantitative improvement—it's a qualitative shift from human-like limitations to superhuman capacity.

### For Scaling Laws

1. **Sharp transition rather than smooth scaling**: The improvement isn't gradual. There's a critical threshold around 1B parameters where models transition from vulnerable to robust under cognitive load.

2. **Diminishing returns on fact-type effects**: The sophisticated distinction between normal and strange facts that matters in GPT-2 becomes irrelevant in larger models, which simply maintain both context and priors simultaneously.

### For AI Safety and Capabilities

1. **Small models are more "human-like" in failure modes**: GPT-2's failure pattern (forgetting context, relying on priors) is more predictable and human-interpretable than perfect performance.

2. **Larger models overcome human intuitions about cognitive load**: What should be "harder" (5 facts vs 0 facts) makes no difference to models above the scaling threshold, suggesting they use fundamentally different processing strategies than humans.

## Experimental Output Summary

The analysis scripts generate detailed performance metrics for each model:

### **Overall Accuracy (Final Layer)**
- **GPT-2**: 67.6% (shows human-like limitations)
- **GPT-2-XL**: 100.0% (perfect performance)
- **Llama-3.2**: 98.3% (near-perfect performance)
- **Llama-3.1**: 100.0% (perfect performance)

### **Key Performance Patterns**
1. **Cognitive Load Effect**: Only GPT-2 shows significant degradation (30 pp drop) with 5 intervening facts
2. **Fact Type Sensitivity**: Only GPT-2 distinguishes between normal vs. strange facts (6.6 pp difference)
3. **Layer Processing**: Larger models show more dramatic accuracy gains across layers (up to 91.9 pp improvement)

### **Interpretation**
The results demonstrate a clear **scaling threshold** around 1.5B parameters where models transition from human-like cognitive limitations to superhuman working memory capacity. This suggests that model scale fundamentally changes cognitive processing strategies, with smaller models exhibiting predictable, human-interpretable failure modes while larger models achieve robust performance under cognitive load.

