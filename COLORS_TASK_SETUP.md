# Colors Task Implementation Summary

## What We Accomplished

We've successfully implemented a new experimental task called **"colors"** to test whether language models can override their prior knowledge when given conflicting contextual information.

---

## The Research Question

**Can models remember contextual information that conflicts with their prior knowledge when there's intervening content?**

### Example:
- **Context**: "I'm looking at a banana. The banana is blue."
- **Question**: "What color is the banana?"
- **Correct answer**: "blue" (from context)
- **Intuitive answer**: "yellow" (from prior knowledge)

### Key Manipulations:
1. **Number of intervening facts** (0-5 facts between critical info and question)
   - Hypothesis: More facts → model forgets critical info → prefers intuitive answer
   
2. **Type of intervening facts** (normal vs. strange)
   - **Normal facts**: "The banana is long." (consistent with prior knowledge)
     - Hypothesis: Reinforces prior → more likely to answer "yellow"
   - **Strange facts**: "The banana is square." (conflicts with prior knowledge)
     - Hypothesis: Less reinforcement → less bias toward "yellow"

---

## What We Implemented

### 1. **Added "colors" to TASKS list** (`src/utils.py`)
The colors task is now recognized as a valid experiment type.

### 2. **Created `get_conditions_for_color_experiment()` function** (`src/utils.py`)
This function generates multiple experimental conditions for each item:

#### Conditions Generated (per item):
- **Baseline**: No intervening facts (prefix + critical fact + question)
- **1 fact conditions**: Each of 5 fact types × 2 styles (normal/strange) = 10 conditions
  - appearance_normal, appearance_strange
  - type_normal, type_strange
  - subtype_normal, subtype_strange
  - place_normal, place_strange
  - size_normal, size_strange
- **All 5 normal facts**: All normal facts together
- **All 5 strange facts**: All strange facts together
- **Mixed condition**: Alternating normal and strange facts

**Total: ~14 conditions per item**

### 3. **Created colors.csv stimuli file** (`data/stimuli/colors.csv`)
20 items with various entities:
- **Plants/Food**: banana, lemon, tomato, strawberry, pumpkin, carrot, spinach, lime, potato, eggplant
- **Animals**: flamingo, pig, elephant, dolphin, zebra, polar bear, tiger, frog, crow
- **Objects**: snow

Each item has:
- All items use "blue" as the correct (contextual) answer
- Different intuitive colors based on prior knowledge (yellow, red, orange, green, pink, gray, black, white, purple, brown)

### 4. **Updated imports** (`src/evaluate.py`)
The evaluation script now imports and uses `get_conditions_for_color_experiment`.

---

## How to Run the Experiment

### Basic Usage:
```bash
bash scripts/run_experiment.sh <MODEL> colors
```

### Examples:
```bash
# Run with GPT-2
bash scripts/run_experiment.sh gpt2 colors

# Run with Llama-2-7b
bash scripts/run_experiment.sh meta-llama/Llama-2-7b-hf colors

# Run with reduced precision (to save memory)
python src/run_experiment.py --model gpt2 --task colors --reduce_precision

# Run with tuned lens instead of logit lens
python src/run_experiment.py --model gpt2 --task colors --use_tuned_lens
```

---

## Output Structure

Results will be saved to: `data/model_output/logit_lens/colors_<model-name>.csv`

### Output columns include:
- **Stimulus metadata**: task, item_id, entity, correct, intuitive, prefix, question
- **Condition metadata**: num_intervening_facts, fact_type_condition, query
- **Layer-by-layer results** (for each layer 0 to L-1):
  - `layer_idx`: Current layer number
  - `rank_correct_first_token`: Rank of correct answer's first token
  - `rank_incorrect_first_token`: Rank of incorrect answer's first token
  - `mean_logprob_correct`, `mean_logprob_incorrect`: Log probabilities
  - `mean_logprob_response`: Model's predicted answer ("correct" or "incorrect")
  - `mean_logprob_response_isCorrect`: Whether prediction is correct (True/False)
  - Similar metrics for `sum_logprob` and `first_logprob`

---

## Expected Results & Analysis

### What to look for:

1. **Baseline performance** (0 intervening facts)
   - Do models prefer "blue" (context) or "yellow" (prior)?
   - At which layer does the correct answer emerge?

2. **Effect of number of facts**
   - Does accuracy decrease as we add more intervening facts?
   - Plot: accuracy vs. number of intervening facts

3. **Effect of fact type**
   - Do normal facts hurt performance more than strange facts?
   - Compare: 5 normal facts vs. 5 strange facts vs. mixed

4. **Layer-wise analysis**
   - At early layers: might prefer intuitive answer
   - At later layers: might incorporate context better
   - Does this pattern change with intervening facts?

### Potential analysis notebook structure:
```python
# Load results
df = pd.read_csv('data/model_output/logit_lens/colors_gpt2.csv')

# Filter to final layer
final_layer = df[df['layer_idx'] == df['layer_idx'].max()]

# Accuracy by number of intervening facts
accuracy_by_num_facts = final_layer.groupby('num_intervening_facts')['mean_logprob_response_isCorrect'].mean()

# Accuracy by fact type (controlling for number)
accuracy_by_type = final_layer[final_layer['num_intervening_facts'] == 5].groupby('fact_type_condition')['mean_logprob_response_isCorrect'].mean()

# Layer-wise trajectory
layer_accuracy = df.groupby(['layer_idx', 'num_intervening_facts'])['mean_logprob_response_isCorrect'].mean()
```

---

## Next Steps

### Immediate:
1. ✅ **Set up cluster access** (follow DSAI documentation)
2. ✅ **Test with a small model locally** (e.g., `gpt2`)
   ```bash
   bash scripts/run_experiment.sh gpt2 colors
   ```
3. **Verify output format** looks correct

### After initial testing:
4. **Run on multiple models** (GPT-2, GPT-2-XL, Llama-2-7b, Llama-2-13b)
5. **Create analysis notebook** (follow pattern from existing notebooks)
6. **Compare with other tasks** (e.g., animals task has similar structure)

### Optional extensions:
- Add more items (currently 20 items)
- Test different colors for critical fact (currently all "blue")
- Test graduated number of facts (2, 3, 4 facts systematically)
- Add human behavioral data for comparison

---

## Code Location Summary

| File | Changes |
|------|---------|
| `src/utils.py` | Added "colors" to TASKS; added `get_conditions_for_color_experiment()` |
| `src/evaluate.py` | Added import for `get_conditions_for_color_experiment` |
| `data/stimuli/colors.csv` | Created new stimuli file with 20 items |
| `scripts/run_experiment.sh` | No changes needed (already generic) |

---

## Questions to Discuss with Jennifer

1. Is the number of conditions (~14 per item × 20 items = ~280 conditions) reasonable?
2. Should we include more gradual conditions (e.g., 2, 3, 4 facts separately)?
3. Should we vary the correct color (currently all "blue")?
4. What models should we prioritize for the initial run?
5. Are there specific analysis plots she'd like to see?

---

## Technical Notes

- The implementation follows the same pattern as `capitals-recognition` task (which also uses multiple conditions per item)
- Each condition gets evaluated at every layer of the model
- Results are saved in long format (one row per layer per condition)
- Compatible with both logit lens and tuned lens approaches

