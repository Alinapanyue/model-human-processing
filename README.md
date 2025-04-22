# Linking forward-pass dynamics in Transformers and real-time human processing

This repository contains code and data for the preprint 
["Linking forward-pass dynamics in Transformers and real-time human processing"](https://arxiv.org/abs/2504.14107)
by Jennifer Hu, Michael Lepori, and Michael Franke. **This is a work in progress.**

## Evaluating models

### Language-based tasks (Study 1a, 1b, 2, 3)

To evaluate a model on one of the language-based tasks, simply run
```bash
bash scripts/run_experiment.sh <MODEL> <TASK>
```

In our experiments, we used `<MODEL> = meta-llama/Llama-2-7b-hf`.
`<TASK>` should be one of the following:
- `capitals-recall`: Study 1a
- `capitals-recognition`: Study 1b
- `animals`: Study 2
- `syllogism`: Study 3

This helper script calls `src/run_experiment.py` under the hood. Please see
the Python file for more details and additional command-line options.

### Vision tasks (Study 4)

The code and data for running the vision tasks (Study 4) is contained in the
`src/vision` folder, and largely adapted from 
[bethgelab/model-vs-human](https://github.com/bethgelab/model-vs-human).

### Analyses

The notebook `analysis/notebooks/process_lm_data.ipynb` is used
to process model data for the language-based tasks. This reads the raw model
output files from `data/model_output`, combines it with the anonymized
trial-level human data from `data/human`, and then saves combined data to
`data/human_model_combined`.

Analogously, the notebook `analysis/notebooks/process_vision_data.ipynb`
does the same for model outputs and human data for the vision tasks.

R scripts for fitting mixed-effects regression models and performing model 
comparison can be found in the `analysis/r_scripts` folder. The results of 
model comparisons are visualized using the notebook 
`analysis/notebooks/compare_regression_models.ipynb`.