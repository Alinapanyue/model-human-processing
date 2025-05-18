#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# load utils   ----
source("analysis/r_scripts/utils.R")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# define global variables: task and model predictor lists  ----

TASK <- "vision"
process_metrics <- list(
  "auc_entropy",
  "layer_biggest_change_entropy",
  "auc_rank_correct",
  "layer_biggest_change_rank_correct",
  "auc_logprob_correct",
  "layer_biggest_change_logprob_correct"
)
iv_list <- c(
  list("baseline_final", "baseline_midpoint"),
  process_metrics
)
DATASETS <- c(
    "colour",
    "contrast",
    "cue-conflict",
    "edge",
    "eidolonI",
    "eidolonII",
    "eidolonIII",
    "false-colour",
    "high-pass",
    "low-pass",
    "phase-scrambling",
    "power-equalisation",
    "rotation",
    "silhouette",
    "sketch",
    "stylized",
    "uniform-noise"
)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# full model fits for each human DV, dataset, and model ----

models <- c("vit_small_patch16_224", "vit_base_patch16_224")
for (i in 1:length(models)) {
  model <- models[[i]]
  message(sprintf("====== Running all model comparisons for %s ======", model))
  for (i in 1:length(DATASETS)) {
    DATASET <- DATASETS[[i]]
    message(DATASET)
    
    # Read data for this dataset.
    df <- load_vision_data(
      sprintf("data/human_model_combined/logit_lens/vision_%s_%s.csv", DATASET, model)  
    )
    if (!(DATASET %in% c("cue-conflict", "edge", "silhouette", "sketch", "stylized"))) {
      df <- mutate(df, condition = as.factor(condition))
    }

    # ACCURACY
    run_model_comparison(
      df,
      "response_correct",
      iv_list,
      TASK,
      "all_trials",
      model,
      family=binomial(link="logit"), 
      use_glmer=T,
      file_suffix=DATASET,
      vision_dataset=DATASET
    )
    
    # RESPONSE TIME (only analyze correct trials)
    df_filtered <- filter(df, response_correct==1)
    run_model_comparison(
      df_filtered, 
      "rt", 
      iv_list, 
      TASK,
      "filteredCorrect",
      model,
      log_y=T,
      file_suffix=DATASET,
      vision_dataset=DATASET
    )
  }
}
