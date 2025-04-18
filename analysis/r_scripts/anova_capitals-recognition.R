#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# load utils   ----
source("analysis/r_scripts/anova_utils.R")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# define global variables: task and model predictor lists  ----

TASK <- "capitals-recognition"
iv_list <- c(
  list("baseline"),
  process_metrics # defined from anova_utils.R
  # control_metrics # defined from anova_utils.R
)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# helper functions  ----

load_capitals_recognition_data <- function(model) {
  df <- load_data(
    sprintf("data/human_model_combined/%s_%s.csv", TASK, model)
  )
  return(df)
}

run_all_model_comparisons <- function(df, model) {
  file_prefix <- model
  # accuracy DV on full dataset (all trials)
  ALL_TRIALS_SUBSET <- "all_trials"
  run_model_comparison(
    df, 
    "response_correct", 
    iv_list, 
    TASK,
    family=binomial(link="logit"), 
    use_glmer=T,
    data_subset=ALL_TRIALS_SUBSET,
    file_prefix=file_prefix
  )

  # use filtered data for the other DVs
  CORRECT_TRIALS_SUBSET <- "filteredCorrect"
  df_filtered <- filter(df, response_correct==1)

  run_model_comparison(
    df_filtered, 
    "rt", 
    iv_list, 
    TASK,
    log_y=T,
    data_subset=CORRECT_TRIALS_SUBSET,
    file_prefix=file_prefix
  )
  run_model_comparison(
    df_filtered, 
    "rt_zscore", 
    iv_list, 
    TASK,
    data_subset=CORRECT_TRIALS_SUBSET,
    file_prefix=file_prefix
  )
}

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# run comparisons for each LM  ----

models <- c("Llama-2-7b-hf", "Llama-2-13b-hf", "Llama-2-70b-hf")
for (i in 1:length(models)) {
  model <- models[[i]]
  message(sprintf("====== Running all model comparisons for %s ======", model))
  df <- load_capitals_recognition_data(model)
  run_all_model_comparisons(df, model)
}