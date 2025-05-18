#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# load utils   ----
source("analysis/r_scripts/utils.R")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# define global variables: task and model predictor lists  ----

TASK <- "animals"
LENS <- "logit_lens"

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# helper functions  ----

load_animals_data <- function(model) {
  df <- load_data(
    sprintf("data/human_model_combined/%s/%s_%s.csv", LENS, TASK, model)
  ) %>% mutate(subject_id = as.factor(subject_nr))
  return(df)
}

run_all_model_comparisons <- function(df, model) {
  # check if any columns are entirely NaN
  process_metrics <- get_nonNan_metrics(PROCESS_METRICS, df)
  iv_list <- c(
    list("baseline_final", "baseline_midpoint"),
    process_metrics
  )
  
  # accuracy DV on full dataset (all trials)
  ALL_TRIALS_SUBSET <- "all_trials"
  run_model_comparison(
    df,
    "response_correct",
    iv_list,
    TASK,
    ALL_TRIALS_SUBSET,
    model,
    family=binomial(link="logit"),
    use_glmer=T
  )

  # use filtered data for the other DVs
  CORRECT_TRIALS_SUBSET <- "filteredCorrect"
  df_filtered <- filter(df, response_correct==1)

  # run_model_comparison(
  #   df_filtered,
  #   "RT_zscore", 
  #   iv_list,
  #   TASK,
  #   CORRECT_TRIALS_SUBSET,
  #   model
  # )
  run_model_comparison(
    df_filtered,
    "RT",
    iv_list,
    TASK,
    CORRECT_TRIALS_SUBSET,
    model,
    log_y=T
  )
  run_model_comparison(
    df_filtered,
    "AUC",
    iv_list,
    TASK,
    CORRECT_TRIALS_SUBSET,
    model,
  )
  run_model_comparison(
    df_filtered,
    "MAD",
    iv_list,
    TASK,
    CORRECT_TRIALS_SUBSET,
    model,
  )
  run_model_comparison(
    df_filtered,
    "acc_max_time",
    iv_list,
    TASK,
    CORRECT_TRIALS_SUBSET,
    model,
    log_y=T
  )
  run_model_comparison(
    df_filtered, 
    "xpos_flips", 
    iv_list, 
    TASK,
    CORRECT_TRIALS_SUBSET,
    model,
    family="poisson",
    use_glmer=T
  )
}

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# run comparisons for each LM  ----

# MODELS is defined from utils.R
for (i in 1:length(MODELS)) {
  model <- MODELS[[i]]
  message(sprintf("====== Running all model comparisons for %s ======", model))
  df <- load_animals_data(model)
  run_all_model_comparisons(df, model)
}