#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# load utils   ----
source("analysis/r_scripts/utils.R")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# define global variables: task ----

TASK <- "capitals-recall"
LENS <- "logit_lens"

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# helper functions  ----

load_capitals_recall_data <- function(model) {
  df <- load_data(
    sprintf("data/human_model_combined/%s/%s_%s.csv", LENS, TASK, model)
  ) %>% mutate(
    response_correct_gpt4 = as.factor(ifelse(`gpt-4o_label` == "Correct", 1, 0)),
    response_correct_strict = as.factor(response_correct_strict),
    log_n_keystrokes_len_norm = log(n_keystrokes_len_norm)
  ) %>% filter(
    # remove trials where there were fewer keystrokes than characters in the final answer
    n_keystrokes_len_diff >= 0,
    n_keystrokes > 0
  )
  return(df)
}

run_all_model_comparisons <- function(df, model) {
  # check if any columns are entirely NaN
  process_metrics <- get_nonNan_metrics(PROCESS_METRICS, df)
  iv_list <- c(
    list("baseline_final", "baseline_midpoint"),
    process_metrics
  )
  
  # accuracy DVs on full dataset (all trials)
  ALL_TRIALS_SUBSET <- "all_trials"
  run_model_comparison(
    df,
    "response_correct_gpt4",
    iv_list,
    TASK,
    ALL_TRIALS_SUBSET,
    model,
    family=binomial(link="logit"),
    use_glmer=T
  )
  run_model_comparison(
    df,
    "response_correct_strict",
    iv_list,
    TASK,
    ALL_TRIALS_SUBSET,
    model,
    family=binomial(link="logit"),
    use_glmer=T
  )

  # use filtered data for the other DVs
  CORRECT_TRIALS_SUBSET <- "filteredCorrect"
  df_filtered <- filter(df, response_correct_gpt4==1)

  # run_model_comparison(
  #   df_filtered, 
  #   "rt_zscore", 
  #   iv_list, 
  #   TASK,
  #   CORRECT_TRIALS_SUBSET,
  #   model
  # )
  run_model_comparison(
    df_filtered, 
    "rt", 
    iv_list,
    TASK,
    CORRECT_TRIALS_SUBSET,
    model,
    log_y=T
  )
  run_model_comparison(
    df_filtered, 
    "n_backspace", 
    iv_list, 
    TASK,
    CORRECT_TRIALS_SUBSET,
    model,
    family="poisson", 
    use_glmer=T
  )
  run_model_comparison(
    df_filtered,
    "n_keystrokes_len_norm",
    iv_list,
    TASK,
    CORRECT_TRIALS_SUBSET,
    model,
    log_y=T
  )
  run_model_comparison(
    df_filtered,
    "time_stroke_after_last_empty_trial",
    iv_list,
    TASK,
    CORRECT_TRIALS_SUBSET,
    model,
    log_y=T
  )
}

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# run comparisons for each LM  ----

# MODELS is defined from utils.R
for (i in 1:length(MODELS)) {
  model <- MODELS[[i]]
  message(sprintf("====== Running all model comparisons for %s ======", model))
  df <- load_capitals_recall_data(model)
  run_all_model_comparisons(df, model)
}