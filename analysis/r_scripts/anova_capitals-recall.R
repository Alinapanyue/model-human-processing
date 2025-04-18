#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# load utils   ----
source("analysis/r_scripts/anova_utils.R")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# define global variables: task and model predictor lists  ----

TASK <- "capitals-recall"
iv_list <- c(
  list("baseline"),
  process_metrics # defined from anova_utils.R
  # control_metrics # defined from anova_utils.R
)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# helper functions  ----

load_capitals_recall_data <- function(model) {
  df <- load_data(
    sprintf("data/human_model_combined/%s_%s.csv", TASK, model)
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
  file_prefix <- model
  # accuracy DVs on full dataset (all trials)
  ALL_TRIALS_SUBSET <- "all_trials"
  run_model_comparison(
    df, 
    "response_correct_gpt4", 
    iv_list, 
    TASK,
    family=binomial(link="logit"), 
    use_glmer=T,
    data_subset=ALL_TRIALS_SUBSET,
    file_prefix=file_prefix
  )
  run_model_comparison(
    df, 
    "response_correct_strict", 
    iv_list, 
    TASK,
    family=binomial(link="logit"), 
    use_glmer=T,
    data_subset=ALL_TRIALS_SUBSET,
    file_prefix=file_prefix
  )

  # use filtered data for the other DVs
  CORRECT_TRIALS_SUBSET <- "filteredCorrect"
  df_filtered <- filter(df, response_correct_gpt4==1)

  run_model_comparison(
    df_filtered, 
    "rt_zscore", 
    iv_list, 
    TASK,
    data_subset=CORRECT_TRIALS_SUBSET,
    file_prefix=file_prefix
  )
  run_model_comparison(
    df_filtered, 
    "rt", 
    iv_list,
    TASK,
    data_subset=CORRECT_TRIALS_SUBSET,
    log_y=T,
    file_prefix=file_prefix
  )
  run_model_comparison(
    df_filtered, 
    "n_backspace", 
    iv_list, 
    TASK,
    family="poisson", 
    use_glmer=T, 
    data_subset=CORRECT_TRIALS_SUBSET,
    file_prefix=file_prefix
  )
  run_model_comparison(
    df_filtered,
    "n_keystrokes_len_norm",
    iv_list,
    TASK,
    data_subset=CORRECT_TRIALS_SUBSET,
    log_y=T,
    file_prefix=file_prefix
  )
  run_model_comparison(
    df_filtered,
    "time_stroke_after_last_empty_trial",
    iv_list,
    TASK,
    data_subset=CORRECT_TRIALS_SUBSET,
    log_y=T,
    file_prefix=file_prefix
  )
}

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# run comparisons for each LM  ----

models <- c("Llama-2-7b-hf", "Llama-2-13b-hf", "Llama-2-70b-hf")
for (i in 1:length(models)) {
  model <- models[[i]]
  message(sprintf("====== Running all model comparisons for %s ======", model))
  df <- load_capitals_recall_data(model)
  run_all_model_comparisons(df, model)
}