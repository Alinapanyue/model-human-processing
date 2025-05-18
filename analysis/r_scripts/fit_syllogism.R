#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# load utils   ----
source("analysis/r_scripts/utils.R")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# define global variables: task and model predictor lists  ----

TASK <- "syllogism"
LENS <- "logit_lens"

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# helper functions  ----

load_syllogism_data <- function(model) {
  df <- load_data(
    sprintf("data/human_model_combined/%s/%s_%s.csv", LENS, TASK, model)
  ) %>% mutate(
    response_correct=as.integer(response_correct),
    answer_with_valid=as.integer(answer_with_valid),
    logic_belief_consistent=((is_valid&is_consistent) | (!is_valid&!is_consistent))
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
  
  # Get filtered dataframes.
  df_realistic <- filter(df, syllogism_condition != "nonsense")
  df_nonsense <- filter(df, syllogism_condition == "nonsense")
  df_contenteffects <- filter(df_realistic, !logic_belief_consistent)
  
  # FULL DATA
  run_model_comparison(
    df,
    "response_correct", 
    iv_list, 
    TASK,
    "all_trials",
    model,
    family=binomial(link="logit"), 
    use_glmer=T
  )
  # REALISTIC only
  run_model_comparison(
    df_realistic,
    "response_correct",
    iv_list,
    TASK,
    "filteredRealistic",
    model,
    family=binomial(link="logit"),
    use_glmer=T
  )
  # CONTENT EFFECTS only
  run_model_comparison(
    df_contenteffects,
    "response_correct",
    iv_list,
    TASK,
    "filteredContentEffects",
    model,
    family=binomial(link="logit"),
    use_glmer=T
  )

  # additionally filter trials for predicting human RTs
  subsets <- c(
    "filteredCorrect",
    "filteredRealisticCorrect",
    "filteredContentEffectsCorrect",
    "filteredContentEffects"
  )
  for (i in 1:length(subsets)){
    data_subset <- subsets[[i]]
    if (data_subset == "filteredCorrect") {
      df_filtered <- filter(df, response_correct==1)
    }
    else if (data_subset == "filteredRealisticCorrect") {
      df_filtered <- filter(df_realistic, response_correct==1)
    }
    else if (data_subset == "filteredContentEffectsCorrect") {
      df_filtered <- filter(df_contenteffects, response_correct==1)
    }
    else if (data_subset == "filteredContentEffects") {
      df_filtered <- df_contenteffects
    }
    run_model_comparison(
      df_filtered,
      "rt", 
      iv_list,
      TASK,
      data_subset,
      model,
      log_y=T
    )
  }
}

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# run comparisons for each LM  ----

# MODELS is defined from utils.R
for (i in 1:length(MODELS)) {
  model <- MODELS[[i]]
  message(sprintf("====== Running all model comparisons for %s ======", model))
  df <- load_syllogism_data(model)
  run_all_model_comparisons(df, model)
}