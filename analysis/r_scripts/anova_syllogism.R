#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# load utils   ----
source("analysis/r_scripts/anova_utils.R")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# define global variables: task and model predictor lists  ----

TASK <- "syllogism"

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# helper functions  ----

load_syllogism_data <- function(model) {
  df <- load_data(
    sprintf("data/human_model_combined/%s_%s.csv", TASK, model)
  ) %>% mutate(
    response_correct=as.integer(response_correct),
    answer_with_valid=as.integer(answer_with_valid),
    logic_belief_consistent=((is_valid&is_consistent) | (!is_valid&!is_consistent))
  )
  return(df)
}

run_all_model_comparisons <- function(df, model) {
  # check if any columns are entirely NaN
  good_process_metrics <- list()
  # process_metrics is defined from anova_utils.R
  for (i in 1:length(process_metrics)) {
    metric <- process_metrics[[i]]
    if (all(is.na(df[[metric]]))) {
      message(sprintf("Removing the following metric (all NA): %s", metric))
    }
    else {
      good_process_metrics <- append(good_process_metrics, metric)
    }
  }
  iv_list <- c(
    list("baseline"),
    good_process_metrics
  )
  
  file_prefix <- model
  df_realistic <- filter(df, syllogism_condition != "nonsense")
  df_nonsense <- filter(df, syllogism_condition == "nonsense")
  df_contenteffects <- filter(df_realistic, !logic_belief_consistent)
  # FULL DATA
  run_model_comparison(
    df,
    "response_correct", 
    iv_list, 
    TASK,
    family=binomial(link="logit"), 
    use_glmer=T,
    data_subset="all_trials",
    file_prefix=file_prefix
  )
  # REALISTIC only
  # run_model_comparison(
  #   df_realistic,
  #   "response_correct", 
  #   iv_list, 
  #   TASK,
  #   family=binomial(link="logit"), 
  #   use_glmer=T,
  #   data_subset="filteredRealistic",
  #   file_prefix=file_prefix
  # )
  # NONSENSE only
  # run_model_comparison(
  #   df_nonsense,
  #   "response_correct", 
  #   iv_list, 
  #   TASK,
  #   family=binomial(link="logit"), 
  #   use_glmer=T,
  #   data_subset="filteredNonsense",
  #   file_prefix=file_prefix
  # )
  # CONTENT EFFECTS only
  # run_model_comparison(
  #   df_contenteffects,
  #   "response_correct", 
  #   iv_list, 
  #   TASK,
  #   family=binomial(link="logit"), 
  #   use_glmer=T,
  #   data_subset="filteredContentEffects",
  #   file_prefix=file_prefix
  # )

  # additionally filter to correct trials, within each of these conditions
  subsets <- c(
    "filteredCorrect",
    # "filteredRealistic",
    "filteredContentEffects"
    # "filteredRealisticCorrect",
    # "filteredContentEffectsCorrect"
  )
  for (i in 1:length(subsets)){
    data_subset <- subsets[[i]]
    if (data_subset == "filteredCorrect") {
      df_filtered <- filter(df, response_correct==1)
    }
    # else if (data_subset == "filteredRealisticCorrect") {
    #   df_filtered <- filter(df_realistic, response_correct==1)
    # }
    # else if (data_subset == "filteredContentEffectsCorrect") {
    #   df_filtered <- filter(df_contenteffects, response_correct==1)
    # }
    # else if (data_subset == "filteredRealistic") {
    #   df_filtered <- df_realistic
    # }
    else if (data_subset == "filteredContentEffects") {
      df_filtered <- df_contenteffects
    }
    run_model_comparison(
      df_filtered,
      "rt", 
      iv_list,
      TASK,
      log_y=T,
      data_subset=data_subset,
      file_prefix=file_prefix
    )
  }
}

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# run comparisons for each LM  ----

models <- c("Llama-2-7b-hf", "Llama-2-13b-hf", "Llama-2-70b-hf")
for (i in 1:length(models)) {
  model <- models[[i]]
  message(sprintf("====== Running all model comparisons for %s ======", model))
  df <- load_syllogism_data(model)
  run_all_model_comparisons(df, model)
}