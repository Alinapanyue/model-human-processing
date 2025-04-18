library(tidyverse)
library(lme4)
library(lmerTest)
library(broom)

# global variables
OUT_DIR <- "analysis/anova_results"

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# helper functions ----

load_data <- function(file_path) {
  read_csv(file_path) |>
    mutate(
      # ~~~~~~~~~~~~~~~~~~ Output (final layer) measures
      # Uncertainty
      output_entropy                = scale(output_entropy),

      # Confidence in correct answer
      output_rank_correct        = scale(output_rank_correct),
      output_logprob_correct        = scale(output_logprob_correct),

      # Confidence in correct answer relative to intuitive
      output_logprobdiff            = scale(output_logprobdiff),

      # ~~~~~~~~~~~~~~~~~~ Processing measures
      # Uncertainty
      auc_entropy                   = scale(auc_entropy),
      layer_biggest_change_entropy        = scale(layer_biggest_change_entropy),

      # Confidence in correct answer
      auc_rank_correct               = scale(auc_rank_correct),
      auc_logprob_correct            = scale(auc_logprob_correct),
      layer_biggest_change_rank_correct    = scale(layer_biggest_change_rank_correct),
      layer_biggest_change_logprob_correct = scale(layer_biggest_change_logprob_correct),

      # Confidence in correct answer relative to intuitive
      # (based on probability differences)
      auc_logprobdiff_pos           = scale(auc_logprobdiff_pos),
      auc_logprobdiff_neg           = scale(auc_logprobdiff_neg),
      layer_biggest_change_logprobdiff    = scale(layer_biggest_change_logprobdiff),

      # Metrics given CONTROL prefix

      # THESE PREDICTORS ARE CONSTANT ACROSS ITEMS
      # control_auc_entropy     = scale(control_auc_entropy),
      # control_layer_biggest_change_entropy  = scale(control_layer_biggest_change_entropy),
      
      control_auc_rank_correct     = scale(control_auc_rank_correct),
      control_layer_biggest_change_rank_correct     = scale(control_layer_biggest_change_rank_correct),
      control_auc_logprob_correct     = scale(control_auc_logprob_correct),
      control_layer_biggest_change_logprob_correct     = scale(control_layer_biggest_change_logprob_correct),
      control_auc_logprobdiff_pos     = scale(control_auc_logprobdiff_pos),
      control_auc_logprobdiff_neg     = scale(control_auc_logprobdiff_neg),
      control_layer_biggest_change_logprobdiff     = scale(control_layer_biggest_change_logprobdiff),

      # Boosting of correct answer relative to intuitive
      auc_boost_pos            = scale(auc_boost_pos),
      auc_boost_neg            = scale(auc_boost_neg),
      layer_argmax_boost = scale(layer_argmax_boost)
    )
}

load_vision_data <- function(file_path) {
  read_csv(file_path) |> drop_na() |>
    mutate_if(is.character, as.factor) |>
    mutate(
      # response_correct = as.factor(response_correct), 
      # ~~~~~~~~~~~~~~~~~~ Output (final layer) measures
      # Uncertainty
      output_entropy         = scale(output_entropy),

      # Confidence
      output_rank_correct    = scale(output_rank_correct),
      output_logprob_correct = scale(output_logprob_correct),

      # ~~~~~~~~~~~~~~~~~~ Processing measures
      # Uncertainty
      auc_entropy                   = scale(auc_entropy),
      layer_biggest_change_entropy  = scale(layer_biggest_change_entropy),

      # Confidence in correct answer
      auc_rank_correct               = scale(auc_rank_correct),
      auc_logprob_correct            = scale(auc_logprob_correct),
      layer_biggest_change_rank_correct    = scale(layer_biggest_change_rank_correct),
      layer_biggest_change_logprob_correct = scale(layer_biggest_change_logprob_correct)
    )
}
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# define model predictor lists  ----

process_metrics <- list(
  "auc_entropy",
  "layer_biggest_change_entropy",
  "auc_rank_correct",
  "layer_biggest_change_rank_correct",
  "auc_logprob_correct",
  "layer_biggest_change_logprob_correct",
  "auc_logprobdiff_pos",
  "auc_logprobdiff_neg",
  "layer_biggest_change_logprobdiff",
  "auc_boost_pos",
  "auc_boost_neg",
  "layer_argmax_boost"
)
control_metrics <- list(
  "control_auc_rank_correct",
  "control_layer_biggest_change_rank_correct",
  "control_auc_logprob_correct",
  "control_layer_biggest_change_logprob_correct",
  "control_auc_logprobdiff_pos",
  "control_auc_logprobdiff_neg",
  "control_layer_biggest_change_logprobdiff"
)

fit_model <- function(df, dv, iv, task, family=NA, use_glmer=F, log_y=F, vision_dataset=NA) {
  dv_str <- ifelse(log_y, sprintf("log(%s)", dv), dv)
  
  # get string defining baseline variables
  if (task == "animals") {
    baseline_var_str <- "group * output_entropy * output_rank_correct * output_logprob_correct * output_logprobdiff"
  }
  else if (task == "vision") {
    # drop condition
    if (vision_dataset %in% c("cue_conflict", "edge", "silhouette", "sketch", "stylized")) {
      baseline_var_str <- "output_entropy * output_rank_correct * output_logprob_correct"
    }
    else {
      baseline_var_str <- "condition * output_entropy * output_rank_correct * output_logprob_correct"
    }
  }
  else if (task == "capitals-recognition") {
    # output_rank_correct is always 1, so it becomes NaN when scaled
    baseline_var_str <- "output_entropy * output_logprob_correct * output_logprobdiff"
  }
  else {
    baseline_var_str <- "output_entropy * output_rank_correct * output_logprob_correct * output_logprobdiff"
  }
  
  if (task != "syllogism") {
    # add random intercept for participants
    # (unfortunately, we don't have participant identifiers for syllogism)
    baseline_var_str <- sprintf("%s + (1 | subject_id)", baseline_var_str)
  }
  else {
    # we do have within-item variation for syllogism, but not the other tasks
    baseline_var_str <- sprintf("%s + (1 | syllogism_name)", baseline_var_str)
  }

  # get full string of formula
  if (iv == "baseline") {
    f_str <- sprintf("%s ~ %s", dv_str, baseline_var_str)
  }
  else {
    f_str <- sprintf("%s ~ %s + %s", dv_str, iv, baseline_var_str)
  }
  message(sprintf("Formula: %s", f_str))
  f <- formula(f_str)
  
  # fit model
  if (use_glmer) {
    m <- glmer(f, df, family=family, control=glmerControl(optimizer="bobyqa"))
    # optCtrl = list(maxfun = 1000)))
  }
  else {
    m <- lmer(f, df, na.action=na.omit, REML=F)
  }
  return(m)
}

get_all_fits <- function(df, dv, iv_list, ...) {
  fit_list <- list()
  for (i in 1:length(iv_list)) {
    fit <- fit_model(df, dv, iv_list[[i]], ...)
    fit_list[[i]] <- list("model"=fit, "iv"=iv_list[[i]])
  }
  return(fit_list)
}

compare_fits <- function(fit_list) {
  # Assumes that baseline is first.
  if (fit_list[[1]]$iv != "baseline") {
    stop()
  }
  anova_results <- list()
  baseline_fit <- fit_list[[1]]$model
  for (i in 2:length(fit_list)) {
    critical_fit <- fit_list[[i]]$model
    anova_res <- anova(baseline_fit, critical_fit)
    anova_tbl <- broom::tidy(anova_res) %>% mutate(
      iv = ifelse(term=="baseline_fit", "baseline", fit_list[[i]]$iv)
    )
    anova_results[[i-1]] <- anova_tbl
  }
  return(bind_rows(anova_results))
}

run_model_comparison <- function(df, dv, iv_list, task, data_subset, file_prefix=NA, ...) {
  message(sprintf(
    "Getting model fits for TASK = %s; DV = %s; DATA SUBSET = %s; FILE PREFIX = %s", 
    task, dv, data_subset, file_prefix
  ))
  fits <- get_all_fits(df, dv, iv_list, task, ...)
  anova_results <- compare_fits(fits)
  # remove duplicates, since there will be repeated baseline rows by construction
  anova_results <- anova_results[!duplicated(anova_results), ]
  anova_results$dv <- dv
  anova_results$data_subset <- data_subset
  anova_results$file_prefix <- file_prefix
  if (is.na(file_prefix)) {
    out_path <- sprintf("%s/%s_%s_%s.csv", OUT_DIR, task, data_subset, dv)
  }
  else {
    out_path <- sprintf("%s/%s_%s_%s_%s.csv", OUT_DIR, task, data_subset, file_prefix, dv)
  }
  # apply FDR procedure
  anova_results$p_fdr <- p.adjust(anova_results$p.value, method="BH")

  # write CSV file with results
  write.csv(anova_results, out_path, row.names=F)
  message("Wrote ANOVA results!")
  return(anova_results)
}
