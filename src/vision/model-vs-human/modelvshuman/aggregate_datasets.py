import os
import pandas as pd
import numpy as np

from constants import DEFAULT_DATASETS

# @MLEPORI EDIT
# Script to consolidate data from subjects and compute 
# model metrics over layers and output

def compute_rr_auc(data):
    baseline_prob = 1/16
    return np.sum(data["value"] - baseline_prob).item()

def compute_acc(data):
    if data[data["layer"] == data["layer"].max()]["value"].item() == 1:
        return 1
    else:
        return 0

def compute_final_prob(data):
    return data[data["layer"] == data["layer"].max()]["value"].item()

def compute_final_entropy(data):
    return data[data["layer"] == data["layer"].max()]["value"].item()

def compute_prob_auc(data):
    return np.sum(data["value"]).item()

os.makedirs("../processed-data", exist_ok=True)

models = ["vit_base_patch16_224", "vit_small_patch16_224"]

for dataset in DEFAULT_DATASETS:
    for model in models:

        outfile_name = model + "_" + dataset + ".csv"

        model_df = {
            "imagename": [],
            "rr_auc": [],
            "model_acc": [],
            "final_prob": [],
            "prob_auc": [],
            "final_ent": [],
        }

        human_dir = os.path.join("../raw-data", dataset)
        model_dir = os.path.join("../raw-data/metrics", model, dataset)

        rank_data = pd.read_csv(os.path.join(model_dir, "Layerwise Reciprocal Rank.csv"))
        prob_data = pd.read_csv(os.path.join(model_dir, "Probability.csv"))
        entropy_data = pd.read_csv(os.path.join(model_dir, "Entropy.csv"))

        rank_data = rank_data.groupby("stimulus")
        for stimulus, rank_grp in rank_data:

            prob_grp = prob_data[prob_data["stimulus"] == stimulus]
            ent_grp = entropy_data[entropy_data["stimulus"] == stimulus]

            if dataset in ["cue-conflict", "edge", "silhouette"]:
                pass
            else:
                stimulus = "_".join(stimulus.split("_")[3:])


            rr_auc = compute_rr_auc(rank_grp)
            acc = compute_acc(rank_grp)
            final_prob = compute_final_prob(prob_grp)
            prob_auc = compute_prob_auc(prob_grp)
            final_ent = compute_final_entropy(ent_grp)

            model_df["imagename"].append(stimulus)
            model_df["rr_auc"].append(rr_auc)
            model_df["model_acc"].append(acc)
            model_df["final_prob"].append(final_prob)
            model_df["prob_auc"].append(prob_auc)
            model_df["final_ent"].append(final_ent)

        model_df = pd.DataFrame.from_dict(model_df)

        subject_df = {
            "imagename": [],
            "subject_acc": [],
            "subject_rt": [],
            "subject_zscore_rt": [],
            "subject": [],
        }
        subject_files = os.listdir(human_dir)

        for subject in subject_files:
            subject_data = pd.read_csv(os.path.join(human_dir, subject))
            subject_data["subject_acc"] = subject_data["category"] == subject_data["object_response"]
            rts = subject_data["rt"]
            zscore_rt = (rts - np.mean(rts))/np.std(rts)
            subject_data["zscore_rt"] = zscore_rt
            
            for _, row in subject_data.iterrows():
                if dataset in ["cue-conflict", "edge", "silhouette"]:
                   subject_df["imagename"].append("_".join(row["imagename"].split("_")[6:]))
                else:
                    subject_df["imagename"].append("_".join(row["imagename"].split("_")[3:]))
                subject_df["subject"].append(row["subj"])
                subject_df["subject_acc"].append(row["subject_acc"])
                subject_df["subject_rt"].append(row["rt"])
                subject_df["subject_zscore_rt"].append(row["zscore_rt"])

        
        subject_df = pd.DataFrame.from_dict(subject_df)
        pd.merge(subject_df, model_df, on="imagename").to_csv("../processed-data/" + outfile_name)

