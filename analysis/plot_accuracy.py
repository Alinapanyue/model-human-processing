import pandas as pd
from os import listdir
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="ticks", font="Arial", font_scale=1.2)

# TEXT-BASED TASKS (Study 1-3)
def plot_text_task_accuracy():
    tasks = ["capitals-recall", "capitals-recognition", "animals", "syllogism"]
    models = ["Llama-2-7b-hf", "Llama-2-13b-hf", "Llama-2-70b-hf"]
    N_LAYERS = {
        "Llama-2-7b-hf": 32, "Llama-2-13b-hf": 40, "Llama-2-70b-hf": 80
    }
    outputs = []
    for task in tasks:
        for model in models:
            model_df = pd.read_csv(
                f"data/model_output/{task}_{model}.csv"
            )
            final_df = model_df[model_df.layer_idx==N_LAYERS[model]-1]
            final_df["model"] = model
            accuracy = final_df["sum_logprob_response_isCorrect"].mean()
            print(task, model, accuracy)
            final_df = final_df.rename(columns={
                "sum_logprob_response_isCorrect": "response_correct"
            })
            outputs.append(final_df)

        # Read human data.
        human_df = pd.read_csv(
            f"data/human/{task}_trial.csv" 
            if task != "capitals-recall" else f"data/human/{task}_trial_labeled.csv"
        )
        if task == "capitals-recall":
            human_df["response_correct"] = (human_df["gpt-4o_label"] == "Correct")
        elif task == "animals":
            human_df = human_df.rename(columns={"correct": "response_correct"})
        assert "response_correct" in human_df.columns
        human_df["task"] = task
        human_df["model"] = "Human"
        outputs.append(human_df)

        print("="*80)

    outputs = pd.concat(outputs).reset_index()
    ax = sns.barplot(
        data=outputs,
        x="task",
        y="response_correct",
        hue="model",
        hue_order=models + ["Human"],
        palette="BuPu",
        err_kws={"lw": 1}
    )
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("")
    ax.set_xticks(
        range(len(tasks)),
        [
            "Study 1a\n(Capitals recall)",
            "Study 1b\n(Capitals recog)",
            "Study 2\n(Animal exemplars)",
            "Study 3\n(Syllogisms)"
        ]
    )
    ax.axhline(0.5, linestyle='--', color="k")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1, 0.5))
    plt.gcf().set_size_inches(9,3)
    sns.despine()
    plt.savefig("figures/accuracy_text_tasks.pdf", dpi=300, bbox_inches="tight")

# VISION-BASED TASKS (Study 4)
def plot_vision_task_accuracy():
    models = ["vit_base_patch16_224", "vit_small_patch16_224"]
    outputs = []
    datasets = sorted([
        f for f in listdir("src/vision/model-vs-human/raw-data") if f != "metrics"
    ])
    for dataset in datasets:
        # read human data
        dataset_folder = f"src/vision/model-vs-human/raw-data/{dataset}"
        human_df = pd.concat([
            pd.read_csv(f"{dataset_folder}/{f}") for f in listdir(dataset_folder)
        ])
        human_df["response_correct"] = (human_df["object_response"] == human_df["category"])
        human_df["dataset_name"] = dataset
        human_df["model"] = "Human"
        outputs.append(human_df)

        # read model data
        for model in models:
            model_df = pd.read_csv(
                f"src/vision/model-vs-human/raw-data/metrics/{model}/{dataset}/Layerwise Reciprocal Rank.csv"
            )
            final_df = model_df[model_df.layer==model_df.layer.max()].rename(
                columns={"subj": "model"}
            )
            final_df["response_correct"] = (final_df["value"] == 1)
            outputs.append(final_df)
    outputs = pd.concat(outputs).reset_index()
    
    ax = sns.barplot(
        data=outputs,
        x="dataset_name",
        order=datasets,
        y="response_correct",
        hue="model",
        hue_order=models + ["Human"],
        palette="BuPu",
        err_kws={"lw": 1}
    )
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("")
    ax.set_xticks(range(len(datasets)), datasets, rotation=45, ha="right")
    ax.axhline(1/16, linestyle='--', color="k")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1, 0.5))
    plt.gcf().set_size_inches(12,3)
    sns.despine()
    plt.savefig("figures/accuracy_vision_tasks.pdf", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    plot_text_task_accuracy()
    plt.clf()
    plot_vision_task_accuracy()