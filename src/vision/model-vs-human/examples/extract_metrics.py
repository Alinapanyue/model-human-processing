# @MLEPORI EDIT
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from modelvshuman import Plot, MetricExtractor
from modelvshuman import constants as c
from plotting_definition import plotting_definition_template


def run_evaluation():
    models = ["vit_small_patch16_224", "vit_base_patch16_224"]
    datasets = c.DEFAULT_DATASETS # or e.g. ["cue-conflict", "uniform-noise"]
    params = {"batch_size": 64, "print_predictions": True, "num_workers": 1}
    MetricExtractor()(models, datasets, **params)


if __name__ == "__main__":
    # Evaluate models on out-of-distribution datasets, and print out metrics
    run_evaluation()
