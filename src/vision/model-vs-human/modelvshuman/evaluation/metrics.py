"""
Generic evaluation functionality: evaluate on several datasets.
"""
from abc import ABC, abstractmethod

from .. import datasets
from ..helper import human_categories as hc
import numpy as np
import scipy.stats as stats
import torch
import copy
from .. import constants as c
from ..datasets import info_mappings


class Metric(ABC):
    def __init__(self, name):
        self.name = name
        self.reset()

    def check_input(self, output, target, assert_ndarray=True):
        assert type(output) is np.ndarray
        assert len(output.shape) == 2, "output needs to have len(output.shape) == 2 instead of " + str(len(output.shape))

        if assert_ndarray:
            assert type(target) is np.ndarray
            assert output.shape[0] == target.shape[0]

    @abstractmethod
    def update(self, predictions, targets, paths):
        pass

    @abstractmethod
    def reset(self):
        pass

    @property
    @abstractmethod
    def value(self):
        pass

    def __str__(self):
        return "{}: {}".format(self.name, self.value)


class Accuracy(Metric):
    def __init__(self, name=None, topk=1):
        if name is None:
            name = "accuracy (top-{})".format(topk)
        super(Accuracy, self).__init__(name)
        self.topk = topk

    def reset(self):
        self._sum = 0
        self._count = 0

    def update(self, predictions, targets, paths):
        correct = [t in p[:self.topk] for t, p in zip(targets, predictions)]
        self._sum += np.sum(correct)
        self._count += len(predictions)

    @property
    def value(self):
        if self._count == 0:
            return 0
        return self._sum / self._count

    def __str__(self):
        return "{0:s}: {1:3.2f}".format(self.name, self.value * 100)

# @MLEPORI EDIT
class ReciprocalRank(Metric):
    def __init__(self, name=None):
        if name is None:
            name = "Layerwise Reciprocal Rank"
        super(ReciprocalRank, self).__init__(name)

    def reset(self):
        self._image2layer2RR = {}

    def update(self, predictions, categories, targets, paths, layer):
        sorted_indices = np.flip(np.argsort(predictions, axis=-1), axis=-1)
        rrs = []
        for idx in range(len(targets)):
            target_category_id = categories.index(targets[idx]) 
            rr = np.where(sorted_indices[idx] == target_category_id)[0][0]
            rrs.append(1/(rr + 1))
        
        for path_idx, path in enumerate(paths):
            if path not in self._image2layer2RR.keys():
                self._image2layer2RR[path] = {}
            if layer not in self._image2layer2RR[path].keys():
                self._image2layer2RR[path][layer] = rrs[path_idx].item()
            else:
                raise KeyError(f"Why was this path/layer combo seen already? {path}, {layer}")

    @property
    def value(self):
        return self._image2layer2RR

    def __str__(self):
        return str(self._image2layer2RR)
    
# @MLEPORI EDIT
class Probability(Metric):
    def __init__(self, name=None):
        if name is None:
            name = "Probability"
        super(Probability, self).__init__(name)

    def reset(self):
        self._image2layer2prob = {}

    def update(self, predictions, categories, targets, paths, layer):
        probs = []
        for idx in range(len(targets)):
            target_category_id = categories.index(targets[idx]) 
            prob = predictions[idx][target_category_id]
            probs.append(prob)
        
        for path_idx, path in enumerate(paths):
            if path not in self._image2layer2prob.keys():
                self._image2layer2prob[path] = {}
            if layer not in self._image2layer2prob[path].keys():
                self._image2layer2prob[path][layer] = probs[path_idx].item()
            else:
                raise KeyError(f"Why was this path/layer combo seen already? {path}, {layer}")

    @property
    def value(self):
        return self._image2layer2prob

    def __str__(self):
        return str(self._image2layer2prob)
    
# @MLEPORI EDIT
class Entropy(Metric):
    def __init__(self, name=None):
        if name is None:
            name = "Entropy"
        super(Entropy, self).__init__(name)

    def reset(self):
        self._image2layer2entropy = {}

    def update(self, predictions, categories, targets, paths, layer):
        entropy = []
        for idx in range(len(targets)):
            ent = stats.entropy(predictions[idx], axis=-1)
            entropy.append(ent)
        
        for path_idx, path in enumerate(paths):
            if path not in self._image2layer2entropy.keys():
                self._image2layer2entropy[path] = {}
            if layer not in self._image2layer2entropy[path].keys():
                self._image2layer2entropy[path][layer] = entropy[path_idx].item()
            else:
                raise KeyError(f"Why was this path/layer combo seen already? {path}, {layer}")

    @property
    def value(self):
        return self._image2layer2entropy

    def __str__(self):
        return str(self._image2layer2entropy)