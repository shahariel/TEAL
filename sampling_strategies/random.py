"""
This code is is taken from the Avalanche library  https://github.com/ContinualAI/avalanche
"""

import random
from typing import List

from avalanche.benchmarks.utils import AvalancheDataset

from utils.storage_policy import ExemplarsSelectionStrategy


class RandomExemplarsSelectionStrategy(ExemplarsSelectionStrategy):
    """Select the exemplars at random in the dataset"""

    def make_sorted_indices(
            self, strategy: "SupervisedTemplate", data: AvalancheDataset
    ) -> List[int]:
        indices = list(range(len(data)))
        random.shuffle(indices)
        return indices
