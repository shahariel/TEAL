"""
This code is based on the implementation of Herding in the
Avalanche library https://github.com/ContinualAI/avalanche
"""

from abc import ABC
from typing import (
    List,
    TYPE_CHECKING,
)

import torch
from numpy import inf
from torch import nn
from torch.utils.data import DataLoader

from avalanche.benchmarks.utils import (
    AvalancheDataset,
)

from utils.storage_policy import ExemplarsSelectionStrategy

if TYPE_CHECKING:
    from avalanche.training.templates import SupervisedTemplate


class HerdingSelectionStrategy(ExemplarsSelectionStrategy, ABC):
    """Feature extraction like Typiclust selection strategy, and the examplars'
    selection is by herding."""

    def __init__(self):
        self.features = None

    def init_features(self, data, model, device):
        inc_classifier = model.linear
        model.linear = nn.Identity()
        model.eval()
        dataloader = DataLoader(data, batch_size=len(data), shuffle=False)
        for batch in dataloader:
            images = batch[0].to(device)  # to get all the images in the dataset
        self.features = model(images).detach().cpu()
        print(f'Features shape: {self.features.shape}')
        model.linear = inc_classifier

    @torch.no_grad()
    def make_sorted_indices(
        self, strategy: "SupervisedTemplate", data: AvalancheDataset
    ) -> List[int]:
        self.init_features(data, strategy.model, strategy.device)
        selected_indices: List[int] = []
        center = self.features.mean(dim=0)
        current_center = center * 0

        for i in range(len(self.features)):
            # Compute distances with real center
            candidate_centers = current_center * i / (i + 1) + self.features / (i + 1)
            distances = pow(candidate_centers - center, 2).sum(dim=1)
            distances[selected_indices] = inf

            # Select best candidate
            new_index = distances.argmin().tolist()
            selected_indices.append(new_index)
            current_center = candidate_centers[new_index]

        return selected_indices
