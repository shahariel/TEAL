from typing import List

import torch
from avalanche.benchmarks.utils import AvalancheDataset
from torch import nn
from torch.utils.data import DataLoader

from utils.storage_policy import ExemplarsSelectionStrategy


class ClosestToCenterSelectionStrategy(ExemplarsSelectionStrategy):
    """A greedy algorithm that selects the remaining exemplar that is the
    closest to the center of all elements (in feature space).
    """

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
        center = self.features.mean(dim=0)
        distances = pow(self.features - center, 2).sum(dim=1)
        return distances.argsort()
