import torch
from avalanche.benchmarks.utils import classification_subset
from avalanche.core import SupervisedPlugin
from torch.utils.data import DataLoader


class NaiveReplayPlugin(SupervisedPlugin):
    """
    Naive Experience Replay plugin. Manages a buffer with a desired storage policy (of updating
    the buffer) while keeping the mini batches the dataloder produces class-balanced.
    """
    def __init__(self, storage_policy):
        super().__init__()
        self.storage_policy = storage_policy

    def before_training_exp(self, strategy,
                            num_workers: int = 0, shuffle: bool = True,
                            **kwargs):
        """ Here we set the dataloader. """
        data = strategy.experience.dataset

        # Get sample idxs per class
        cl_idxs = {}
        for idx, target in enumerate(data.targets):
            if target not in cl_idxs:
                cl_idxs[target] = []
            cl_idxs[target].append(idx)

        # Make AvalancheSubset per class
        cl_datasets = None
        for c, c_idxs in cl_idxs.items():
            if cl_datasets is None:
                cl_datasets = classification_subset(data, indices=c_idxs)
            else:
                cl_datasets += classification_subset(data, indices=c_idxs)

        if len(self.storage_policy.buffer_datasets) == 0:
            ds = cl_datasets
        else:
            buffer_datasets = self.storage_policy.buffer_datasets[0]
            for buffer_dataset in self.storage_policy.buffer_datasets[1:]:
                buffer_datasets += buffer_dataset
            ds = buffer_datasets + cl_datasets

        weights = {cl: 1. / count for cl, count in ds.targets.count.items()}
        samples_weight = torch.tensor([weights[t] for t in list(ds.targets)])
        sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, len(ds), replacement=True)
        strategy.dataloader = DataLoader(ds, batch_size=strategy.train_mb_size, num_workers=num_workers, sampler=sampler)

    def after_training_exp(self, strategy, **kwargs):
        print("Buffer update.")
        self.storage_policy.update(strategy, **kwargs)
