"""
This code is based on the code of:
rainbow-memory
Copyright 2021-present NAVER Corp.
GPLv3
"""

import random
from typing import List

import numpy as np
import pandas as pd
import torch

from avalanche.benchmarks.utils import AvalancheDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import ImageOps

from utils.storage_policy import ExemplarsSelectionStrategy


def get_statistics(dataset: str):
    """
    Returns statistics of the dataset given a string of dataset name. To add new dataset,
    please add required statistics here
    """
    assert dataset in [
        "mnist",
        "KMNIST",
        "EMNIST",
        "FashionMNIST",
        "SVHN",
        "cifar10",
        "cifar100",
        "CINIC10",
        "imagenet100",
        "imagenet1000",
        "TinyImagenet",
    ]
    mean = {
        "mnist": (0.1307,),
        "KMNIST": (0.1307,),
        "EMNIST": (0.1307,),
        "FashionMNIST": (0.1307,),
        "SVHN": (0.4377, 0.4438, 0.4728),
        "cifar10": (0.4914, 0.4822, 0.4465),
        "cifar100": (0.5071, 0.4867, 0.4408),
        "CINIC10": (0.47889522, 0.47227842, 0.43047404),
        "TinyImagenet": (0.4802, 0.4481, 0.3975),
        "imagenet100": (0.485, 0.456, 0.406),
        "imagenet1000": (0.485, 0.456, 0.406),
    }

    std = {
        "mnist": (0.3081,),
        "KMNIST": (0.3081,),
        "EMNIST": (0.3081,),
        "FashionMNIST": (0.3081,),
        "SVHN": (0.1969, 0.1999, 0.1958),
        "cifar10": (0.2023, 0.1994, 0.2010),
        "cifar100": (0.2675, 0.2565, 0.2761),
        "CINIC10": (0.24205776, 0.23828046, 0.25874835),
        "TinyImagenet": (0.2302, 0.2265, 0.2262),
        "imagenet100": (0.229, 0.224, 0.225),
        "imagenet1000": (0.229, 0.224, 0.225),
    }

    classes = {
        "mnist": 10,
        "KMNIST": 10,
        "EMNIST": 49,
        "FashionMNIST": 10,
        "SVHN": 10,
        "cifar10": 10,
        "cifar100": 100,
        "CINIC10": 10,
        "TinyImagenet": 200,
        "imagenet100": 100,
        "imagenet1000": 1000,
    }

    in_channels = {
        "mnist": 1,
        "KMNIST": 1,
        "EMNIST": 1,
        "FashionMNIST": 1,
        "SVHN": 3,
        "cifar10": 3,
        "cifar100": 3,
        "CINIC10": 3,
        "TinyImagenet": 3,
        "imagenet100": 3,
        "imagenet1000": 3,
    }

    inp_size = {
        "mnist": 28,
        "KMNIST": 28,
        "EMNIST": 28,
        "FashionMNIST": 28,
        "SVHN": 32,
        "cifar10": 32,
        "cifar100": 32,
        "CINIC10": 32,
        "TinyImagenet": 64,
        "imagenet100": 224,
        "imagenet1000": 224,
    }
    return (
        mean[dataset],
        std[dataset],
        classes[dataset],
        inp_size[dataset],
        in_channels[dataset],
    )


class RainbowMemorySelectionStrategy(ExemplarsSelectionStrategy):
    def __init__(self, args, device):
        # self.uncert_metric = args["uncert_metric"]
        self.device = device
        self.uncert_metric = "vr"
        mean, std, n_classes, inp_size, _ = get_statistics(dataset=args.dataset.lower())
        self.test_transform = transforms.Compose(
            [
                transforms.Resize((inp_size, inp_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    def make_sorted_indices(
            self, strategy: "SupervisedTemplate", data: AvalancheDataset
    ) -> List[int]:
        """
        Should return the sorted list of indices to keep as exemplars.

        The last indices will be the first to be removed when cutoff memory.
        """
        if strategy.experience.current_experience == 0:
            selected = self.equal_class_sampling(data)
        else:
            selected = self.uncertainty_sampling(data, len(strategy.experience.classes_seen_so_far),
                                                 mem_size=strategy.plugins[1].storage_policy.max_size,
                                                 model=strategy.model)
        return selected

    def uncertainty_sampling(self, samples, num_class, mem_size, model):
        """uncertainty based sampling

        Args:
            samples ([list]): [training_list + memory_list]
        """
        mem_per_cls = mem_size // num_class
        if len(samples) <= mem_per_cls:
            return list(range(len(samples)))

        uncert_dicts = [{} for _ in range(len(samples))]
        self.montecarlo(samples, uncert_dicts, model, uncert_metric=self.uncert_metric)

        sample_df = pd.DataFrame(uncert_dicts)

        ret = []

        jump_idx = len(sample_df) // mem_per_cls
        uncertain_samples = sample_df.sort_values(by="uncertainty")[::jump_idx]
        ret += list(uncertain_samples.index)[:mem_per_cls]

        num_rest_slots = mem_per_cls - len(ret)
        if num_rest_slots > 0:
            print("Fill the unused slots by breaking the equilibrium.")
            ret += list(sample_df[~sample_df.index.isin(ret)].sample(n=num_rest_slots).index)

        num_dups = len(set(ret)) - len(ret)
        if num_dups > 0:
            print(f"Duplicated samples in memory for class {samples.targets.uniques}: {num_dups}")

        return ret

    def _compute_uncert(self, infer_list, infer_transform, uncert_dicts, uncert_name, model):
        batch_size = 32

        infer_dataset = infer_list.replace_current_transform_group(infer_transform)
        infer_loader = DataLoader(infer_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        model.eval()
        with torch.no_grad():
            for n_batch, data in enumerate(infer_loader):
                x = data[0]
                x = x.to(self.device)
                logit = model(x)
                logit = logit.detach().cpu()

                for i, cert_value in enumerate(logit):
                    sample = uncert_dicts[batch_size * n_batch + i]
                    sample[uncert_name] = 1 - cert_value

    def montecarlo(self, candidates, uncert_dicts, model, uncert_metric="vr"):
        transform_cands = []
        print(f"Compute uncertainty by {uncert_metric} for class {candidates.targets.uniques}")
        if uncert_metric == "vr":
            transform_cands = [
                Cutout(size=8),
                Cutout(size=16),
                Cutout(size=24),
                Cutout(size=32),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(45),
                transforms.RandomRotation(90),
                Invert(),
                Solarize(v=128),
                Solarize(v=64),
                Solarize(v=32),
            ]

        elif uncert_metric == "vr_cutout":
            transform_cands = [Cutout(size=16)] * 12

        n_transforms = len(transform_cands)

        for idx, tr in enumerate(transform_cands):
            _tr = transforms.Compose([tr] + self.test_transform.transforms)
            self._compute_uncert(candidates, _tr, uncert_dicts, f"uncert_{str(idx)}", model)

        for sample in uncert_dicts:
            self.variance_ratio(sample, n_transforms)

    def variance_ratio(self, sample, cand_length):
        vote_counter = torch.zeros(sample["uncert_0"].size(0))
        for i in range(cand_length):
            top_class = int(torch.argmin(sample[f"uncert_{i}"]))  # uncert argmin.
            vote_counter[top_class] += 1
        assert vote_counter.sum() == cand_length
        sample["uncertainty"] = (1 - vote_counter.max() / cand_length).item()

    def equal_class_sampling(self, samples):
        indices = list(range(len(samples)))
        random.shuffle(indices)
        return indices


class Cutout:
    def __init__(self, size=16) -> None:
        self.size = size

    def _create_cutout_mask(self, img_height, img_width, num_channels, size):
        """Creates a zero mask used for cutout of shape `img_height` x `img_width`.
        Args:
          img_height: Height of image cutout mask will be applied to.
          img_width: Width of image cutout mask will be applied to.
          num_channels: Number of channels in the image.
          size: Size of the zeros mask.
        Returns:
          A mask of shape `img_height` x `img_width` with all ones except for a
          square of zeros of shape `size` x `size`. This mask is meant to be
          elementwise multiplied with the original image. Additionally returns
          the `upper_coord` and `lower_coord` which specify where the cutout mask
          will be applied.
        """
        # Sample center where cutout mask will be applied
        height_loc = np.random.randint(low=0, high=img_height)
        width_loc = np.random.randint(low=0, high=img_width)

        size = int(size)
        # Determine upper right and lower left corners of patch
        upper_coord = (max(0, height_loc - size // 2), max(0, width_loc - size // 2))
        lower_coord = (
            min(img_height, height_loc + size // 2),
            min(img_width, width_loc + size // 2),
        )
        mask_height = lower_coord[0] - upper_coord[0]
        mask_width = lower_coord[1] - upper_coord[1]
        assert mask_height > 0
        assert mask_width > 0

        mask = np.ones((img_height, img_width, num_channels))
        zeros = np.zeros((mask_height, mask_width, num_channels))
        mask[
            upper_coord[0] : lower_coord[0], upper_coord[1] : lower_coord[1], :
        ] = zeros
        return mask, upper_coord, lower_coord

    def __call__(self, pil_img):
        pil_img = pil_img.copy()
        img_height, img_width, num_channels = (*pil_img.size, 3)
        _, upper_coord, lower_coord = self._create_cutout_mask(
            img_height, img_width, num_channels, self.size
        )
        pixels = pil_img.load()  # create the pixel map
        for i in range(upper_coord[0], lower_coord[0]):  # for every col:
            for j in range(upper_coord[1], lower_coord[1]):  # For every row
                pixels[i, j] = (125, 122, 113, 0)  # set the colour accordingly
        return pil_img


class Invert:
    def __call__(self, pil_img):
        return ImageOps.invert(pil_img)


class Solarize:
    def __init__(self, v):
        assert 0 <= v <= 256
        self.v = v

    def __call__(self, pil_img):
        return ImageOps.solarize(pil_img, self.v)