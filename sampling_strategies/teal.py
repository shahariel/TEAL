"""
Some of the code here is based on the code of Typiclust in the following repository:
https://github.com/avihu111/TypiClust
"""

from datetime import datetime
from typing import List

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.training import ExemplarsSelectionStrategy
import faiss
from sklearn.cluster import MiniBatchKMeans, KMeans
import itertools


def get_nn(features, num_neighbors):
    # calculates nearest neighbors on GPU
    d = features.shape[1]
    features = features.astype(np.float32)
    cpu_index = faiss.IndexFlatL2(d)
    gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
    gpu_index.add(features)  # add vectors to the index
    distances, indices = gpu_index.search(features, num_neighbors + 1)
    # 0 index is the same sample, dropping it
    return distances[:, 1:], indices[:, 1:]


def get_mean_nn_dist(features, num_neighbors, return_indices=False):
    distances, indices = get_nn(features, num_neighbors)
    mean_distance = distances.mean(axis=1)
    if return_indices:
        return mean_distance, indices
    return mean_distance


def calculate_typicality(features, num_neighbors):
    mean_distance = get_mean_nn_dist(features, num_neighbors)
    # low distance to NN is high density
    typicality = 1 / (mean_distance + 1e-5)
    return typicality


def kmeans(features, num_clusters):
    km = KMeans(n_clusters=num_clusters)
    km.fit_predict(features)
    return km.labels_


class TEALExemplarsSelectionStrategy(ExemplarsSelectionStrategy):
    """Select the exemplars by typicality and diversity in the dataset, using TEAL"""
    K_NN = 20
    MIN_CLUSTER_SIZE = 5
    MAX_NUM_CLUSTERS = 500

    def __init__(self, args, device, extra_args=None):
        self.clusters = None
        self.features = None
        self.group_to_len = {}
        self.seen_groups = set()  # groups that were seen in this current buffer update
        self.groups_in_buffer = set()
        self.group_exps_indices = {}
        self.device = device
        self.make_sorted_indices_func = self.get_make_sorted_func(args.teal_type)
        self.dataset_name = args.dataset

    def init_features(self, data, model):
        inc_classifier = model.linear
        model.linear = nn.Identity()
        model.eval()

        dataloader = DataLoader(data, batch_size=len(data), shuffle=False)
        for batch in dataloader:
            images = batch[0].to(self.device)  # to get all the images in the dataset

        self.features = model(images).detach().cpu().numpy()

        model.linear = inc_classifier

    def init_clusters(self, ll):
        num_clusters = ll if ll / len(self.features) < 0.2 else ll // 5
        num_clusters = min(num_clusters, self.MAX_NUM_CLUSTERS)
        print(f'Clustering into {num_clusters} clusters...')
        self.clusters = kmeans(self.features, num_clusters=num_clusters)

    def start_buffer_update(self, storage_policy):
        # Initialize self.group_to_len
        lens = storage_policy.get_group_lengths(len(storage_policy.seen_groups))
        for group_id, ll in zip(storage_policy.seen_groups, lens):
            self.group_to_len[group_id] = ll
        print(self.group_to_len)
        # Initialize self.seen_groups
        self.seen_groups = set()

    def get_make_sorted_func(self, teal_type):
        print(f"TEAL type: {teal_type}")
        if teal_type == 'one_time':
            return self.make_sorted_indices_one_time
        elif teal_type == 'log_iterative':
            return self.make_sorted_indices_log_iterative
        else:
            print(f"TEAL type: {teal_type} not recognized")
            exit()

    def make_sorted_indices(
        self, strategy: "SupervisedTemplate", data: AvalancheDataset
    ) -> List[int]:
        return self.make_sorted_indices_func(strategy, data)

    def make_sorted_indices_one_time(
        self, strategy: "SupervisedTemplate", data: AvalancheDataset
    ) -> List[int]:
        """
        TEAL One-time
        """
        cur_class = list(data.targets.uniques)[0]
        if self.group_to_len.get(cur_class) is None:  # Starting a new buffer update
            self.start_buffer_update(storage_policy=strategy.plugins[1].storage_policy)

        ll = self.group_to_len[cur_class]

        if cur_class in self.seen_groups:  # Buffer was already updated
            return list(range(ll))

        if cur_class in self.groups_in_buffer:
            self.seen_groups.add(cur_class)
            return list(range(ll))  # takes the top ll from the original TEAL "selected" order - Verified

        self.init_features(data, strategy.model)
        self.init_clusters(ll)
        labels = np.copy(self.clusters)

        # counting cluster sizes and number of labeled samples per cluster
        cluster_ids, cluster_sizes = np.unique(labels, return_counts=True)
        clusters_df = pd.DataFrame({'cluster_id': cluster_ids, 'cluster_size': cluster_sizes,
                                    'neg_cluster_size': -1 * cluster_sizes})
        # drop too small clusters
        clusters_df = clusters_df[clusters_df.cluster_size > self.MIN_CLUSTER_SIZE]
        # sort clusters by lowest number of existing samples, and then by cluster sizes (large to small)
        clusters_df = clusters_df.sort_values(['neg_cluster_size'])

        selected = []

        for i in range(ll):
            indices, j = [], 0
            while (len(indices) == 0):  # skip empty clusters
                cluster = clusters_df.iloc[(i+j) % len(clusters_df)].cluster_id
                indices = (labels == cluster).nonzero()[0]
                j += 1

            rel_feats = self.features[indices]
            # in case we have too small cluster, calculate density among half of the cluster
            typicality = calculate_typicality(rel_feats, min(self.K_NN, len(indices) // 2))
            idx = indices[typicality.argmax()]
            selected.append(idx)
            labels[idx] = -1

        self.seen_groups.add(cur_class)
        self.groups_in_buffer.add(cur_class)
        return selected

    def make_sorted_indices_log_iterative(
        self, strategy: "SupervisedTemplate", data: AvalancheDataset
    ) -> List[int]:
        """
        TEAL log-iterative selection
        """
        cur_class = list(data.targets.uniques)[0]

        if self.group_to_len.get(cur_class) is None:  # Starting a new buffer update
            self.start_buffer_update(storage_policy=strategy.plugins[1].storage_policy)

        ll = self.group_to_len[cur_class]

        if cur_class in self.seen_groups:  # Buffer was already updated
            return list(range(ll))

        if cur_class in self.groups_in_buffer:
            self.seen_groups.add(cur_class)
            return list(range(ll))  # takes the top ll from the original TEAL "selected" order - Verified

        base = 1.4
        if np.emath.logn(base, ll) + 1 <= 4:
            log_iterative_sizes = [ll]
        else:
            log_iterative_sizes = (base ** np.arange(4, np.emath.logn(base, ll) + 1)).astype('int')
            log_iterative_sizes[-1] = ll
        self.init_features(data, strategy.model)
        existing_indices = []
        selected = []
        for i in range(len(log_iterative_sizes)):
            self.init_clusters(log_iterative_sizes[i])
            labels = np.copy(self.clusters)

            # counting cluster sizes and number of labeled samples per cluster
            cluster_ids, cluster_sizes = np.unique(labels, return_counts=True)
            cluster_labeled_counts = np.bincount(labels[existing_indices], minlength=len(cluster_ids))

            clusters_df = pd.DataFrame({'cluster_id': cluster_ids, 'cluster_size': cluster_sizes,
                                        'existing_count': cluster_labeled_counts,
                                        'neg_cluster_size': -1 * cluster_sizes})
            # drop too small clusters
            clusters_df = clusters_df[clusters_df.cluster_size > self.MIN_CLUSTER_SIZE]
            # sort clusters by lowest number of existing samples, and then by cluster sizes (large to small)
            clusters_df = clusters_df.sort_values(['existing_count', 'neg_cluster_size'])
            labels[existing_indices] = -1

            if i == 0:
                buffer_diff = log_iterative_sizes[0]
            else:
                buffer_diff = log_iterative_sizes[i] - log_iterative_sizes[i - 1]

            for j in range(buffer_diff):
                indices, h = [], 0
                while (len(indices) == 0):
                    cluster = clusters_df.iloc[(j + h) % len(clusters_df)].cluster_id
                    indices = (labels == cluster).nonzero()[0]
                    h += 1
                rel_feats = self.features[indices]
                # in case we have too small cluster, calculate density among half of the cluster
                typicality = calculate_typicality(rel_feats, min(self.K_NN, len(indices) // 2))
                idx = indices[typicality.argmax()]
                selected.append(idx)
                labels[idx] = -1
            existing_indices = np.where(labels == -1)

        self.seen_groups.add(cur_class)
        self.groups_in_buffer.add(cur_class)

        return selected[:ll]