import sys
import numpy as np
import math
import random
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data.sampler import Sampler
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import TensorDataset, DataLoader
#from networks.resnet_big import SupConResNet, SupConCNN, Predictor

#from losses import SupConLoss, IRDLoss


class SeqSampler(Sampler):
    def __init__(self, data_source, blend_ratio, n_concurrent_classes):
        """data_source is a Subset"""
        self.num_samples = len(data_source)
        self.blend_ratio = blend_ratio
        self.n_concurrent_classes = n_concurrent_classes
        if torch.is_tensor(data_source.dataset.targets):
            self.labels = data_source.dataset.targets.detach().cpu().numpy()
        else:  # targets in cifar10 and cifar100 is a list
            self.labels = np.array(data_source.dataset.targets)
        self.labels = self.labels[data_source.indices]
        self.classes = list(set(self.labels))
        self.n_classes = len(self.classes)

    def __iter__(self):
        """Sequential sampler"""
        # Configure concurrent classes
        cmin = []
        cmax = []
        for i in range(int(self.n_classes / self.n_concurrent_classes)):
            for _ in range(self.n_concurrent_classes):
                cmin.append(i * self.n_concurrent_classes)
                cmax.append((i + 1) * self.n_concurrent_classes)
        print('cmin', cmin)
        print('cmax', cmax)

        filter_fn = lambda y: np.logical_and(
            np.greater_equal(y, cmin[c]), np.less(y, cmax[c]))

        # Configure sequential class-incremental input
        sample_idx = []
        for c in self.classes:
            filtered_train_ind = filter_fn(self.labels)
            filtered_ind = np.arange(self.labels.shape[0])[filtered_train_ind]
            np.random.shuffle(filtered_ind)
            sample_idx.append(filtered_ind.tolist())

        # Configure blending class
        if self.blend_ratio > 0.0:
            for c in range(len(self.classes)):
                # Blend examples from the previous class if not the first
                if c > 0:
                    blendable_sample_num = \
                        int(min(len(sample_idx[c]), len(sample_idx[c-1])) * self.blend_ratio / 2)
                    # Generate a gradual blend probability
                    blend_prob = np.arange(0.5, 0.05, -0.45 / blendable_sample_num)
                    assert blend_prob.size == blendable_sample_num, \
                        'unmatched sample and probability count'

                    # Exchange with the samples from the end of the previous
                    # class if satisfying the probability, which decays
                    # gradually
                    for ind in range(blendable_sample_num):
                        if random.random() < blend_prob[ind]:
                            tmp = sample_idx[c-1][-ind-1]
                            sample_idx[c-1][-ind-1] = sample_idx[c][ind]
                            sample_idx[c][ind] = tmp

        final_idx = []
        for sample in sample_idx:
            final_idx += sample
        return iter(final_idx)