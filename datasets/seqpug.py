# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace
import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets import CIFAR10

from utils.conf import base_path
from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import (ContinualDataset, fix_class_names_order,
                                              store_masked_loaders)
from datasets.utils import set_default_from_args        
    

class CUSTOMPUG(Dataset):
    """
    Overrides the CIFAR10 dataset to change the getitem function.
    """

    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        
        if self.train:
            self.split = 'Train'
        else:
            self.split = 'Test'
        
        all_data = []
        all_targets = []
        for file_idx in [0,1,2,3]:
            data = np.load(f'data_from_hs/{self.split}_Data_Target{file_idx}.npy')
            targets = np.load(f'data_from_hs/{self.split}_Label_Target{file_idx}.npy')
            all_data.append(data)
            all_targets.append(targets)
        
        
        self.data = np.concatenate(all_data)
        self.targets = np.concatenate(all_targets)
        self.data = np.reshape(self.data, (-1, 6, 250))
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.uint8(255 * img))
        original_img = img.copy()

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, original_img, self.logits[index]

        return img, target, img


class SequentialPUG(ContinualDataset):
    """Sequential PUG Dataset.

    Args:
        NAME (str): name of the dataset.
        SETTING (str): setting of the dataset.
        N_CLASSES_PER_TASK (int): number of classes per task.
        N_TASKS (int): number of tasks.
        N_CLASSES (int): number of classes.
        SIZE (tuple): size of the images.
        MEAN (tuple): mean of the dataset.
        STD (tuple): standard deviation of the dataset.
        TRANSFORM (torchvision.transforms): transformations to apply to the dataset.
    """

    NAME = 'seq-pug'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 1
    N_TASKS = 4
    N_CLASSES = N_CLASSES_PER_TASK * N_TASKS
    SIZE = (6, 250)
    TRANSFORM = transforms.Compose([transforms.ToTensor()])

    TEST_TRANSFORM = transforms.Compose([transforms.ToTensor()])

    def __init__(self, args, transform_type: str = 'weak'):
        super().__init__(args)

        assert transform_type in ['weak', 'strong'], "Transform type must be either 'weak' or 'strong'."

        if transform_type == 'strong':
            logging.info("Using strong augmentation for PUG")
            self.TRANSFORM = transforms.Compose([transforms.ToTensor()])

    def get_data_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """Class method that returns the train and test loaders."""
        transform = self.TRANSFORM

        train_dataset = CUSTOMPUG(base_path() + 'PUG', train=True,
                                  download=True, transform=transform)
        test_dataset = CUSTOMPUG(base_path() + 'PUG', train=False,
                                download=True, transform=self.TEST_TRANSFORM)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialPUG.TRANSFORM])
        return transform

    @set_default_from_args("backbone")
    def get_backbone():
        return "pugbackbone"

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize(SequentialPUG.MEAN, SequentialPUG.STD)
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(SequentialPUG.MEAN, SequentialPUG.STD)
        return transform

    @set_default_from_args('n_epochs')
    def get_epochs(self):
        return 30

    @set_default_from_args('batch_size')
    def get_batch_size(self):
        return 10

    def get_class_names(self):
        if self.class_names is not None:
            return self.class_names
        classes = CIFAR10(base_path() + 'SequentialPUG', train=True, download=True).classes
        classes = fix_class_names_order(classes, self.args)
        self.class_names = classes
        return self.class_names
