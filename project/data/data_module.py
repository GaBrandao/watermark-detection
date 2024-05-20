import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision

from project.data.memory_dataset import MemoryDataset

import numpy as np
import os

from .settings import TRAIN_MEAN, TRAIN_STD

class DataModule(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()
        self.opt = hparams
        if 'num_workers' not in hparams.keys():
            self.opt['num_workers'] = 8
        if 'load_method' not in hparams.keys():
            self.opt['load_method'] = 'image'
        self.prepare_data()

    def prepare_data(self, ROOT="dataset", transforms: torchvision.transforms.Compose = None):
        if transforms == None:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(), 
                torchvision.transforms.Normalize(TRAIN_MEAN, TRAIN_STD)
            ])

        dataset_train = torchvision.datasets.ImageFolder(root=os.path.join(ROOT, 'train'), transform=transforms)
        dataset_valid = torchvision.datasets.ImageFolder(root=os.path.join(ROOT, 'valid'), transform=transforms)
        dataset_test = torchvision.datasets.ImageFolder(root=os.path.join(ROOT, 'test'), transform=transforms)

        if 'load_method' == 'memory':
            dataset_train = MemoryDataset(dataset=dataset_train)
            dataset_valid = MemoryDataset(dataset=dataset_valid)
            dataset_test = MemoryDataset(dataset=dataset_test)

        self.dataset = {
            'train': dataset_train,
            'valid': dataset_valid,
            'test': dataset_test
        }

    def get_dataloader_args(self, set: str):
        arg_dict = {
            'batch_size': self.opt['batch_size'],
            'num_workers': self.opt['num_workers'],
            'persistent_workers': True,
            'pin_memory': True,
            'shuffle': True if set == 'train' else False
        }

        return arg_dict
    
    def get_idx_to_class_dict(self):
        return {idx : c for c, idx in self.dataset['train'].class_to_idx.items()}

    def get_train_dataloader(self):
        args = self.get_dataloader_args('train')
        return DataLoader(self.dataset['train'], **args)

    def get_valid_dataloader(self):
        args = self.get_dataloader_args('valid')
        return DataLoader(self.dataset['valid'], **args)
    
    def get_test_dataloader(self):
        args = self.get_dataloader_args('test')
        return DataLoader(self.dataset['test'], **args)