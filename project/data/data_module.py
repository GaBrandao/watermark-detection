import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import torchvision
import torchvision.transforms as transforms

import numpy as np

class DataModule(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.opt = hparams
        if 'loading_method' not in hparams.keys():
            self.opt['loading_method'] = 'Image'
        if 'num_workers' not in hparams.keys():
            self.opt['num_workers'] = 2

    def prepare_data(self, stage=None, ROOT="../../watermark/data"):
        mean = [140.562, 133.033, 124.148] 
        std = [60.562, 61.536, 65.176]

        augmented_transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize(mean, std)
        ])
        # Make sure to use a consistent transform for validation/test
        safe_transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize(mean, std)
        ])
        
        if self.opt['loading_method'] == 'Image':
            # Set up a full dataset with the two respective transforms
            cifar_complete_augmented = torchvision.datasets.ImageFolder(root=ROOT, transform=my_transform)
            cifar_complete_train_val = torchvision.datasets.ImageFolder(root=ROOT, transform=train_val_transform)

            # Instead of splitting the dataset in the beginning you can also 
            # split using a sampler. This is not better, but we wanted to 
            # show it off here as an example by using the default
            # ImageFolder dataset :)

            # First regular splitting which we did for you before
            N = len(cifar_complete_augmented)        
            num_train, num_val = int(N*split['train']), int(N*split['val'])
            indices = np.random.permutation(N)
            train_idx, val_idx, test_idx = indices[:num_train], indices[num_train:num_train+num_val], indices[num_train+num_val:]

            # Now we can set the sampler via the respective subsets
            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)
            test_sampler= SubsetRandomSampler(test_idx)
            self.sampler = {"train": train_sampler, "val": val_sampler, "test": test_sampler}

            # assign to use in dataloaders
            self.dataset = {}
            self.dataset["train"], self.dataset["val"], self.dataset["test"] = cifar_complete_augmented,\
                cifar_complete_train_val, cifar_complete_train_val

        elif self.opt['loading_method'] == 'Memory':
            self.dataset = {}
            self.sampler = {}

            for mode in ['train', 'val', 'test']:
                # Set transforms
                if mode == 'train':
                    transform = my_transform
                else:
                    transform = train_val_transform

                self.dataset[mode] = MemoryImageFolderDataset(
                    root = CIFAR_ROOT,
                    transform = transform,
                    mode = mode,
                    split = split
                )
        else:
            raise NotImplementedError("Wrong loading method")

    def return_dataloader_dict(self, mode):
        arg_dict = {
            'batch_size': self.opt["batch_size"],
            'num_workers': self.opt['num_workers'],
            'persistent_workers': True,
            'pin_memory': True
        }
        if self.opt['loading_method'] == 'Image':
            arg_dict['sampler'] = self.sampler[mode]
        elif self.opt['loading_method'] == 'Memory':
            arg_dict['shuffle'] = True if mode == 'train' else False
        return arg_dict

    def train_dataloader(self):
        arg_dict = self.return_dataloader_dict('train')
        return DataLoader(self.dataset["train"], **arg_dict)

    def val_dataloader(self):
        arg_dict = self.return_dataloader_dict('val')
        return DataLoader(self.dataset["val"], **arg_dict)
    
    def test_dataloader(self):
        arg_dict = self.return_dataloader_dict('train')
        return DataLoader(self.dataset["train"], **arg_dict)