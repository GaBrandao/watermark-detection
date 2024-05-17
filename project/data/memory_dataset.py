import torch

class MemoryDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.data = []
        for image, label in dataset:
            self.data.append((image, label))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image, label = self.data[index]
        return image, label