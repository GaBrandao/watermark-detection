import torch
import torch.nn as nn

import numpy as np
from tqdm import tqdm


class NaiveModel(nn.Module):
    """
    Simple Fully Convulutional Network
    """
    def __init__(self, hparams: dict = {}):
        super().__init__()

        self.device = hparams['device']
        self.hparam = hparams

        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, groups=512),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(512, 2048, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(2048, 1024, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(1024, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)
        return x.view(-1)
    
    def general_step(self, batch, batch_idx, mode):
        images, targets = batch

        # forward pass
        out = self.forward(images)

        # loss
        loss = self.loss(out, targets)

        preds = out.argmax(axis=1)
        n_correct = (targets == preds).sum()
        n_total = len(targets)
        return loss, n_correct, n_total
    
    def general_end(self, outputs, mode):
        # average over all batches aggregated during one epoch
        avg_loss = torch.stack([x[mode + '_loss'] for x in outputs]).mean()
        length = sum([x[mode + '_n_total'] for x in outputs])
        total_correct = torch.stack([x[mode + '_n_correct'] for x in outputs]).sum().cpu().numpy()
        acc = total_correct / length
        return avg_loss, acc

    def training_step(self, batch, batch_idx):
        loss, n_correct, n_total = self.general_step(batch, batch_idx, "train")
        self.log('loss',loss)
        return {'loss': loss, 'train_n_correct':n_correct, 'train_n_total': n_total}

    def validation_step(self, batch, batch_idx):
        loss, n_correct, n_total = self.general_step(batch, batch_idx, "val")
        self.log('val_loss',loss)
        return {'val_loss': loss, 'val_n_correct':n_correct, 'val_n_total': n_total}
    
    def test_step(self, batch, batch_idx):
        loss, n_correct, n_total = self.general_step(batch, batch_idx, "test")
        return {'test_loss': loss, 'test_n_correct':n_correct, 'test_n_total': n_total}

    def validation_epoch_end(self, outputs):
        avg_loss, acc = self.general_end(outputs, "val")
        self.log('val_loss',avg_loss)
        self.log('val_acc',acc)
        return {'val_loss': avg_loss, 'val_acc': acc}

    def get_accuracy(self, dataloader):
        self.model.eval()
        self.model = self.model.to(self.device)

        scores = []
        labels = []

        for batch in tqdm(dataloader):
            X, y = batch
            X = X.to(self.device)
            score = self.forward(X)
            scores.append(score.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

        scores = np.concatenate(scores, axis=0)
        labels = np.concatenate(labels, axis=0)

        preds = np.where(scores > 0.5, 1, 0)
        acc = (labels == preds).mean()
        return preds, acc



