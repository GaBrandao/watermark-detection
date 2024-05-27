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

        self.hparam = hparams

        self.layer1 = nn.Sequential(nn.Conv2d(3, 96, 11, stride=4), nn.ReLU(), nn.MaxPool2d(3, stride=2))
        
        self.layer2 = nn.Sequential(nn.Conv2d(96, 256, 5, padding=2), nn.ReLU(), nn.MaxPool2d(3, stride=2))

        self.layer3 = nn.Sequential(nn.Conv2d(256, 384, 3, padding=1), nn.ReLU())#, nn.MaxPool2d(2))

        self.layer4 = nn.Sequential(nn.Conv2d(384, 384, 3), nn.ReLU())#, nn.MaxPool2d(2))
            
        self.layer5 = nn.Sequential(nn.Conv2d(384, 256, 3), nn.ReLU(), nn.MaxPool2d(3, stride=2))
        
        self.layer6 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(9216, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x.view(-1)
    
    def general_step(self, batch, loss_func):
        images, targets = batch

        # forward pass
        out = self.forward(images)

        # loss
        loss = loss_func(out, targets.float())

        preds = (out.detach().cpu().numpy()) > 0.5
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

        scores = []
        labels = []

        for batch in tqdm(dataloader):
            X, y = batch
            X = X.to(self.hparams['device'])
            score = self.forward(X)
            scores.append(score.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

        scores = np.concatenate(scores, axis=0)
        labels = np.concatenate(labels, axis=0)

        preds = np.where(scores > 0.5, 1, 0)
        acc = (labels == preds).mean()
        return preds, acc



