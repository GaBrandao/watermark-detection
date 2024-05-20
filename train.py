import torch
import torch.nn as nn

import numpy as np
from tqdm import tqdm

try:
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.core.xla_model as xm
except ImportError:
    xmp = None

from settings import hparams
from project.data.data_module import DataModule
from project.networks.naive import NaiveModel
from project.utils.models import init_weights
from project.logger.writer import get_summary_writer

def create_tqdm_bar(iterable, desc):
    return tqdm(enumerate(iterable),total=len(iterable), ncols=100, desc=desc)

def train_model(index):
    data_module = DataModule(hparams)

    naive_model = NaiveModel(hparams=hparams)
    naive_model.apply(init_weights)

    model = NaiveModel(hparams)

    loss_func = nn.BCELoss()

    train_loader = data_module.get_loader('train')
    valid_loader = data_module.get_loader('valid')

    logger = get_summary_writer(model)

    loss_cutoff = len(train_loader) // 10
    optimizer = torch.optim.Adam(model.parameters(), hparams['learning_rate'])

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        train_loader = pl.MpDeviceLoader(train_loader, device)
        valid_loader = pl.MpDeviceLoader(valid_loader, device)
    else:
        device = xm.xla_device()

    model.to(device)

    model_name = model._get_name()
    epochs = hparams['epochs']

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(epochs * len(train_dataloader) / 5), gamma=hparams.get('gamma', 0.8))

    for epoch in range(epochs):
        model.train() 

        training_loss = []
        validation_loss = []

        # training_loop = create_tqdm_bar(train_loader, desc=f'Training Epoch [{epoch + 1}/{epochs}]')
        for images, labels in train_loader:# train_iteration, batch in training_loop:
            optimizer.zero_grad() 
            images, labels = batch 
            images, labels = images.to(device), labels.to(device)

            pred = model(images) 
            loss = loss_func(pred, labels.float())
            
            loss.backward()  

            if xmp is not None:
                xm.optimizer_step(optimizer)
            else:
                optimizer.step() 
            # scheduler.step() 

            training_loss.append(loss.item())
            training_loss = training_loss[-loss_cutoff:]

            # training_loop.set_postfix(curr_train_loss = "{:.8f}".format(np.mean(training_loss)),
            #                           lr = "{:.8f}".format(optimizer.param_groups[0]['lr'])
            # )

            logger.add_scalar(f'classifier_{model_name}/train_loss', loss.item(), epoch * len(train_loader) + train_iteration)

        model.eval()
        val_loop = create_tqdm_bar(valid_loader, desc=f'Validation Epoch [{epoch + 1}/{hparams["epochs"]}]')

        with torch.no_grad():
            for val_iteration, batch in val_loop:
                images, labels = batch
                images, labels = images.to(device), labels.to(device)

                pred = model(images)
                loss = loss_func(pred, labels.float())
                validation_loss.append(loss.item())

                val_loop.set_postfix(val_loss = "{:.8f}".format(np.mean(validation_loss)))

                logger.add_scalar(f'classifier_{model_name}/val_loss', loss.item(), epoch * len(valid_loader) + val_iteration)

    return model

if __name__ == '__main__':
    if xmp is not None:
        xmp.spawn(train_model, args=())
    else:
        train_model()

    # train_dataloader = data_module.get_train_dataloader()
    # test_dataloader = data_module.get_test_dataloader()

    # print(f"Training Acc: {naive_model.get_accuracy(test_dataloader)[1] * 100}%")
    # print(f"Validation Acc: {naive_model.get_accuracy(test_dataloader)[1] * 100}%")
