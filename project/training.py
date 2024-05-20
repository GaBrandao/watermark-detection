import torch

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import numpy as np

from project.data.data_module import DataModule

try:
    import torch_xla.core.xla_model as xm
except ImportError:
    xm = None

def create_tqdm_bar(iterable, desc):
    return tqdm(enumerate(iterable),total=len(iterable), ncols=150, desc=desc)


def train_model(model, data_module : DataModule, loss_func, logger, hparams):
    train_dataloader = data_module.get_train_dataloader()
    valid_dataloader = data_module.get_valid_dataloader()

    loss_cutoff = len(train_dataloader) // 10
    optimizer = torch.optim.Adam(model.parameters(), hparams['learning_rate'])

    if torch.backends.mps.is_available():
        device = torch.device("mps")
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

        training_loop = create_tqdm_bar(train_dataloader, desc=f'Training Epoch [{epoch + 1}/{epochs}]')
        for train_iteration, batch in training_loop:
            optimizer.zero_grad() 
            images, labels = batch 
            images, labels = images.to(device), labels.to(device)

            pred = model(images) 
            loss = loss_func(pred, labels.float())
            
            loss.backward()  

            if xm is not None:
                xm.optimizer_step(optimizer)
            else:
                optimizer.step() 
            # scheduler.step() 

            training_loss.append(loss.item())
            training_loss = training_loss[-loss_cutoff:]

            training_loop.set_postfix(curr_train_loss = "{:.8f}".format(np.mean(training_loss)),
                                      lr = "{:.8f}".format(optimizer.param_groups[0]['lr'])
            )

            logger.add_scalar(f'classifier_{model_name}/train_loss', loss.item(), epoch * len(train_dataloader) + train_iteration)

        model.eval()
        val_loop = create_tqdm_bar(valid_dataloader, desc=f'Validation Epoch [{epoch + 1}/{hparams["epochs"]}]')

        with torch.no_grad():
            for val_iteration, batch in val_loop:
                images, labels = batch
                images, labels = images.to(device), labels.to(device)

                pred = model(images)
                loss = loss_func(pred, labels.float())
                validation_loss.append(loss.item())

                val_loop.set_postfix(val_loss = "{:.8f}".format(np.mean(validation_loss)))

                logger.add_scalar(f'classifier_{model_name}/val_loss', loss.item(), epoch * len(valid_dataloader) + val_iteration)




