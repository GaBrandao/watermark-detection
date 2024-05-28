import torch
import torch.nn as nn

import numpy as np
from tqdm.autonotebook import tqdm

def create_tqdm_bar(iterable, desc):
    return tqdm(enumerate(iterable),total=len(iterable), ncols=100, desc=desc)

def train_model(model, data_module, loss_func, optimizer, hparams, scheduler=None, logger=None):
    train_loader = data_module.get_loader('train')
    valid_loader = data_module.get_loader('valid')

    device = hparams['device']

    model.to(device)

    model_name = model._get_name()
    epochs = hparams['epochs']
    loss_cutoff = len(train_loader) // 10

    for epoch in range(epochs):
        model.train() 

        training_loss = []
        validation_loss = []

        train_loop = create_tqdm_bar(train_loader, desc=f'Training Epoch [{epoch + 1}/{epochs}]')
        
        for iter, batch in train_loop:
            optimizer.zero_grad() 

            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            pred = model(images) 
            loss = loss_func(pred, labels.unsqueeze(1).float())
            
            loss.backward()  
            optimizer.step() 

            if scheduler is not None:
                scheduler.step() 

            training_loss.append(loss.item())
            training_loss = training_loss[-loss_cutoff:]

            train_loop.set_postfix(curr_train_loss = "{:.8f}".format(np.mean(training_loss)),
                                      lr = "{:.8f}".format(optimizer.param_groups[0]['lr'])
            )

            if logger is not None:
                logger.add_scalar(f'classifier_{model_name}/train_loss', loss.item(), epoch * len(train_loader) + iter)

        model.eval()
        valid_loop = create_tqdm_bar(valid_loader, desc=f'Validation Epoch [{epoch + 1}/{hparams["epochs"]}]')

        with torch.no_grad():
            iter = 0
            for iter, batch in valid_loop:
                images, labels = batch

                images, labels = images.to(device), labels.to(device)

                pred = model(images)
                loss = loss_func(pred, labels.unsqueeze(1).float())
                validation_loss.append(loss.item())

                valid_loop.set_postfix(valid_loss = "{:.8f}".format(np.mean(validation_loss)))

                if logger is not None:
                    logger.add_scalar(f'classifier_{model_name}/val_loss', loss.item(), epoch * len(valid_loader) + iter)            