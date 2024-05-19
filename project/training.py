import torch

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import numpy as np

from project.data.data_module import DataModule


def create_tqdm_bar(iterable, desc):
    return tqdm(enumerate(iterable),total=len(iterable), ncols=150, desc=desc)


def train_model(model, data_module : DataModule, loss_func, logger, hparams):
    train_dataloader = data_module.get_train_dataloader()
    valid_dataloader = data_module.get_valid_dataloader()

    loss_cutoff = len(train_dataloader) // 10
    optimizer = torch.optim.Adam(model.parameters(), hparams['learning_rate'])

    # The scheduler is used to change the learning rate every few "n" steps.
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(hparams['epochs'] * len(train_dataloader) / 5), gamma=hparams.get('gamma', 0.8))

    for epoch in range(hparams['epochs']):

        # Training stage, where we want to update the parameters.
        model.train()  # Set the model to training mode

        training_loss = []
        validation_loss = []

        # Create a progress bar for the training loop.
        training_loop = create_tqdm_bar(train_dataloader, desc=f'Training Epoch [{epoch + 1}/{hparams['epochs']}]')
        for train_iteration, batch in training_loop:
            optimizer.zero_grad() # Reset the gradients - VERY important! Otherwise they accumulate.
            images, labels = batch # Get the images and labels from the batch, in the fashion we defined in the dataset and dataloader.
            images, labels = images.to(hparams['device']), labels.to(hparams['device']) # Send the data to the device (GPU or CPU) - it has to be the same device as the model.

            pred = model(images) # Stage 1: Forward().
            loss = loss_func(pred, labels) # Compute the loss over the predictions and the ground truth.
            loss.backward()  # Stage 2: Backward().
            optimizer.step() # Stage 3: Update the parameters.
            # scheduler.step() # Update the learning rate.

            training_loss.append(loss.item())
            training_loss = training_loss[-loss_cutoff:]

            # Update the progress bar.
            training_loop.set_postfix(curr_train_loss = "{:.8f}".format(np.mean(training_loss)),
                                      lr = "{:.8f}".format(optimizer.param_groups[0]['lr'])
            )

            # Update the tensorboard logger.
            logger.add_scalar(f'classifier_{hparams["name"]}/train_loss', loss.item(), epoch * len(train_dataloader) + train_iteration)

        # Validation stage, where we don't want to update the parameters. Pay attention to the classifier.eval() line
        # and "with torch.no_grad()" wrapper.
        model.eval()
        val_loop = create_tqdm_bar(valid_dataloader, desc=f'Validation Epoch [{epoch + 1}/{hparams["epochs"]}]')

        with torch.no_grad():
            for val_iteration, batch in val_loop:
                images, labels = batch
                images, labels = images.to(hparams['device']), labels.to(hparams['device'])

                pred = model(images)
                loss = loss_func(pred, labels)
                validation_loss.append(loss.item())
                # Update the progress bar.
                val_loop.set_postfix(val_loss = "{:.8f}".format(np.mean(validation_loss)))

                # Update the tensorboard logger.
                logger.add_scalar(f'classifier_{hparams["name"]}/val_loss', loss.item(), epoch * len(valid_dataloader) + val_iteration)




