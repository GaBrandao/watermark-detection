import torch
import torch.nn as nn

import numpy as np
from tqdm.auto import tqdm

from accelerate import Accelerator
import transformers
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup
)

from settings import hparams
from project.data.data_module import DataModule
from project.networks.naive import NaiveModel
from project.utils.models import init_weights
from project.logger.writer import get_summary_writer

def create_tqdm_bar(iterable, desc):
    return tqdm(enumerate(iterable),total=len(iterable), ncols=100, desc=desc)

def train_model(model, hparams=hparams, metric):
    accelerator = Accelerator()

    name = model._get_name()

    if accelerator.is_main_process:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    data_module = DataModule(hparams)

    loss_func = nn.BCELoss()

    train_loader = data_module.get_loader('train')
    valid_loader = data_module.get_loader('valid')

    logger = get_summary_writer(model)

    optimizer = AdamW(params=model.parameters(), lr=hparams['learning_rate'])

    model_name = model._get_name()
    epochs = hparams['epochs']

    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer, 
        num_warmup_steps=100,
        num_training_steps=len(train_loader) * epochs
    )

    progress_bar = tqdm(range(epochs * len(train_loader)), disable=not accelerator.m)

    for epoch in range(epochs):
        model.train() 
        training_loss = []
        for iter, batch in enumerate(train_loader):
            images, labels = batch

            pred = model(images) 
            loss = loss_func(pred, labels.float())
            
            loss.backward()
            optimizer.step()
            scheduler.step() 

            logger.add_scalar(f'model_{name}/train_loss', loss.item(), epoch * len(train_loader) + iter)

            optimizer.zero_grad()

        model.eval()
        validation_loss = []

        for iter, batch in enumerate(valid_loader):
            with torch.no_grad():
                images, labels = batch
                pred = model(images)

                loss = loss_func(pred, labels.float())

                logger.add_scalar(f'classifier_{model_name}/val_loss', loss.item(), epoch * len(valid_loader) + iter)
        
            

    return model

if __name__ == '__main__':
    naive_model = NaiveModel(hparams=hparams)
    naive_model.apply(init_weights)

    train_model()

    # train_dataloader = data_module.get_train_dataloader()
    # test_dataloader = data_module.get_test_dataloader()

    # print(f"Training Acc: {naive_model.get_accuracy(test_dataloader)[1] * 100}%")
    # print(f"Validation Acc: {naive_model.get_accuracy(test_dataloader)[1] * 100}%")
