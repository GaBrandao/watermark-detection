import torch
import torch.nn as nn

import numpy as np
import os

from accelerate import Accelerator
from accelerate.tracking import TensorBoardTracker
from accelerate.utils import (
    LoggerType,
    tqdm
)
import transformers
from transformers import get_linear_schedule_with_warmup


from settings import hparams
from project.data.data_module import DataModule
from project.networks.naive import NaiveModel
from project.utils.models import init_weights
from project.logger.writer import get_writer_path

os.environ["KAGGLE_TPU"] = "yes" # adding a fake env to launch on TPUs
os.environ["TPU_NAME"] = "dummy"
# make the TPU available accelerator to torch-xla
os.environ["XRT_TPU_CONFIG"]="localservice;0;localhost:51011"

def train_model(model, args):
    logs_path = get_writer_path(model)
    model_name = model._get_name()

    accelerator = Accelerator(
        log_with=TensorBoardTracker(
            run_name=model_name, 
            logging_dir=logs_path
    ))

    if accelerator.is_main_process:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    data_module = DataModule(hparams)

    loss_func = nn.BCELoss()

    train_loader = data_module.get_loader('train')
    valid_loader = data_module.get_loader('valid')

    accelerator.init_trackers(model_name, config=hparams)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=hparams['learning_rate'])

    epochs = hparams['epochs']
    
    model, optimizer, train_loader, valid_loader = accelerator.prepare(
        model, optimizer, train_loader, valid_loader
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer, 
        num_warmup_steps=100,
        num_training_steps=len(train_loader) * epochs
    )

    progress_bar = tqdm(range(epochs * len(train_loader)))

    device = accelerator.device

    for epoch in range(epochs):
        model.train() 
        
        for iter, batch in enumerate(train_loader):
            images, labels = batch

            pred = model(images) 
            loss = loss_func(pred, labels.float()).to(device)
            
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            accelerator.log({'train_loss': loss.item()}, step=iter)
            progress_bar.update(1)

        model.eval()
        for iter, batch in enumerate(valid_loader):
            with torch.no_grad():
                images, labels = batch
                pred = model(images)
            loss = loss_func(pred, labels.float()).to(device)
            loss = accelerator.gather_for_metrics(loss.item())

            accelerator.log({'valid_loss': loss}, step=iter)
        
    accelerator.end_training()
    return model

if __name__ == '__main__':
    naive_model = NaiveModel(hparams=hparams)
    naive_model.apply(init_weights)

    train_model(naive_model, hparams)

    # train_dataloader = data_module.get_train_dataloader()
    # test_dataloader = data_module.get_test_dataloader()

    # print(f"Training Acc: {naive_model.get_accuracy(test_dataloader)[1] * 100}%")
    # print(f"Validation Acc: {naive_model.get_accuracy(test_dataloader)[1] * 100}%")
