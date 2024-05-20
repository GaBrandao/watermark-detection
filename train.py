import os

import torch
import torch.nn as nn
try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
except ImportError:
    xm = None

from settings import hparams
from project.data.data_module import DataModule

from project.networks.naive import NaiveModel
from project.utils.models import init_weights, number_of_parameters

from project.training import train_model

from torch.utils.tensorboard import SummaryWriter

from logs.settings import LOGS_ROOT

if __name__ == '__main__':
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = xm.xla_device()

    print('Current device:', device)

    hparams['device'] = device
    data_module = DataModule(hparams)

    naive_model = NaiveModel(hparams=hparams).to(device)
    naive_model.apply(init_weights)

    print('# Parameters: ', number_of_parameters(naive_model))

    logs_path = os.path.join(LOGS_ROOT, naive_model._get_name())

    num_of_runs = len(os.listdir(logs_path)) if os.path.exists(logs_path) else 0

    logs_path = os.path.join(logs_path, f'run_{num_of_runs + 1}')
    logger = SummaryWriter(logs_path)

    loss = nn.BCELoss()

    if xmp is not None:
        xmp.spawn(train_model, args=(naive_model, data_module, loss, logger, hparams))
    else:
        train_model(naive_model, data_module, loss, logger, hparams)

    train_dataloader = data_module.get_train_dataloader()
    test_dataloader = data_module.get_test_dataloader()

    print(f"Training Acc: {naive_model.get_accuracy(test_dataloader)[1] * 100}%")
    print(f"Validation Acc: {naive_model.get_accuracy(test_dataloader)[1] * 100}%")
