import os
from torch.utils.tensorboard import SummaryWriter

from .settings import LOGS_ROOT

def get_writer_path(model):
    logs_path = os.path.join(LOGS_ROOT, model._get_name())

    num_of_runs = len(os.listdir(logs_path)) if os.path.exists(logs_path) else 0

    logs_path = os.path.join(logs_path, f'run_{num_of_runs + 1}')
    return logs_path