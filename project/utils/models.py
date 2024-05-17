import torch
import torch.nn as nn

def number_of_parameters(model):
    n_params = sum(p.numel() for p in model.parameters())
    return n_params

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)