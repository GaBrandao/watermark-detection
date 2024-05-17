import matplotlib.pyplot as plt
import numpy as np

from project.utils.data_transforms import NormalizeInverse
from project.data.config import TRAIN_MEAN, TRAIN_STD

def imshow(img, ax, mean=TRAIN_MEAN, std=TRAIN_STD):
    unormalize = NormalizeInverse(mean, std)
    img = unormalize(img)
    np_img = img.numpy()
    ax.imshow(np.transpose(np_img, (1, 2, 0)))