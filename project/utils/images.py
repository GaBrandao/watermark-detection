import matplotlib.pyplot as plt
import numpy as np

from utils.data_transforms import NormalizeInverse


def imshow(img, ax, mean=TRAIN_MEAN, std=TRAIN_STD):
    unormalize = NormalizeInverse(mean, std)
    img = unormalize(img)
    np_img = img.numpy()
    ax.imshow(np.transpose(np_img, (1, 2, 0)))