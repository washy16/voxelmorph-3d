# data_loader_3d.py

import numpy as np
from pair_selection import select_best_pairs


def load_data():

    data = np.load("data/train.npz")

    images = data["images"]

    pairs = select_best_pairs(images, top_k=50)

    fixed = []
    moving = []

    for i, j in pairs:
        fixed.append(images[i])
        moving.append(images[j])

    fixed = np.array(fixed)
    moving = np.array(moving)

    split = int(len(fixed) * 0.8)

    train_f = fixed[:split]
    train_m = moving[:split]

    val_f = fixed[split:]
    val_m = moving[split:]

    return train_f, train_m, val_f, val_m