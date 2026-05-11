import numpy as np
from pair_selection import select_best_pairs


def load_data():

    data = np.load("data/train.npz")

    fixed = data["fixed"]
    moving = data["moving"]

    print("✔ FIXED:", fixed.shape)
    print("✔ MOVING:", moving.shape)

    # PASPER (best pairs sur fixed/moving index pairing)
    n = len(fixed)

    pairs = [(i, i) for i in range(n)]  # pairing simple stable

    fixed_list = []
    moving_list = []

    for i, j in pairs:
        fixed_list.append(fixed[i])
        moving_list.append(moving[j])

    fixed = np.array(fixed_list)
    moving = np.array(moving_list)

    split = int(len(fixed) * 0.8)

    train_f = fixed[:split]
    train_m = moving[:split]

    val_f = fixed[split:]
    val_m = moving[split:]

    return train_f, train_m, val_f, val_m