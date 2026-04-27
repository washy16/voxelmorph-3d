import numpy as np

def fix_shape(x):

    x = np.asarray(x).astype(np.float32)

    # si déjà (D,H,W,1,1) → on corrige
    while len(x.shape) > 4:
        x = np.squeeze(x, -1)

    # garantit channel dim
    if len(x.shape) == 4:
        x = x[..., None]

    return x


def load_npz_file(path):

    data = np.load(path)

    keys = list(data.keys())

    fixed = fix_shape(data[keys[0]])
    moving = fix_shape(data[keys[1]])

    return fixed, moving


def load_data():

    print("📦 Loading train/val NPZ")

    train_f, train_m = load_npz_file("data/train.npz")
    val_f, val_m = load_npz_file("data/val.npz")

    print("✔ TRAIN:", train_f.shape)
    print("✔ VAL:", val_f.shape)

    return train_f, train_m, val_f, val_m