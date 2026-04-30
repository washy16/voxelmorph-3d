import numpy as np
from config import TRAIN_FILE, VAL_FILE


def load_data():

    print("📦 Loading dataset from Drive")

    train = np.load(TRAIN_FILE)
    val = np.load(VAL_FILE)

    train_fixed = train["fixed"]
    train_moving = train["moving"]

    val_fixed = val["fixed"]
    val_moving = val["moving"]

    print("✔ TRAIN:", train_fixed.shape)
    print("✔ VAL:", val_fixed.shape)

    return train_fixed, train_moving, val_fixed, val_moving