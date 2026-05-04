import numpy as np
import os

from config import TRAIN_FILE, VAL_FILE


def load_data():

    print("📦 Loading dataset (LOCAL)")

    # =========================
    # CHECK FILES
    # =========================
    if not os.path.exists(TRAIN_FILE):
        raise FileNotFoundError(f"❌ TRAIN FILE NOT FOUND: {TRAIN_FILE}")

    if not os.path.exists(VAL_FILE):
        raise FileNotFoundError(f"❌ VAL FILE NOT FOUND: {VAL_FILE}")

    # =========================
    # LOAD
    # =========================
    train = np.load(TRAIN_FILE)
    val = np.load(VAL_FILE)

    train_fixed = train["fixed"]
    train_moving = train["moving"]

    val_fixed = val["fixed"]
    val_moving = val["moving"]

    # =========================
    # DEBUG PRINT
    # =========================
    print("✔ TRAIN SHAPE:", train_fixed.shape)
    print("✔ VAL SHAPE:", val_fixed.shape)

    # =========================
    # SAFETY CHECKS
    # =========================
    if len(train_fixed) == 0:
        raise ValueError("❌ TRAIN DATA EMPTY")

    if len(val_fixed) == 0:
        raise ValueError("❌ VAL DATA EMPTY")

    # éviter bug dtype
    train_fixed = train_fixed.astype(np.float32)
    train_moving = train_moving.astype(np.float32)
    val_fixed = val_fixed.astype(np.float32)
    val_moving = val_moving.astype(np.float32)

    return train_fixed, train_moving, val_fixed, val_moving