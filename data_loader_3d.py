import numpy as np
import os
from config import TRAIN_FILE, VAL_FILE

# =========================
# LOAD DATA
# =========================
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
    # LOAD NPZ
    # =========================
    train = np.load(TRAIN_FILE)
    val = np.load(VAL_FILE)

    train_fixed = train["fixed"]
    train_moving = train["moving"]

    val_fixed = val["fixed"]
    val_moving = val["moving"]

    # =========================
    # BASIC DEBUG
    # =========================
    print("✔ TRAIN SHAPE:", train_fixed.shape)
    print("✔ VAL SHAPE:", val_fixed.shape)

    # =========================
    # SANITY CHECKS
    # =========================
    assert train_fixed.shape == train_moving.shape, "❌ TRAIN mismatch fixed/moving"
    assert val_fixed.shape == val_moving.shape, "❌ VAL mismatch fixed/moving"

    assert len(train_fixed) > 0, "❌ TRAIN EMPTY"
    assert len(val_fixed) > 0, "❌ VAL EMPTY"

    # =========================
    # TYPE SAFETY
    # =========================
    train_fixed = np.asarray(train_fixed, dtype=np.float32)
    train_moving = np.asarray(train_moving, dtype=np.float32)
    val_fixed = np.asarray(val_fixed, dtype=np.float32)
    val_moving = np.asarray(val_moving, dtype=np.float32)

    # =========================
    # VALUE CHECK (IMPORTANT)
    # =========================
    print("📊 Fixed range:", train_fixed.min(), train_fixed.max())
    print("📊 Moving range:", train_moving.min(), train_moving.max())

    return train_fixed, train_moving, val_fixed, val_moving