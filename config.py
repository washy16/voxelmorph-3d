# =========================
# VOXELMORPH 3D CONFIG (STABLE)
# =========================

IMG_SHAPE = (96, 96, 96, 1)

# =========================
# TRAINING
# =========================
LR = 1e-4          # 🔥 important (plus stable que 1e-3)
BATCH_SIZE = 1
EPOCHS = 20        # 5 = trop faible pour voir convergence réelle

# =========================
# REGULARIZATION
# =========================
LAMBDA_REG = 0.01   # stabilise fortement le flow

# =========================
# DATASET (DEBUG LIMIT)
# =========================
MAX_SAMPLES = 10

# =========================
# PATHS
# =========================
DATA_PATH = "data"

TRAIN_FILE = f"{DATA_PATH}/train.npz"
VAL_FILE = f"{DATA_PATH}/val.npz"

RAW_T1 = f"{DATA_PATH}/raw/T1"
RAW_T2 = f"{DATA_PATH}/raw/T2"

CKPT_PATH = "checkpoints"