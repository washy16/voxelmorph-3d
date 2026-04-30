# =========================
# VOXELMORPH 3D CONFIG
# =========================

IMG_SHAPE = (96, 96, 96, 1)

# training
LR = 1e-4
BATCH_SIZE = 1
EPOCHS = 30

# regularization (flow smoothness)
LAMBDA_REG = 1.0

# =========================
# PATHS (DRIVE ONLY)
# =========================
DRIVE_ROOT = "/content/drive/MyDrive/voxelmorph-3d"

DATA_PATH = f"{DRIVE_ROOT}/data"

TRAIN_FILE = f"{DATA_PATH}/train.npz"
VAL_FILE = f"{DATA_PATH}/val.npz"

RAW_T1 = f"{DATA_PATH}/raw/T1"
RAW_T2 = f"{DATA_PATH}/raw/T2"

CKPT_PATH = f"{DRIVE_ROOT}/checkpoints"