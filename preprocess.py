import os
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from config import RAW_T1, RAW_T2, TRAIN_FILE

print("📦 PREPROCESS START (IXI + ROBUST PIPELINE)")

# =========================
# TARGET SHAPE
# =========================
TARGET_SHAPE = (96, 96, 96)

# =========================
# LOAD FILES
# =========================
t1_files = [f for f in os.listdir(RAW_T1) if "T1" in f and (f.endswith(".nii") or f.endswith(".nii.gz"))]
t2_files = [f for f in os.listdir(RAW_T2) if "T2" in f and (f.endswith(".nii") or f.endswith(".nii.gz"))]

print("✔ T1 files:", len(t1_files))
print("✔ T2 files:", len(t2_files))

# =========================
# EXTRACT PATIENT ID (IXI FORMAT)
# =========================
def get_id(fname):
    name = fname.replace(".nii.gz", "").replace(".nii", "")

    if "-T1" in name:
        return name.replace("-T1", "")
    if "-T2" in name:
        return name.replace("-T2", "")

    return name

t1_dict = {get_id(f): f for f in t1_files}
t2_dict = {get_id(f): f for f in t2_files}

common_ids = sorted(set(t1_dict.keys()) & set(t2_dict.keys()))

print("✔ MATCHED PAIRS:", len(common_ids))
print("⚠️ UNMATCHED T1:", len(t1_dict) - len(common_ids))
print("⚠️ UNMATCHED T2:", len(t2_dict) - len(common_ids))

if len(common_ids) == 0:
    raise ValueError("❌ No matching T1/T2 pairs found!")

# =========================
# LOAD NIFTI
# =========================
def load_nii(path):
    img = nib.load(path)
    data = img.get_fdata().astype(np.float32)
    return data

# =========================
# RESAMPLE FUNCTION
# =========================
def resize_3d(volume):
    factors = (
        TARGET_SHAPE[0] / volume.shape[0],
        TARGET_SHAPE[1] / volume.shape[1],
        TARGET_SHAPE[2] / volume.shape[2],
    )
    return zoom(volume, factors, order=1)

# =========================
# BUILD DATASET
# =========================
fixed_list = []
moving_list = []

for i, pid in enumerate(common_ids):

    print(f"🔄 Processing {i+1}/{len(common_ids)} : {pid}")

    p1 = os.path.join(RAW_T1, t1_dict[pid])
    p2 = os.path.join(RAW_T2, t2_dict[pid])

    fixed = load_nii(p1)
    moving = load_nii(p2)

    # resize
    fixed = resize_3d(fixed)
    moving = resize_3d(moving)

    # normalize per volume
    fixed = (fixed - fixed.min()) / (fixed.max() - fixed.min() + 1e-8)
    moving = (moving - moving.min()) / (moving.max() - moving.min() + 1e-8)

    fixed_list.append(fixed)
    moving_list.append(moving)

# =========================
# STACK
# =========================
fixed = np.stack(fixed_list).astype(np.float32)
moving = np.stack(moving_list).astype(np.float32)

# add channel dimension
fixed = fixed[..., np.newaxis]
moving = moving[..., np.newaxis]

print("✔ FINAL FIXED:", fixed.shape)
print("✔ FINAL MOVING:", moving.shape)

# =========================
# SAVE DATASET
# =========================
os.makedirs(os.path.dirname(TRAIN_FILE), exist_ok=True)

np.savez_compressed(
    TRAIN_FILE,
    fixed=fixed,
    moving=moving
)

print("💾 SAVED:", TRAIN_FILE)
print("🚀 PREPROCESS COMPLETE")