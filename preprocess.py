import os
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split

from config import RAW_T1, RAW_T2, MAX_SAMPLES

TARGET_SHAPE = (96, 96, 96)

# =========================
# LOAD NIFTI
# =========================
def load_nifti(path):
    img = nib.load(path)
    img = nib.as_closest_canonical(img)
    data = img.get_fdata()
    data = np.nan_to_num(data)
    return data.astype(np.float32)

# =========================
# NORMALIZATION (robuste)
# =========================
def normalize(img):
    min_val = img.min()
    max_val = img.max()
    return (img - min_val) / (max_val - min_val + 1e-8)

# =========================
# RESIZE 3D VOLUME
# =========================
def resize_volume(img):
    factors = (
        TARGET_SHAPE[0] / img.shape[0],
        TARGET_SHAPE[1] / img.shape[1],
        TARGET_SHAPE[2] / img.shape[2],
    )
    return zoom(img, factors, order=1)

# =========================
# GET FILES
# =========================
def get_files(folder):
    return sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(".nii") or f.endswith(".nii.gz")
    ])

# =========================
# PROCESS DATASET
# =========================
def process():

    print("📦 Loading T1 / T2 dataset...")

    t1_files = get_files(RAW_T1)
    t2_files = get_files(RAW_T2)

    print(f"🔍 FOUND T1: {len(t1_files)}")
    print(f"🔍 FOUND T2: {len(t2_files)}")

    # sécurité
    assert len(t1_files) > 0, "❌ No T1 files found"
    assert len(t2_files) > 0, "❌ No T2 files found"

    # limiter samples si besoin
    t1_files = t1_files[:MAX_SAMPLES]
    t2_files = t2_files[:MAX_SAMPLES]

    fixed_all = []
    moving_all = []

    # =========================
    # BUILD PAIRS T2 → T1
    # =========================
    for i in range(min(len(t1_files), len(t2_files))):

        try:
            # LOAD
            t1 = load_nifti(t1_files[i])
            t2 = load_nifti(t2_files[i])

            # NORMALIZE
            t1 = normalize(t1)
            t2 = normalize(t2)

            # RESIZE
            t1 = resize_volume(t1)
            t2 = resize_volume(t2)

            # ADD CHANNEL DIM
            t1 = np.expand_dims(t1, -1)
            t2 = np.expand_dims(t2, -1)

            # STORE
            fixed_all.append(t1)   # T1 = reference
            moving_all.append(t2)  # T2 = deformable

            print(f"✔ processed {i+1}")

        except Exception as e:
            print(f"❌ ERROR at sample {i}: {e}")

    # =========================
    # TO ARRAY
    # =========================
    fixed_all = np.array(fixed_all, dtype=np.float32)
    moving_all = np.array(moving_all, dtype=np.float32)

    # =========================
    # SPLIT TRAIN / VAL
    # =========================
    idx = np.arange(len(fixed_all))

    train_idx, val_idx = train_test_split(
        idx,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    train_fixed = fixed_all[train_idx]
    train_moving = moving_all[train_idx]

    val_fixed = fixed_all[val_idx]
    val_moving = moving_all[val_idx]

    # =========================
    # SAVE DATA
    # =========================
    os.makedirs("data", exist_ok=True)

    np.savez_compressed(
        "data/train.npz",
        fixed=train_fixed,
        moving=train_moving
    )

    np.savez_compressed(
        "data/val.npz",
        fixed=val_fixed,
        moving=val_moving
    )

    # =========================
    # DEBUG INFO
    # =========================
    print("\n✅ PREPROCESS DONE")
    print("TRAIN:", train_fixed.shape, train_moving.shape)
    print("VAL:", val_fixed.shape, val_moving.shape)

    print("\n🧠 READY FOR VOXELMORPH TRAINING (T2 → T1)")


# =========================
# RUN
# =========================
if __name__ == "__main__":
    process()