import os
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split

from config import RAW_T1, RAW_T2, MAX_SAMPLES

TARGET_SHAPE = (96, 96, 96)


# =========================
# LOAD + FORCE RAS ORIENTATION
# =========================
def load_nii_ras(path):

    img = nib.load(path)

    # 🔥 standard medical orientation
    img = nib.as_closest_canonical(img)

    data = img.get_fdata()

    return np.nan_to_num(data).astype(np.float32)


# =========================
# NORMALIZATION
# =========================
def normalize(img):
    return (img - img.min()) / (img.max() - img.min() + 1e-8)


# =========================
# RESIZE TO FIXED VOXEL GRID
# =========================
def resize(img):

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
# MAIN PREPROCESS
# =========================
def preprocess():

    print("📦 PREPROCESS STARTED (RAS + AXIAL FIX)")

    t1_files = get_files(RAW_T1)[:MAX_SAMPLES]
    t2_files = get_files(RAW_T2)[:MAX_SAMPLES]

    print("🔍 T1:", len(t1_files), "| T2:", len(t2_files))

    n = min(len(t1_files), len(t2_files))
    print("🔗 matched pairs:", n)

    fixed, moving = [], []

    mismatch_count = 0

    for i in range(n):

        t1 = load_nii_ras(t1_files[i])
        t2 = load_nii_ras(t2_files[i])

        # =========================
        # DEBUG GEOMETRY
        # =========================
        if t1.shape != t2.shape:
            print(f"⚠️ BEFORE ALIGNMENT {i}: {t1.shape} vs {t2.shape}")
            mismatch_count += 1

        # =========================
        # RESIZE + NORMALIZE
        # =========================
        t1 = normalize(resize(t1))
        t2 = normalize(resize(t2))

        # add channel dim
        t1 = np.expand_dims(t1, -1)
        t2 = np.expand_dims(t2, -1)

        fixed.append(t1)
        moving.append(t2)

        print(f"✔ processed {i+1}/{n}")

    fixed = np.array(fixed, dtype=np.float32)
    moving = np.array(moving, dtype=np.float32)

    # =========================
    # TRAIN / VAL SPLIT
    # =========================
    idx = np.arange(len(fixed))
    train_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=42)

    os.makedirs("data", exist_ok=True)

    np.savez_compressed("data/train.npz",
                        fixed=fixed[train_idx],
                        moving=moving[train_idx])

    np.savez_compressed("data/val.npz",
                        fixed=fixed[val_idx],
                        moving=moving[val_idx])

    print("\n✅ PREPROCESS DONE")
    print("⚠️ mismatches BEFORE RAS fix:", mismatch_count)
    print("TRAIN:", fixed[train_idx].shape)
    print("VAL:", fixed[val_idx].shape)


if __name__ == "__main__":
    preprocess()

