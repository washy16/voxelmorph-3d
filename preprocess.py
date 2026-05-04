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
# NORMALIZATION
# =========================
def normalize(img):
    p1 = np.percentile(img, 1)
    p99 = np.percentile(img, 99)
    return (img - p1) / (p99 - p1 + 1e-8)

# =========================
# RESIZE
# =========================
def resize_volume(img):
    factors = (
        TARGET_SHAPE[0] / img.shape[0],
        TARGET_SHAPE[1] / img.shape[1],
        TARGET_SHAPE[2] / img.shape[2],
    )
    return zoom(img, factors, order=1)

# =========================
# FILES
# =========================
def get_files(folder):
    return sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(".nii") or f.endswith(".nii.gz")
    ])

# =========================
# PAIRING SAFE
# =========================
def match_pairs(t1_files, t2_files):
    n = min(len(t1_files), len(t2_files))
    return t1_files[:n], t2_files[:n]

# =========================
# PROCESS PIPELINE
# =========================
def process():

    print("📦 PREPROCESS STARTED")

    t1_files = get_files(RAW_T1)
    t2_files = get_files(RAW_T2)

    print(f"🔍 T1: {len(t1_files)} | T2: {len(t2_files)}")

    assert len(t1_files) > 0, "No T1 files"
    assert len(t2_files) > 0, "No T2 files"

    t1_files, t2_files = match_pairs(t1_files, t2_files)

    t1_files = t1_files[:MAX_SAMPLES]
    t2_files = t2_files[:MAX_SAMPLES]

    fixed_all = []
    moving_all = []

    for i, (t1_path, t2_path) in enumerate(zip(t1_files, t2_files)):

        try:
            t1 = load_nifti(t1_path)
            t2 = load_nifti(t2_path)

            if t1.shape != t2.shape:
                print(f"⚠️ shape mismatch {i}: {t1.shape} vs {t2.shape}")

            t1 = normalize(t1)
            t2 = normalize(t2)

            t1 = resize_volume(t1)
            t2 = resize_volume(t2)

            t1 = np.expand_dims(t1, -1)
            t2 = np.expand_dims(t2, -1)

            fixed_all.append(t1)   # T1 fixed
            moving_all.append(t2)  # T2 moving

            print(f"✔ {i+1}/{len(t1_files)}")

        except Exception as e:
            print(f"❌ error {i}: {e}")

    fixed_all = np.array(fixed_all, dtype=np.float32)
    moving_all = np.array(moving_all, dtype=np.float32)

    idx = np.arange(len(fixed_all))
    train_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=42)

    os.makedirs("data", exist_ok=True)

    np.savez_compressed(
        "data/train.npz",
        fixed=fixed_all[train_idx],
        moving=moving_all[train_idx]
    )

    np.savez_compressed(
        "data/val.npz",
        fixed=fixed_all[val_idx],
        moving=moving_all[val_idx]
    )

    print("\n✅ PREPROCESS DONE")
    print("TRAIN:", fixed_all[train_idx].shape)
    print("VAL:", fixed_all[val_idx].shape)

# =========================
# ENTRY POINT (IMPORTANT)
# =========================
if __name__ == "__main__":
    process()