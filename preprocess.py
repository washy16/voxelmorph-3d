import os
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split

from config import RAW_T1, RAW_T2, MAX_SAMPLES

TARGET_SHAPE = (96, 96, 96)

# =========================
# LOAD NIFTI (CLEAN + SAFE)
# =========================
def load_nifti(path):
    img = nib.load(path)

    # 🔥 FIX IMPORTANT: canonical orientation
    img = nib.as_closest_canonical(img)

    data = img.get_fdata()
    data = np.nan_to_num(data)

    return data.astype(np.float32)

# =========================
# NORMALIZATION (ROBUST)
# =========================
def normalize(img):
    min_val = np.percentile(img, 1)
    max_val = np.percentile(img, 99)
    return (img - min_val) / (max_val - min_val + 1e-8)

# =========================
# RESIZE VOLUME
# =========================
def resize_volume(img):
    factors = (
        TARGET_SHAPE[0] / img.shape[0],
        TARGET_SHAPE[1] / img.shape[1],
        TARGET_SHAPE[2] / img.shape[2],
    )
    return zoom(img, factors, order=1)

# =========================
# FILES LOADER
# =========================
def get_files(folder):
    return sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(".nii") or f.endswith(".nii.gz")
    ])

# =========================
# SIMPLE SAFE PAIRING
# =========================
def match_pairs(t1_files, t2_files):
    """
    ⚠️ IMPORTANT:
    Ici on suppose même ordre sujet.
    Si dataset réel avec IDs → améliorer plus tard.
    """
    n = min(len(t1_files), len(t2_files))
    return t1_files[:n], t2_files[:n]

# =========================
# PROCESS PIPELINE
# =========================
def process():

    print("📦 Loading dataset (DOCTORAT PIPELINE)")

    t1_files = get_files(RAW_T1)
    t2_files = get_files(RAW_T2)

    print(f"🔍 T1 found: {len(t1_files)}")
    print(f"🔍 T2 found: {len(t2_files)}")

    assert len(t1_files) > 0
    assert len(t2_files) > 0

    # 🔥 pairing safe
    t1_files, t2_files = match_pairs(t1_files, t2_files)

    t1_files = t1_files[:MAX_SAMPLES]
    t2_files = t2_files[:MAX_SAMPLES]

    fixed_all = []
    moving_all = []

    for i, (t1_path, t2_path) in enumerate(zip(t1_files, t2_files)):

        try:
            # LOAD
            t1 = load_nifti(t1_path)
            t2 = load_nifti(t2_path)

            # SHAPE CHECK
            if t1.shape != t2.shape:
                print(f"⚠️ shape mismatch at {i}: {t1.shape} vs {t2.shape}")

            # NORMALIZE (robust percentile)
            t1 = normalize(t1)
            t2 = normalize(t2)

            # RESIZE (same grid)
            t1 = resize_volume(t1)
            t2 = resize_volume(t2)

            # CHANNEL DIM
            t1 = np.expand_dims(t1, -1)
            t2 = np.expand_dims(t2, -1)

            # FIXED / MOVING (clear semantics)
            fixed_all.append(t1)   # T1 reference
            moving_all.append(t2)  # T2 deformable

            print(f"✔ sample {i+1}/{len(t1_files)}")

        except Exception as e:
            print(f"❌ error {i}: {e}")

    # =========================
    # TO ARRAY
    # =========================
    fixed_all = np.array(fixed_all, dtype=np.float32)
    moving_all = np.array(moving_all, dtype=np.float32)

    # =========================
    # SPLIT
    # =========================
    idx = np.arange(len(fixed_all))

    train_idx, val_idx = train_test_split(
        idx, test_size=0.2, random_state=42
    )

    # =========================
    # SAVE
    # =========================
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

    # =========================
    # DEBUG
    # =========================
    print("\n✅ PREPROCESS COMPLETE")
    print("TRAIN:", fixed_all[train_idx].shape)
    print("VAL:", fixed_all[val_idx].shape)

    print("\n🧠 READY FOR VOXELMORPH")

    if __name__ == "__main__":
        process()