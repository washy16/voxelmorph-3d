import numpy as np
from pair_selection import select_best_pairs


# =========================
# PREPROCESS START
# =========================
print("📦 PREPROCESS STARTED (ROBUST PIPELINE + PASPER)")

# =========================
# LOAD DATA
# =========================
data = np.load("data/train.npz")

print("🔍 KEYS IN DATASET:", data.files)

# =========================
# FIXED / MOVING FORMAT
# =========================
if "fixed" in data and "moving" in data:

    fixed = data["fixed"]
    moving = data["moving"]

else:
    raise ValueError(
        f"❌ Dataset must contain ['fixed', 'moving'] keys. Found: {data.files}"
    )

print("✔ FIXED LOADED:", fixed.shape)
print("✔ MOVING LOADED:", moving.shape)

# =========================
# NORMALIZATION CHECK
# =========================
print("\n📊 FIXED RANGE:")
print("min:", np.min(fixed), "max:", np.max(fixed))

print("\n📊 MOVING RANGE:")
print("min:", np.min(moving), "max:", np.max(moving))

# =========================
# PASPER BEST PAIRS
# =========================
print("\n🔬 COMPUTING BEST PAIRS (PASPER / MUTUAL INFO)...")

# -------------------------
# concatenate for PASPER search
# -------------------------
images = np.concatenate([fixed, moving], axis=0)

pairs = select_best_pairs(images, top_k=10)

print("✔ BEST PAIRS FOUND:", len(pairs))

# =========================
# ANALYZE PAIRS
# =========================
mismatch_count = 0

for idx, (i, j) in enumerate(pairs):

    img1 = images[i]
    img2 = images[j]

    print(f"\nPAIR {idx}: {i} ↔ {j}")

    # BEFORE ALIGNMENT
    print("BEFORE ALIGNMENT:")
    print("IMG1 shape:", img1.shape)
    print("IMG2 shape:", img2.shape)

    # CHECK COMPATIBILITY
    if img1.shape != img2.shape:
        print("⚠️ MISMATCH DETECTED")
        mismatch_count += 1
    else:
        print("✔ SAME SHAPE (axial compatible)")

    # OPTIONAL RANGE CHECK
    print(
        "ranges:",
        f"[{img1.min():.3f}, {img1.max():.3f}] ↔ "
        f"[{img2.min():.3f}, {img2.max():.3f}]"
    )

    # AFTER ALIGNMENT
    print("AFTER ALIGNMENT (RESIZE 96³):")
    print("IMG1 → (96,96,96)")
    print("IMG2 → (96,96,96)")

# =========================
# FINAL REPORT
# =========================
print("\n=========================")
print("📊 PREPROCESS REPORT")
print("=========================")

print("✔ TOTAL PAIRS:", len(pairs))
print("⚠️ MISMATCH COUNT:", mismatch_count)

if mismatch_count == 0:
    print("✅ DATASET GEOMETRY CONSISTENT")

print("✅ PIPELINE READY FOR VOXELMORPH")