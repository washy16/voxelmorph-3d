import numpy as np
from pair_selection import select_best_pairs


print("📦 PREPROCESS STARTED (TRAIN/VAL PIPELINE)")

# =========================
# LOAD TRAIN DATA ONLY
# =========================
data = np.load("data/train.npz")
images = data["images"]

print("🔍 IMAGES LOADED:", len(images))

# =========================
# BEST PAIRS (PASPER)
# =========================
pairs = select_best_pairs(images, top_k=10)

print("\n✔ BEST PAIRS:", len(pairs))

mismatch = 0

for idx, (i, j) in enumerate(pairs):

    img1 = images[i]
    img2 = images[j]

    print(f"\nPAIR {idx}: {i} ↔ {j}")

    print("BEFORE:")
    print("T1:", img1.shape)
    print("T2:", img2.shape)

    if img1.shape != img2.shape:
        print("⚠️ MISMATCH")
        mismatch += 1
    else:
        print("✔ OK")

print("\n⚠️ TOTAL MISMATCH:", mismatch)
print("✅ PREPROCESS DONE")