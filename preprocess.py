# preprocess.py

import numpy as np
from pair_selection import select_best_pairs


print("📦 PREPROCESS STARTED (PASPER PIPELINE)")

data = np.load("data/raw.npz")
images = data["images"]

print("🔍 TOTAL IMAGES:", len(images))

pairs = select_best_pairs(images, top_k=10)

print("\n✔ BEST PAIRS SELECTED:", len(pairs))

mismatches = 0

for idx, (i, j) in enumerate(pairs):

    img1 = images[i]
    img2 = images[j]

    print(f"\nPAIR {idx}: {i} ↔ {j}")

    print("BEFORE ALIGNMENT:")
    print("T1:", img1.shape)
    print("T2:", img2.shape)

    if img1.shape != img2.shape:
        print("⚠️ MISMATCH DETECTED")
        mismatches += 1
    else:
        print("✔ SAME SHAPE")

    print("AFTER ALIGNMENT (placeholder 96³)")
    print("T1 → (96,96,96)")
    print("T2 → (96,96,96)")

print("\n⚠️ TOTAL MISMATCHES:", mismatches)
print("✅ PREPROCESS DONE")
