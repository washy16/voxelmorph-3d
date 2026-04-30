import numpy as np
import matplotlib.pyplot as plt


# =========================
# 🔥 VISUALISATION 3D SLICE
# =========================
def show_slice(volume, title=""):

    volume = np.array(volume)

    # sécurité : éviter images vides
    if volume.max() == volume.min():
        print(f"⚠️ Volume vide ou constant : {title}")
        return

    # normalisation
    volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)

    # slice centrale
    z = volume.shape[2] // 2
    slice_img = volume[:, :, z]

    plt.figure(figsize=(5, 5))
    plt.imshow(slice_img, cmap="gray")
    plt.title(title)
    plt.axis("off")

    plt.show()   # ✅ IMPORTANT POUR COLAB


# =========================
# 🔥 FUNCTION PRINCIPALE
# =========================
def visualize():

    print("🚀 VISUALIZATION STARTED")

    dummy = np.random.rand(64, 64, 64)

    show_slice(dummy, "TEST FIXED (dummy)")
    show_slice(dummy, "TEST MOVING (dummy)")
    show_slice(dummy, "TEST WARPED (dummy)")

    print("✅ VISUALIZATION FINISHED")


# =========================
# RUN LOCAL (optionnel)
# =========================
if __name__ == "__main__":
    visualize()