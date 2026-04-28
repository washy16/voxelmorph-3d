import matplotlib.pyplot as plt
import numpy as np

def show_slice(img, title=""):
    # sécurité : conversion float + normalisation
    img = np.array(img)

    # normalisation (IMPORTANT pour visibilité)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    # slice centrale
    slice_idx = img.shape[2] // 2
    slice_img = img[:, :, slice_idx]

    plt.figure(figsize=(5,5))
    plt.imshow(slice_img, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()