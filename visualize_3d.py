import matplotlib.pyplot as plt


def show_slice(img, title=""):

    plt.imshow(img[:,:,img.shape[2]//2], cmap="gray")
    plt.title(title)
    plt.show()