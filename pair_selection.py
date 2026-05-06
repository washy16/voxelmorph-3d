
# pair_selection.py

import numpy as np


def mutual_information(x, y, bins=32):

    x = x.flatten()
    y = y.flatten()

    hist_2d, _, _ = np.histogram2d(x, y, bins=bins)
    hist_x = np.histogram(x, bins=bins)[0]
    hist_y = np.histogram(y, bins=bins)[0]

    hist_2d = hist_2d / (np.sum(hist_2d) + 1e-8)
    hist_x = hist_x / (np.sum(hist_x) + 1e-8)
    hist_y = hist_y / (np.sum(hist_y) + 1e-8)

    mi = np.sum(hist_2d * np.log(hist_2d + 1e-8))

    return mi


def select_best_pairs(images, top_k=50):

    pairs = []
    scores = []

    n = len(images)

    for i in range(n):
        for j in range(i + 1, n):

            score = mutual_information(images[i], images[j])

            pairs.append((i, j))
            scores.append(score)

    best_idx = np.argsort(scores)[::-1][:top_k]

    return [pairs[i] for i in best_idx]
