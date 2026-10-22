"""data.py: load the raw MNIST images, split them into train/test, and preprocess.

Every classifier gets its data from get_data(), so all five models are trained and
tested on exactly the same images. That is what makes the comparison fair.
"""
from pathlib import Path
import numpy as np
from PIL import Image

DATA_DIR = Path("data/MNIST")

def load_images(data_dir, max_per_digit=None):
    # Returns X: (N, 28, 28) uint8 pixels and y: (N,) digit labels.
    # max_per_digit lets us test on a small subset (e.g. 100 per digit) before the full 60,000.
    images, labels = [], []
    for digit in range(10):
        # Only *.png files are loaded, so .DS_Store and other stray files are skipped.
        # sorted() fixes the order, so index i means the same image on every run;
        # the saved split indices depend on that.
        files = sorted((data_dir / str(digit)).glob("*.png"))
        if max_per_digit is not None:
            files = files[:max_per_digit]
        for path in files:
            images.append(np.array(Image.open(path)))
            labels.append(digit)  # the label is the folder name
    # Stack the lists into single arrays: much faster to work with than Python lists.
    return np.array(images), np.array(labels)

def stratified_split(y, test_frac=0.2, seed=0):
    # Returns index arrays (not images), so the same split can be applied to any version
    # of X (flattened, 2D, normalized...). Also reused by torch_utils to carve out validation.
    # A fixed seed makes the "random" shuffle identical every run (reproducibility).
    rng = np.random.default_rng(seed)
    train_parts, test_parts = [], []
    # Stratified: split each digit separately, so every digit is 80/20 on its own.
    # A plain random split could, by chance, put slightly more 5s in test than 1s,
    # and the digit counts are already uneven (5,421 fives vs 6,742 ones).
    for digit in range(10):
        idx = np.where(y == digit)[0]   # positions of every image of this digit
        idx = rng.permutation(idx)      # shuffle them
        n_test = int(round(test_frac * len(idx)))
        test_parts.append(idx[:n_test])   # first 20% -> test
        train_parts.append(idx[n_test:])  # remaining 80% -> train
    return np.concatenate(train_parts), np.concatenate(test_parts)

def preprocess(X, flatten=True):
  # Convert to float32 BEFORE any arithmetic: uint8 math wraps around (e.g. 3 - 5 = 254).
  # Dividing by 255 scales pixels from 0-255 to [0, 1], so features are on a small,
  # consistent scale (and Naive Bayes can binarize at 0.5).
  X = X.astype(np.float32) / 255
  if flatten:
      # (N, 28, 28) -> (N, 784): KNN, Naive Bayes, Linear and MLP treat each image as one
      # 784-long vector. -1 tells NumPy to work out that size itself (28 * 28).
      X = X.reshape(len(X), -1)
  return X

def get_data(flatten=True, max_per_digit=None, seed=0):
    # The one function every model calls. Returns X_train, y_train, X_test, y_test.
    X, y = load_images(DATA_DIR, max_per_digit=max_per_digit)
    train_idx, test_idx = stratified_split(y, seed=seed)
    # Save the indices so the exact split can be checked or reused later.
    # Note: a small max_per_digit run overwrites this file with a small split;
    # a full run (no max_per_digit) restores the real 48,000 / 12,000 one.
    np.savez("results/split.npz", train_idx=train_idx, test_idx=test_idx)
    X = preprocess(X, flatten=flatten)
    # Indexing with the same index array keeps every image paired with its label.
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = get_data()
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    # expect (48000, 784) (48000,) (12000, 784) (12000,)
    print(np.bincount(y_test))   # each digit ≈ 20% of its count
