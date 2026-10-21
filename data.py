"""data.py: load the raw MNIST images, make the train/test split, preprocess.

Used by: every classifier (all five use the SAME split).

Requirements:
- Load directly from the raw PNG files (one folder per digit, label = folder name).
- Skip non-image files (e.g. .DS_Store).
- Create my own partition: random 80% train / 20% test, fixed seed, stratified by digit.
- Normalize to [0, 1] (convert uint8 -> float32 FIRST).
- Flatten to 784-d vectors when needed; keep 28x28 for the CNN.

TODO:
- [ ] load all images + labels
- [ ] reproducible stratified split (save the indices)
- [ ] normalize
- [ ] flatten
- [ ] (optional) cache the arrays so loading 60k PNGs only happens once
"""
from pathlib import Path
import numpy as np
from PIL import Image

DATA_DIR = Path("data/MNIST")

def load_images(data_dir, max_per_digit=None):
    images, labels = [], []
    for digit in range(10):
        files = sorted((data_dir / str(digit)).glob("*.png"))
        if max_per_digit is not None:
            files = files[:max_per_digit]
        for path in files:
            images.append(np.array(Image.open(path)))
            labels.append(digit)
    return np.array(images), np.array(labels)

def stratified_split(y, test_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    train_parts, test_parts = [], []
    for digit in range(10):
        idx = np.where(y == digit)[0]
        idx = rng.permutation(idx)
        n_test = int(round(test_frac * len(idx)))
        test_parts.append(idx[:n_test])
        train_parts.append(idx[n_test:])
    return np.concatenate(train_parts), np.concatenate(test_parts)

def preprocess(X, flatten=True):
  X = X.astype(np.float32) / 255
  if flatten:
      X = X.reshape(len(X), -1)
  return X

def get_data(flatten=True, max_per_digit=None, seed=0):
    X, y = load_images(DATA_DIR, max_per_digit=max_per_digit)
    train_idx, test_idx = stratified_split(y, seed=seed)
    np.savez("results/split.npz", train_idx=train_idx, test_idx=test_idx)
    X = preprocess(X, flatten=flatten)
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = get_data()
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    # expect (48000, 784) (48000,) (12000, 784) (12000,)
    print(np.bincount(y_test))   # each digit ≈ 20% of its count
