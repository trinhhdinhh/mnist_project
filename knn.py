"""knn.py: K-Nearest Neighbor classifier. NumPy ONLY.

Requirements:
- Input: flattened 784-d vectors from data.py.
- Euclidean distance (required).
- Classify by majority vote among the k nearest training images.
- Test at least k = 1, 3, 5.
- No scikit-learn (no KNeighborsClassifier).

Blueprint:
1. "Training" = just keep X_train and y_train (KNN has no parameters to learn).
2. For each batch of test images:
   a. pairwise_sq_dists  -> (B, N) squared Euclidean distances to every training image
   b. k_nearest_labels   -> labels of the k_max closest images, sorted nearest-first
   c. majority_vote      -> one prediction per test image, for every k (reuses step b)
3. Accuracy = fraction of predictions equal to y_test.

Decisions (record the reasoning in docs/journal.md):
- Ties: among the tied labels, pick the one whose neighbor is closest.
- Batch size 1000: a (1000, 48000) float32 distance matrix is ~190 MB;
  the full (12000, 48000) one would be ~2.3 GB.
"""
import argparse
import time
import numpy as np
from data import get_data

def pairwise_sq_dists(A, X):
    """Squared Euclidean distances between every row of A (B, D) and every row of X (N, D).

    Uses ||a - x||^2 = ||a||^2 + ||x||^2 - 2 a.x, so the whole batch is one matrix
    multiply instead of building a (B, N, 784) array of differences.
    We skip the sqrt: it is monotonic, so the nearest neighbors are the same
    with or without it (the metric is still Euclidean).
    """
    a_sq = np.sum(A ** 2, axis=1)[:, None]   # (B, 1): broadcasts across columns
    x_sq = np.sum(X ** 2, axis=1)[None, :]   # (1, N): broadcasts down rows
    D = a_sq + x_sq - 2 * (A @ X.T)          # (B, N)
    # Rounding can make the distance between near-identical images slightly negative.
    return np.maximum(D, 0)

def k_nearest_labels(D, y_train, k_max):
    """Labels of the k_max nearest training images for each row of D, nearest first -> (B, k_max)."""
    # argpartition finds the k_max smallest in O(N) instead of sorting all N columns,
    # but returns them in no particular order...
    idx = np.argpartition(D, k_max - 1, axis=1)[:, :k_max]
    # ...so sort just those k_max columns by distance (needed for the tie rule).
    order = np.argsort(np.take_along_axis(D, idx, axis=1), axis=1, kind="stable")
    idx = np.take_along_axis(idx, order, axis=1)
    return y_train[idx]

def majority_vote(labels):
    """Most common label in each row of labels (B, k), which is sorted nearest-first."""
    preds = np.empty(len(labels), dtype=labels.dtype)
    for i, row in enumerate(labels):
        counts = np.bincount(row, minlength=10)
        tied = counts == counts.max()
        # Plain argmax would always favor the smallest digit in a tie. Instead,
        # walk the neighbors nearest-first and take the first one with a winning label.
        preds[i] = next(label for label in row if tied[label])
    return preds

def predict(X_train, y_train, X_test, ks=(1, 3, 5), batch_size=1000):
    """Predict X_test for every k in ks. Distances are computed once per batch and reused for all k."""
    k_max = max(ks)
    preds = {k: [] for k in ks}
    for start in range(0, len(X_test), batch_size):
        D = pairwise_sq_dists(X_test[start:start + batch_size], X_train)
        labels = k_nearest_labels(D, y_train, k_max)
        for k in ks:
            preds[k].append(majority_vote(labels[:, :k]))
    return {k: np.concatenate(p) for k, p in preds.items()}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KNN on MNIST")
    parser.add_argument("--max-per-digit", type=int, default=None,
                        help="use only this many images per digit (e.g. 100 for a quick test)")
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5])
    args = parser.parse_args()

    X_train, y_train, X_test, y_test = get_data(flatten=True, max_per_digit=args.max_per_digit)
    print(f"train {X_train.shape}, test {X_test.shape}")

    # Sanity check: with k=1 every training image's nearest neighbor is itself -> 100%.
    n = min(500, len(X_train))
    self_acc = np.mean(predict(X_train, y_train, X_train[:n], ks=(1,))[1] == y_train[:n])
    print(f"sanity check, k=1 on training images: {self_acc:.2%} (expect 100%)")

    start = time.perf_counter()
    preds = predict(X_train, y_train, X_test, ks=args.ks)
    predict_time = time.perf_counter() - start

    print(f"predict time for all k (distances shared): {predict_time:.1f} s")
    for k in args.ks:
        print(f"k={k}: test accuracy {np.mean(preds[k] == y_test):.2%}")
