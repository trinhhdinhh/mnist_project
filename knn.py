"""knn.py: K-Nearest Neighbor classifier, NumPy only.

KNN has no training step: it keeps the training set and labels each test image
by a majority vote among its k closest training images (Euclidean distance).
Run: python knn.py   (add --max-per-digit 100 for a quick test)
"""
import argparse
import time
import numpy as np
from data import get_data

def pairwise_sq_dists(A, X):
    # Squared Euclidean distance from every test row in A (B, 784) to every
    # training row in X (N, 784) -> (B, N).
    # Expanding ||a - x||^2 = ||a||^2 + ||x||^2 - 2 a.x turns the work into one
    # matrix multiply, which is far faster than looping over pairs.
    # No sqrt: it doesn't change which neighbors are closest, so it's wasted work.
    a_sq = np.sum(A ** 2, axis=1)[:, None]   # (B, 1)
    x_sq = np.sum(X ** 2, axis=1)[None, :]   # (1, N)
    D = a_sq + x_sq - 2 * (A @ X.T)          # (B, N)
    # Floating-point rounding can leave tiny negatives; a distance can't be below 0.
    return np.maximum(D, 0)

def k_nearest_labels(D, y_train, k_max):
    # Labels of the k_max closest training images for each test row -> (B, k_max),
    # ordered nearest first.
    # argpartition finds the k_max smallest without sorting all N distances,
    # but leaves them unordered...
    idx = np.argpartition(D, k_max - 1, axis=1)[:, :k_max]
    # ...so sort just those k_max by distance. majority_vote relies on this order.
    order = np.argsort(np.take_along_axis(D, idx, axis=1), axis=1, kind="stable")
    idx = np.take_along_axis(idx, order, axis=1)
    return y_train[idx]

def majority_vote(labels):
    # Most common label in each row of labels (B, k) -> (B,).
    preds = np.empty(len(labels), dtype=labels.dtype)
    for i, row in enumerate(labels):
        counts = np.bincount(row, minlength=10)
        tied = counts == counts.max()
        # Tie rule: argmax would always pick the smallest digit. Instead take the
        # first winning label in the row, i.e. the one with the closest neighbor.
        preds[i] = next(label for label in row if tied[label])
    return preds

def predict(X_train, y_train, X_test, ks=(1, 3, 5), batch_size=1000):
    # Predictions for every k in ks -> {k: (n_test,) array}.
    # Test images go in batches: the full 12,000 x 48,000 distance matrix would
    # need ~2.3 GB; a batch of 1,000 needs ~190 MB.
    k_max = max(ks)
    preds = {k: [] for k in ks}
    for start in range(0, len(X_test), batch_size):
        D = pairwise_sq_dists(X_test[start:start + batch_size], X_train)
        # Find neighbors once for the largest k; smaller k are just the first k columns.
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

    # Sanity check: with k=1 a training image's nearest neighbor is itself, so expect 100%.
    n = min(500, len(X_train))
    self_acc = np.mean(predict(X_train, y_train, X_train[:n], ks=(1,))[1] == y_train[:n])
    print(f"sanity check, k=1 on training images: {self_acc:.2%} (expect 100%)")

    # Training time is zero for KNN, so only prediction is timed.
    start = time.perf_counter()
    preds = predict(X_train, y_train, X_test, ks=args.ks)
    predict_time = time.perf_counter() - start

    print(f"predict time for all k (distances shared): {predict_time:.1f} s")
    for k in args.ks:
        print(f"k={k}: test accuracy {np.mean(preds[k] == y_test):.2%}")
