"""naive_bayes.py: Naive Bayes classifier on binarized pixels, NumPy only.

Each pixel is treated as an on/off coin flip whose probability depends on the digit.
Training counts how often each pixel is on for each digit; prediction uses Bayes' rule
and assumes pixels are independent given the digit (the "naive" part).
Run: python naive_bayes.py   (add --max-per-digit 100 for a quick test)
"""
import argparse
import time
import numpy as np
from data import get_data

def binarize(X):
    # Pixels in [0, 1] -> 1 if brighter than 0.5, else 0 (required by the assignment).
    # Turns each pixel into an on/off feature that a Bernoulli probability can model.
    return (X > 0.5).astype(np.float32)

def fit(X, y, alpha=1.0, n_classes=10):
    # Learn the model from counts. Returns:
    #   log_prior (10,):     log P(digit)
    #   theta     (10, 784): P(pixel j is on | digit c)
    Xb = binarize(X)
    counts = np.bincount(y, minlength=n_classes)   # number of training images per digit
    log_prior = np.log(counts / len(y))
    # For each digit, how many of its images have each pixel on -> (10, 784).
    on_counts = np.stack([Xb[y == c].sum(axis=0) for c in range(n_classes)])
    # Laplace smoothing: pretend we saw alpha extra "on" and alpha extra "off" images.
    # Without it, a pixel never on for a digit gets theta = 0, log(0) = -inf, and a
    # single stray pixel would rule that digit out completely.
    theta = (on_counts + alpha) / (counts[:, None] + 2 * alpha)
    return log_prior, theta

def predict(X, log_prior, theta):
    # Bayes' rule: P(c | x) is proportional to P(c) * prod_j P(x_j | c).
    # Work in log space: multiplying 784 probabilities underflows to 0 in floating point,
    # but adding their logs doesn't. Taking the log doesn't change which class is largest.
    Xb = binarize(X)
    # For each pixel: log(theta) if it's on, log(1 - theta) if it's off.
    # The two matrix multiplies do that sum over all 784 pixels for all images and
    # classes at once -> (N, 10) log-posterior scores (up to a constant).
    log_post = Xb @ np.log(theta).T + (1 - Xb) @ np.log(1 - theta).T + log_prior
    return np.argmax(log_post, axis=1)

def plot_theta(theta, path="results/nb_theta.png"):
    # Show theta for each digit as a 28x28 image: brighter = pixel more often on.
    # These come out as blurry "average" digits, showing what the model has learned.
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.5))
    for c, ax in enumerate(axes.flat):
        # Fixed 0-1 range so brightness means the same probability in every panel.
        ax.imshow(theta[c].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"P(on | {c})")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Naive Bayes on MNIST")
    parser.add_argument("--max-per-digit", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()

    X_train, y_train, X_test, y_test = get_data(flatten=True, max_per_digit=args.max_per_digit)
    print(f"train {X_train.shape}, test {X_test.shape}")

    # "Training" is just counting, so it takes a fraction of a second.
    start = time.perf_counter()
    log_prior, theta = fit(X_train, y_train, alpha=args.alpha)
    train_time = time.perf_counter() - start

    start = time.perf_counter()
    preds = predict(X_test, log_prior, theta)
    predict_time = time.perf_counter() - start

    print(f"alpha={args.alpha}: test accuracy {np.mean(preds == y_test):.2%}")
    print(f"train time {train_time:.2f} s, predict time {predict_time:.2f} s")

    plot_theta(theta)
    print("saved results/nb_theta.png")
