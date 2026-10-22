"""linear.py: linear classifier y = Wx with L2 loss, trained by gradient descent. NumPy only.

Each digit gets one row of W; a digit's score is the dot product of its row with
the image, and the highest score wins. Forward pass, gradient and update are all
written by hand (NumPy rather than PyTorch, so the gradient can be checked numerically).
Run: python linear.py   (add --max-per-digit 100 for a quick test)
"""
import argparse
import time
import numpy as np
from data import get_data

def add_bias(X):
    # Append a constant 1 to every image: (N, 784) -> (N, 785).
    # The last column of W then acts as the bias, so the model stays exactly y = Wx.
    return np.hstack([X, np.ones((len(X), 1), dtype=X.dtype)])

def one_hot(y, n_classes=10):
    # Labels -> target vectors (N, 10), e.g. 3 -> [0,0,0,1,0,0,0,0,0,0].
    # L2 loss needs a target vector to compare the 10 scores against.
    T = np.zeros((len(y), n_classes), dtype=np.float32)
    T[np.arange(len(y)), y] = 1
    return T

def forward(W, Xb):
    # Scores for every image and class: (N, 785) @ (785, 10) -> (N, 10).
    # This is y = Wx for all images at once (each row of the result is W x_n).
    return Xb @ W.T

def l2_loss(W, Xb, T):
    # L = (1 / 2N) * sum_n ||W x_n - t_n||^2
    # The 1/2 cancels the 2 from differentiating the square; averaging over N
    # keeps the loss (and the step size) independent of the batch size.
    return 0.5 * np.mean(np.sum((forward(W, Xb) - T) ** 2, axis=1))

def l2_grad(W, Xb, T):
    # dL/dW = (1 / N) * (XW^T - T)^T X  -> (10, 785), same shape as W.
    # (XW^T - T) is the error for every image and class; multiplying by X
    # sums each pixel's contribution to that error over the batch.
    return (forward(W, Xb) - T).T @ Xb / len(Xb)

def grad_check(W, Xb, T, n_checks=20, eps=1e-5, seed=0):
    # Compare l2_grad against a numerical estimate at random entries of W.
    # Central difference: dL/dW_ij ~ (L(W_ij + eps) - L(W_ij - eps)) / (2 eps).
    # Returns the largest relative error; below ~1e-6 means the derivation is right.
    rng = np.random.default_rng(seed)
    analytic = l2_grad(W, Xb, T)
    errors = []
    for _ in range(n_checks):
        i, j = rng.integers(W.shape[0]), rng.integers(W.shape[1])
        W_plus, W_minus = W.copy(), W.copy()
        W_plus[i, j] += eps
        W_minus[i, j] -= eps
        numeric = (l2_loss(W_plus, Xb, T) - l2_loss(W_minus, Xb, T)) / (2 * eps)
        # Relative (not absolute) error, so tiny and large gradients are judged fairly;
        # the 1e-12 avoids dividing by zero when both are 0.
        errors.append(abs(numeric - analytic[i, j]) / max(abs(numeric) + abs(analytic[i, j]), 1e-12))
    return max(errors)

def train(X, y, lr=0.01, epochs=20, batch_size=100, seed=0):
    # Mini-batch gradient descent. Returns the learned W (10, 785) and the
    # training loss after each epoch.
    rng = np.random.default_rng(seed)
    Xb, T = add_bias(X), one_hot(y)
    # Starting from zeros is fine here: L2 loss on a linear model is convex
    # (one bowl-shaped minimum), so the start point doesn't change where we end up.
    W = np.zeros((T.shape[1], Xb.shape[1]), dtype=np.float32)
    losses = []
    for epoch in range(epochs):
        # Reshuffle every epoch so batches differ; the seed keeps it reproducible.
        order = rng.permutation(len(Xb))
        for start in range(0, len(Xb), batch_size):
            batch = order[start:start + batch_size]
            # Step downhill: move W against the gradient, scaled by the learning rate.
            W -= lr * l2_grad(W, Xb[batch], T[batch])
        # Loss on the full training set, to check it goes down (plotted in linear_loss.png).
        losses.append(l2_loss(W, Xb, T))
        print(f"epoch {epoch + 1}/{epochs}: train loss {losses[-1]:.4f}")
    return W, losses

def predict(W, X):
    # The predicted digit is the class with the highest score.
    return np.argmax(forward(W, add_bias(X)), axis=1)

def plot_templates(W, path="results/linear_W.png"):
    # Show each row of W (minus the bias) as a 28x28 image: a "template" for that digit.
    # Red pixels raise the digit's score when they're on, blue pixels lower it.
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.5))
    for c, ax in enumerate(axes.flat):
        w = W[c, :784].reshape(28, 28)
        # Symmetric color range, so white means exactly 0 in every panel.
        lim = np.abs(w).max()
        ax.imshow(w, cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set_title(f"W row {c}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def plot_losses(losses, path="results/linear_loss.png"):
    # Training loss per epoch: should fall quickly, then flatten out.
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(range(1, len(losses) + 1), losses, marker="o")
    ax.set_xlabel("epoch")
    ax.set_ylabel("train L2 loss")
    ax.set_title("Linear classifier training loss")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Linear classifier (L2 loss) on MNIST")
    parser.add_argument("--max-per-digit", type=int, default=None)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()

    X_train, y_train, X_test, y_test = get_data(flatten=True, max_per_digit=args.max_per_digit)
    print(f"train {X_train.shape}, test {X_test.shape}")

    # Check the gradient before training with it. Uses a small random W and 50 images,
    # in float64: float32 rounding would swamp the tiny eps differences.
    rng = np.random.default_rng(0)
    W_check = rng.normal(0, 0.01, (10, 785))
    rel_err = grad_check(W_check, add_bias(X_train[:50].astype(np.float64)), one_hot(y_train[:50]).astype(np.float64))
    print(f"gradient check, max relative error: {rel_err:.2e} (expect < 1e-6)")

    start = time.perf_counter()
    W, losses = train(X_train, y_train, lr=args.lr, epochs=args.epochs, batch_size=args.batch_size)
    train_time = time.perf_counter() - start

    start = time.perf_counter()
    preds = predict(W, X_test)
    predict_time = time.perf_counter() - start

    print(f"lr={args.lr}, epochs={args.epochs}, batch={args.batch_size}: test accuracy {np.mean(preds == y_test):.2%}")
    print(f"train time {train_time:.2f} s, predict time {predict_time:.3f} s")

    plot_templates(W)
    plot_losses(losses)
    print("saved results/linear_W.png and results/linear_loss.png")
