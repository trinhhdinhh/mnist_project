"""main.py: train and test all five classifiers on the SAME split, then compare them.

One command reproduces every number and figure in the report:
    python main.py                      (about 1 minute on CPU)
    python main.py --max-per-digit 100  (quick test)
    python main.py --models mlp cnn     (only some models)
Writes to results/: comparison.md, results.json, confusion_*.png and each model's figures.
"""
import argparse
import json
import time
import numpy as np
import torch
from torch import nn
import knn
import naive_bayes
import linear
from data import get_data
from mlp import MLP
from cnn import CNN
from torch_utils import set_seed, make_loaders, train, predict, plot_history

RESULTS_DIR = "results"

def timed(fn, *args, **kwargs):
    # Call fn and also return how long it took, so every model is timed the same way.
    start = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - start

# Each run_* function trains/tests one model and returns a list of result dicts
# (a list because KNN produces one result per k). Every dict has the same keys,
# so the comparison code at the bottom treats all models identically.

def run_knn(X_train, y_train, X_test, args):
    # One prediction pass covers every k (distances are shared), so all k get the same
    # predict time. KNN has no training step, so train time is 0.
    preds, predict_time = timed(knn.predict, X_train, y_train, X_test, ks=args.ks)
    return [{"model": f"KNN (k={k})", "preds": preds[k], "train_time": 0.0, "predict_time": predict_time,
             "params": {"k": k, "distance": "euclidean", "predict_time_shared_across_k": True}}
            for k in args.ks]

def run_naive_bayes(X_train, y_train, X_test, args):
    (log_prior, theta), train_time = timed(naive_bayes.fit, X_train, y_train, alpha=args.nb_alpha)
    preds, predict_time = timed(naive_bayes.predict, X_test, log_prior, theta)
    naive_bayes.plot_theta(theta, f"{RESULTS_DIR}/nb_theta.png")
    return [{"model": "Naive Bayes", "preds": preds, "train_time": train_time, "predict_time": predict_time,
             "params": {"alpha": args.nb_alpha, "binarize_threshold": 0.5}}]

def run_linear(X_train, y_train, X_test, args):
    (W, losses), train_time = timed(linear.train, X_train, y_train, lr=args.linear_lr,
                                    epochs=args.linear_epochs, batch_size=args.batch_size, seed=args.seed)
    preds, predict_time = timed(linear.predict, W, X_test)
    linear.plot_templates(W, f"{RESULTS_DIR}/linear_W.png")
    linear.plot_losses(losses, f"{RESULTS_DIR}/linear_loss.png")
    return [{"model": "Linear (L2)", "preds": preds, "train_time": train_time, "predict_time": predict_time,
             "params": {"lr": args.linear_lr, "epochs": args.linear_epochs, "batch_size": args.batch_size,
                        "loss": "L2", "optimizer": "mini-batch GD"}}]

def run_torch(name, model_cls, image_shape, lr, epochs, X_train, y_train, X_test, y_test, args, device="cpu"):
    # Shared by the MLP and CNN; they differ only in model_cls and image_shape.
    # Re-seed, then build the model, so its starting weights don't depend on
    # which models ran before it (e.g. running --models cnn alone gives the same CNN).
    set_seed(args.seed)
    model = model_cls()
    train_loader, val_loader, test_loader = make_loaders(X_train, y_train, X_test, y_test, image_shape=image_shape,
                                                         batch_size=args.torch_batch_size, seed=args.seed)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum)
    history, train_time = timed(train, model, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(),
                                epochs=epochs, device=device)
    preds, predict_time = timed(predict, model, test_loader, device)
    plot_history(history, f"{name} training", f"{RESULTS_DIR}/{name.lower()}_history.png")
    return [{"model": name, "preds": preds, "train_time": train_time, "predict_time": predict_time,
             "params": {"lr": lr, "momentum": args.momentum, "epochs": epochs, "batch_size": args.torch_batch_size,
                        "loss": "cross-entropy", "optimizer": "SGD", "device": device,
                        "final_val_acc": float(history["val_acc"][-1])}}]

def confusion_matrix(y_true, y_pred, n_classes=10):
    # cm[i, j] = number of test images of digit i that were predicted as j.
    # The diagonal is correct predictions; everything off it is a mistake.
    cm = np.zeros((n_classes, n_classes), dtype=int)
    # add.at adds 1 for every (true, pred) pair, including repeats (plain cm[...] += 1 wouldn't).
    np.add.at(cm, (y_true, y_pred), 1)
    return cm

def most_confused(cm, n=3):
    # The n largest off-diagonal cells -> [(true digit, predicted digit, count), ...].
    off = cm.copy()
    np.fill_diagonal(off, 0)  # ignore correct predictions
    flat = np.argsort(off, axis=None)[::-1][:n]  # positions of the n biggest counts
    return [(int(i), int(j), int(off[i, j])) for i, j in zip(*np.unravel_index(flat, off.shape))]

def plot_confusion(cm, title, path):
    # Heatmap of the confusion matrix with the count written in each cell.
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm, cmap="Blues")
    for i in range(10):
        for j in range(10):
            # White text on dark cells, black on light ones, so every number is readable.
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def slug(name):
    # Model name -> safe filename part, e.g. "KNN (k=3)" -> "knn_k_3".
    return "".join(ch if ch.isalnum() else "_" for ch in name.lower()).strip("_").replace("__", "_")

def write_table(results, path):
    # The comparison table as Markdown, ready to paste into the report.
    lines = ["| Model | Test accuracy | Train time (s) | Predict time (s) | Most confused (true→pred, count) |",
             "|---|---|---|---|---|"]
    for r in results:
        pairs = ", ".join(f"{i}→{j} ({n})" for i, j, n in r["most_confused"])
        lines.append(f"| {r['model']} | {r['accuracy']:.2%} | {r['train_time']:.1f} | {r['predict_time']:.2f} | {pairs} |")
    table = "\n".join(lines)
    with open(path, "w") as f:
        f.write(table + "\n")
    return table

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and compare all five MNIST classifiers on the same split")
    parser.add_argument("--max-per-digit", type=int, default=None)
    parser.add_argument("--models", nargs="+", default=["knn", "nb", "linear", "mlp", "cnn"],
                        choices=["knn", "nb", "linear", "mlp", "cnn"])
    parser.add_argument("--seed", type=int, default=0)
    # Hyperparameters for each model; defaults are the values used in the report.
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--nb-alpha", type=float, default=1.0)
    parser.add_argument("--linear-lr", type=float, default=0.01)
    parser.add_argument("--linear-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--torch-batch-size", type=int, default=64)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--mlp-lr", type=float, default=0.01)
    parser.add_argument("--mlp-epochs", type=int, default=10)
    parser.add_argument("--cnn-lr", type=float, default=0.01)
    parser.add_argument("--cnn-epochs", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    # Load and split ONCE; every model below gets exactly the same train and test images.
    set_seed(args.seed)
    (X_train, y_train, X_test, y_test), load_time = timed(get_data, flatten=True, max_per_digit=args.max_per_digit,
                                                         seed=args.seed)
    print(f"loaded train {X_train.shape}, test {X_test.shape} in {load_time:.1f} s")

    results = []
    if "knn" in args.models:
        print("\n== KNN ==")
        results += run_knn(X_train, y_train, X_test, args)
    if "nb" in args.models:
        print("\n== Naive Bayes ==")
        results += run_naive_bayes(X_train, y_train, X_test, args)
    if "linear" in args.models:
        print("\n== Linear ==")
        results += run_linear(X_train, y_train, X_test, args)
    if "mlp" in args.models:
        print("\n== MLP ==")
        results += run_torch("MLP", MLP, (784,), args.mlp_lr, args.mlp_epochs,
                             X_train, y_train, X_test, y_test, args)
    if "cnn" in args.models:
        print("\n== CNN ==")
        results += run_torch("CNN", CNN, (1, 28, 28), args.cnn_lr, args.cnn_epochs,
                             X_train, y_train, X_test, y_test, args, device=args.device)

    # Score every model the same way: accuracy, confusion matrix, most-confused pairs.
    for r in results:
        r["accuracy"] = float(np.mean(r["preds"] == y_test))
        cm = confusion_matrix(y_test, r["preds"])
        r["confusion_matrix"] = cm.tolist()
        r["most_confused"] = most_confused(cm)
        plot_confusion(cm, f"{r['model']}: {r['accuracy']:.2%}", f"{RESULTS_DIR}/confusion_{slug(r['model'])}.png")

    print("\n" + write_table(results, f"{RESULTS_DIR}/comparison.md"))
    # Everything except the raw predictions (12,000 numbers per model) goes to JSON.
    with open(f"{RESULTS_DIR}/results.json", "w") as f:
        json.dump({"seed": args.seed, "n_train": len(y_train), "n_test": len(y_test),
                   "results": [{k: v for k, v in r.items() if k != "preds"} for r in results]}, f, indent=2)
    print(f"\nsaved {RESULTS_DIR}/comparison.md, results.json, confusion_*.png and model figures")
