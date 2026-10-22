"""mlp.py: Multilayer Perceptron, PyTorch, trained with SGD.

Like the linear classifier, but with hidden layers and ReLU in between, so it can
learn nonlinear combinations of pixels (e.g. "this stroke AND that loop").
Default architecture: 784 -> 256 -> ReLU -> 128 -> ReLU -> 10.
Run: python mlp.py   (add --max-per-digit 100 for a quick test)
"""
import argparse
import time
import numpy as np
import torch
from torch import nn
from data import get_data
from torch_utils import set_seed, make_loaders, train, predict, plot_history

class MLP(nn.Module):
    def __init__(self, hidden=(256, 128), n_inputs=784, n_classes=10):
        super().__init__()
        # Build Linear -> ReLU for each hidden size, so --hidden can change the
        # architecture without editing code (useful for the hidden-size sweep).
        layers, size = [], n_inputs
        for h in hidden:
            # Without the ReLU, stacked Linear layers collapse into one linear map,
            # and the MLP could do no better than the linear classifier.
            layers += [nn.Linear(size, h), nn.ReLU()]
            size = h
        # Output layer: 10 raw scores (logits). No softmax here, because
        # CrossEntropyLoss applies it internally.
        layers.append(nn.Linear(size, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch, 784) -> (batch, 10)
        return self.net(x)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLP on MNIST")
    parser.add_argument("--max-per-digit", type=int, default=None)
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Seed before creating the model, so the initial weights are reproducible.
    set_seed(args.seed)
    # Flattened images, same as KNN / Naive Bayes / Linear.
    X_train, y_train, X_test, y_test = get_data(flatten=True, max_per_digit=args.max_per_digit)
    train_loader, val_loader, test_loader = make_loaders(X_train, y_train, X_test, y_test,
                                                         batch_size=args.batch_size, seed=args.seed)
    print(f"train {len(train_loader.dataset)}, val {len(val_loader.dataset)}, test {len(test_loader.dataset)}")

    model = MLP(hidden=args.hidden)
    # SGD as the assignment requires. Momentum keeps a running average of past gradients,
    # which smooths out noisy mini-batch steps; --momentum 0 gives plain SGD.
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    start = time.perf_counter()
    history = train(model, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(), epochs=args.epochs)
    train_time = time.perf_counter() - start

    # The test set is used only here, once training is finished.
    start = time.perf_counter()
    preds = predict(model, test_loader)
    predict_time = time.perf_counter() - start

    print(f"hidden={args.hidden}, lr={args.lr}, momentum={args.momentum}, epochs={args.epochs}: "
          f"test accuracy {np.mean(preds == y_test):.2%}")
    print(f"train time {train_time:.1f} s, predict time {predict_time:.3f} s")

    plot_history(history, "MLP training", "results/mlp_history.png")
    print("saved results/mlp_history.png")
