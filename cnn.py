"""cnn.py: Convolutional Neural Network, PyTorch, trained with cross-entropy loss.

Unlike the other four models, the CNN keeps each image as a 28x28 grid. Small 3x3
filters slide over it looking for local patterns (edges, curves), so it uses the fact
that neighboring pixels belong together, which flattening throws away.
Run: python cnn.py   (add --max-per-digit 100 for a quick test, --device mps for the Mac GPU)
"""
import argparse
import time
import numpy as np
import torch
from torch import nn
from data import get_data
from torch_utils import set_seed, make_loaders, train, predict, plot_history

class CNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        # Feature extractor. Shapes for one image (channels x height x width):
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),   # 1x28x28 -> 32x26x26  (no padding: 28 - 3 + 1 = 26)
            nn.ReLU(),
            nn.MaxPool2d(2),                   # -> 32x13x13  (keep the max of each 2x2 block)
            nn.Conv2d(32, 64, kernel_size=3),  # -> 64x11x11  (13 - 3 + 1 = 11)
            nn.ReLU(),
            nn.MaxPool2d(2),                   # -> 64x5x5    (11 / 2 rounds down to 5)
        )
        # Pooling halves the size, so the next layer's filters see a wider area of the
        # original image, and small shifts of the digit matter less.
        # Classifier: the same idea as the MLP, applied to the learned features.
        self.classifier = nn.Sequential(
            nn.Flatten(),                # 64x5x5 -> 1600
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            # 10 logits; CrossEntropyLoss applies the softmax, so there's no Softmax layer here.
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        # x: (batch, 1, 28, 28) -> (batch, 10)
        return self.classifier(self.features(x))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CNN on MNIST")
    parser.add_argument("--max-per-digit", type=int, default=None)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", help="cpu, or mps for the Apple GPU (faster, less reproducible)")
    args = parser.parse_args()

    set_seed(args.seed)
    # Load flattened like the other models (same split), then make_loaders reshapes
    # back to (1, 28, 28) via image_shape.
    X_train, y_train, X_test, y_test = get_data(flatten=True, max_per_digit=args.max_per_digit)
    train_loader, val_loader, test_loader = make_loaders(X_train, y_train, X_test, y_test, image_shape=(1, 28, 28),
                                                         batch_size=args.batch_size, seed=args.seed)
    print(f"train {len(train_loader.dataset)}, val {len(val_loader.dataset)}, test {len(test_loader.dataset)}")

    model = CNN()
    # Same optimizer settings as the MLP, so differences come from the architecture.
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    start = time.perf_counter()
    history = train(model, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(),
                    epochs=args.epochs, device=args.device)
    train_time = time.perf_counter() - start

    start = time.perf_counter()
    preds = predict(model, test_loader, device=args.device)
    predict_time = time.perf_counter() - start

    print(f"lr={args.lr}, momentum={args.momentum}, epochs={args.epochs}: "
          f"test accuracy {np.mean(preds == y_test):.2%}")
    print(f"train time {train_time:.1f} s, predict time {predict_time:.2f} s")

    plot_history(history, "CNN training", "results/cnn_history.png")
    print("saved results/cnn_history.png")
