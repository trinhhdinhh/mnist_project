"""torch_utils.py: shared PyTorch helpers for the MLP and CNN.

Both neural networks use the same data pipeline, training loop and evaluation,
so they live here once instead of being copied into mlp.py and cnn.py.
"""
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from data import stratified_split

def set_seed(seed=0):
    # Seed every random number generator a run might touch, so weight initialization
    # (and anything else random) is the same every time.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def to_dataset(X, y, image_shape):
    # NumPy arrays -> a PyTorch dataset of (image, label) pairs.
    # image_shape is (784,) for the MLP or (1, 28, 28) for the CNN (1 = one gray channel).
    X = torch.from_numpy(X).float().reshape(-1, *image_shape)
    # [0, 1] -> [-1, 1]: inputs centered around 0 tend to train faster and more stably.
    X = (X - 0.5) / 0.5
    # CrossEntropyLoss expects integer class labels as int64 ("long").
    return TensorDataset(X, torch.from_numpy(y).long())

def make_loaders(X_train, y_train, X_test, y_test, image_shape=(784,), val_frac=0.1, batch_size=64, seed=0):
    # Returns train / validation / test DataLoaders.
    # The validation set is carved out of TRAIN (10%, stratified like the main split).
    # It's used to watch training and choose hyperparameters; the test set is only
    # used once at the end, so the reported test accuracy stays an honest estimate.
    fit_idx, val_idx = stratified_split(y_train, test_frac=val_frac, seed=seed)
    # A seeded generator makes the batch shuffling reproducible.
    generator = torch.Generator().manual_seed(seed)
    # Shuffle training batches each epoch so the model doesn't see the digits in a fixed order.
    train_loader = DataLoader(to_dataset(X_train[fit_idx], y_train[fit_idx], image_shape),
                              batch_size=batch_size, shuffle=True, generator=generator)
    # No shuffling for evaluation; large batches just make it faster.
    val_loader = DataLoader(to_dataset(X_train[val_idx], y_train[val_idx], image_shape), batch_size=1000)
    test_loader = DataLoader(to_dataset(X_test, y_test, image_shape), batch_size=1000)
    return train_loader, val_loader, test_loader

@torch.no_grad()  # no gradients needed for prediction: saves memory and time
def predict(model, loader, device="cpu"):
    # Predicted digit for every image in loader -> NumPy array.
    model.eval()  # switch to evaluation mode (matters for layers like dropout/batch norm)
    # The model outputs 10 scores (logits) per image; the highest one is the prediction.
    return torch.cat([model(X.to(device)).argmax(dim=1).cpu() for X, _ in loader]).numpy()

def accuracy(model, loader, device="cpu"):
    # Fraction of images in loader that the model classifies correctly.
    y = torch.cat([y for _, y in loader]).numpy()
    return np.mean(predict(model, loader, device) == y)

def train(model, train_loader, val_loader, optimizer, loss_fn=None, epochs=10, device="cpu"):
    # Standard PyTorch training loop. Returns the average training loss and the
    # validation accuracy for each epoch (plotted by plot_history).
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    model.to(device)
    history = {"train_loss": [], "val_acc": []}
    for epoch in range(epochs):
        model.train()  # back to training mode (predict() switched it to eval)
        total, n = 0.0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            # The four steps of every update:
            optimizer.zero_grad()        # 1. clear gradients left over from the last batch
            loss = loss_fn(model(X), y)  # 2. forward pass + loss
            loss.backward()              # 3. backpropagation: autograd computes all gradients
            optimizer.step()             # 4. update the weights using those gradients
            # Weight by batch size so the epoch average is correct even if the last batch is smaller.
            total += loss.item() * len(X)
            n += len(X)
        history["train_loss"].append(total / n)
        history["val_acc"].append(accuracy(model, val_loader, device))
        print(f"epoch {epoch + 1}/{epochs}: train loss {history['train_loss'][-1]:.4f}, "
              f"val acc {history['val_acc'][-1]:.2%}")
    return history

def plot_history(history, title, path):
    # Training curves side by side: loss should fall; validation accuracy should rise
    # and level off. Validation accuracy dropping while loss keeps falling = overfitting.
    import matplotlib.pyplot as plt
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))
    ax1.plot(epochs, history["train_loss"], marker="o")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("train loss")
    ax2.plot(epochs, history["val_acc"], marker="o")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("validation accuracy")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
