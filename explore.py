"""explore.py: a first look at the raw MNIST folder before writing any model code.

Checks the facts the rest of the project relies on: how many images per digit,
what one image looks like as numbers, and what the digits look like as pictures.
Run once: python explore.py
"""
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Relative path, so scripts must be run from the project root.
DATA_DIR = Path("data/MNIST")

def count_per_digit(data_dir):  # -> dict {digit: count}
    counts = {}
    for item in data_dir.iterdir():
        if item.is_dir():
            # The folder name IS the label (e.g. "3"); only .png files count,
            # so stray files like .DS_Store don't inflate the totals.
            label = item.name
            n = sum(1 for f in item.iterdir() if f.suffix == ".png")
            counts[label] = n
    return counts

def inspect_one(path):
    # Converting the PIL image to a NumPy array shows what the models will actually see:
    # a 28x28 grid of uint8 brightness values (0 = black background, 255 = white ink).
    img = np.array(Image.open(path))
    print(f"shape={img.shape}  dtype={img.dtype}  min={img.min()}  max={img.max()}")
    # Most of each image is background, which is why Naive Bayes and the linear
    # classifier's W maps end up mostly empty around the edges.
    print(f"fraction of black (0) pixels: {(img == 0).mean():.2f}")

def save_sample_grid(data_dir, out_path, n_per_digit=10):
    # One row per digit, n_per_digit examples per row: a quick visual check that
    # each folder really holds the digit its name says (labels come from folder names).
    fig, axes = plt.subplots(10, n_per_digit, figsize=(8, 8))
    for digit in range(10):
        # sorted() so the same images appear every run.
        files = sorted((data_dir / str(digit)).glob("*.png"))[:n_per_digit]
        for col, path in enumerate(files):
            img = np.array(Image.open(path))
            axes[digit, col].imshow(img, cmap="grey")
            axes[digit, col].axis("off")
    fig.savefig(out_path)
    plt.close(fig)  # free the figure's memory; we only need the saved file

if __name__ == "__main__":
    # 1. Class balance: slightly uneven counts are why data.py stratifies the split.
    counts = count_per_digit(DATA_DIR)
    for digit in sorted(counts):
        print(digit, counts[digit])
    print("total", sum(counts.values()))

    # 2. One image as numbers: confirms 28x28, uint8, values 0-255.
    first = next((DATA_DIR / "3").glob("*.png"))
    inspect_one(first)

    # 3. Many images as pictures.
    save_sample_grid(DATA_DIR, "results/sample_grid.png")
