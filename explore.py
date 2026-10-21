"""explore.py: Step 0, understand the data before writing any model.

Questions to answer (record predictions in docs/journal.md BEFORE running):
- [ ] folder layout; is there a provided split?
- [ ] number of images per digit
- [ ] one image: shape, dtype, min, max
- [ ] white-on-black or black-on-white?
- [ ] grid of sample images per digit -> save to results/
"""
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

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
    img = np.array(Image.open(path))
    print(f"shape={img.shape}  dtype={img.dtype}  min={img.min()}  max={img.max()}")
    print(f"fraction of black (0) pixels: {(img == 0).mean():.2f}")

def save_sample_grid(data_dir, out_path, n_per_digit=10):
    fig, axes = plt.subplots(10, n_per_digit, figsize=(8, 8))
    for digit in range(10):
        files = sorted((data_dir / str(digit)).glob("*.png"))[:n_per_digit]
        for col, path in enumerate(files):
            img = np.array(Image.open(path))
            axes[digit, col].imshow(img, cmap="grey")
            axes[digit, col].axis("off")
    fig.savefig(out_path)
    plt.close(fig)

if __name__ == "__main__":
    counts = count_per_digit(DATA_DIR)
    for digit in sorted(counts):
        print(digit, counts[digit])
    print("total", sum(counts.values()))

    first = next((DATA_DIR / "3").glob("*.png"))
    inspect_one(first)

    save_sample_grid(DATA_DIR, "results/sample_grid.png")

