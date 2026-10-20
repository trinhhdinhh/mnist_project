"""data.py: load the raw MNIST images, make the train/test split, preprocess.

Used by: every classifier (all five use the SAME split).

Requirements:
- Load directly from the raw PNG files (one folder per digit, label = folder name).
- Skip non-image files (e.g. .DS_Store).
- Create my own partition: random 80% train / 20% test, fixed seed, stratified by digit.
- Normalize to [0, 1] (convert uint8 -> float32 FIRST).
- Flatten to 784-d vectors when needed; keep 28x28 for the CNN.

TODO:
- [ ] load all images + labels
- [ ] reproducible stratified split (save the indices)
- [ ] normalize
- [ ] flatten
- [ ] (optional) cache the arrays so loading 60k PNGs only happens once
"""
