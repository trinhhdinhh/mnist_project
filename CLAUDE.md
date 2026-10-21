# CLAUDE.md: MNIST Classifiers Project

## Project summary
A course project that compares classical ML and deep learning for image classification.
Five classifiers (KNN, Naive Bayes, Linear, MLP, CNN) are trained and tested on the **same
split** of the provided MNIST images, and the results are compared.

**Two deliverables:** (1) modular, reproducible, well-commented code, and (2) a 3–5 page
PDF report (see the Report section). AI tool use is **encouraged** by the course, but it must be
disclosed in an AI Use Statement, and the report must show my thinking process.

---

## How Claude should work in this repo (READ FIRST)
I (Trinh) am writing this code myself to learn it, and I document my AI use and thinking.

- **Teach, don't write.** Explain concepts, give hints, and point to the line or idea to look at.
  Do NOT write complete functions or files for classifier code unless I explicitly ask
  ("write this for me").
- **Review on request.** When I paste code, list the issues and explain why. Let me fix them.
- **Check math, don't hand it over.** For derivations (e.g., the linear-classifier gradient),
  tell me whether mine is right and give a hint if it isn't.
- **Respect library limits.** Never suggest scikit-learn classifiers, or PyTorch for KNN or
  Naive Bayes (see the table below).
- **Remind me to log.** After meaningful help, remind me to add a row to `docs/ai_log.md`.
- **Small first.** Suggest testing on ~500–1,000 images before running the full dataset.
- Boilerplate (plotting, argparse, file I/O) is fine to write directly if I ask.

---

## Dataset (provided on Moodle)
- Location on my machine: `~/Downloads/MNIST/` (keep a copy or symlink at `data/MNIST/` in the repo).
- Layout: one folder per digit, `0/` … `9/`. Label = folder name.
- **60,000 PNG images**: 28×28, 8-bit grayscale (mode `L`), uint8 pixels in 0–255,
  **white digit on black background** (background = 0).
- Per-digit counts: 0: 5,923 · 1: 6,742 · 2: 5,958 · 3: 6,131 · 4: 5,842 ·
  5: 5,421 · 6: 5,918 · 7: 6,265 · 8: 5,851 · 9: 5,949. Slightly imbalanced, so stratify the split.
- **No train/test split is provided.** I must create my own.
- Ignore non-image files (e.g., macOS `.DS_Store`) when loading.

### Partition
- Random **80% train / 20% test** (≈48,000 / 12,000), fixed seed, stratified by digit.
- Required for KNN and Naive Bayes; **use the same split for ALL five models** for a fair
  comparison. Save the split indices for reproducibility.
- The MLP and CNN carve a validation set out of *train*. The test set is never used for tuning.

### Preprocessing
- Normalize pixels: **[0, 1]** for NumPy models; **[-1, 1]** for PyTorch
  (`transforms.Normalize((0.5,), (0.5,))` or the equivalent `(x - 0.5) / 0.5`).
- Convert uint8 → float32 **before** any arithmetic (uint8 subtraction wraps around).
- **Flatten** to 784-dim vectors for KNN, Naive Bayes, Linear and MLP.
- **Keep 2D** (1 × 28 × 28) for the CNN.

---

## Classifiers: all five required

| # | Classifier | Library | Hard requirements |
|---|---|---|---|
| 1 | **K-Nearest Neighbor** | **NumPy only** (no `KNeighborsClassifier`, no scikit-learn) | Load from raw image files; own train/test partition; flattened 784-d vectors; **Euclidean distance (required)**; majority vote among k neighbors; test **at least k = 1, 3, 5** |
| 2 | **Naive Bayes** | **NumPy only** | Binarize each pixel (**1 if > 0.5, else 0**, after normalizing to [0,1]); estimate P(pixel on \| digit) per class; classify with Bayes' rule + the independence assumption |
| 3 | **Linear classifier** | NumPy **or** PyTorch | Model **y = Wx**; **L2 loss**; trained with **gradient descent**. NumPy: hand-written forward, backward, update. PyTorch: `nn.Linear` + optimizer + autograd |
| 4 | **Multilayer Perceptron (MLP)** | **PyTorch required** | At least one hidden layer with a nonlinearity (e.g., ReLU); example 784 → 256 → 128 → 10; **trained with SGD** |
| 5 | **Convolutional Neural Network (CNN)** | **PyTorch required** | **At least two conv layers**, each with activation + pooling; example Conv(1→32, 3×3) → ReLU → MaxPool → Conv(32→64, 3×3) → ReLU → MaxPool → FC → Softmax; **cross-entropy loss** |

Implementation notes:
- KNN: vectorize distances (‖a−b‖² = ‖a‖² + ‖b‖² − 2a·b), batch the test set, compute
  distances once and reuse them for all k.
- Naive Bayes: work in log space; use Laplace smoothing to avoid log(0).
- Linear: chosen implementation = ______ (decide + record why). Do a numerical gradient check.
- CNN: `nn.CrossEntropyLoss` already applies softmax, so the model outputs logits.

## Deliverable 1: Code
- **Modular:** separate functions/files for each method.
- **Reproducible:** fixed seeds (NumPy, PyTorch, DataLoader); one command reruns everything.
- **Well-commented:** explain the *why*, not just the *what*.

## Evaluation (all five on the same test set)
Required baseline:
- Test accuracy, train time, predict time, and key hyperparameters for every model.

"Interesting things to consider" from the assignment (each one strengthens the report):
- [ ] **Failure-mode analysis:** a 10×10 confusion matrix per model, the most-confused digit
      pairs, and example misclassified images.
- [ ] **Hyperparameter sweeps:** e.g., KNN k (1, 3, 5, 7, …), NB smoothing α, linear/MLP/CNN
      learning rate, epochs, hidden sizes, momentum.
- [ ] **Different loss functions:** e.g., the linear classifier with L2 vs. cross-entropy (softmax),
      or the MLP with MSE vs. cross-entropy.
- [ ] **Visualizations:** the linear classifier's W rows reshaped to 28×28 (10 "template" images);
      Naive Bayes θ (P(pixel on | digit)) as 10 digit-like probability maps; training curves.

## Deliverable 2: Report (PDF, 3–5 pages)
Required structure, in this order:

| Part | What goes in it |
|---|---|
| **Title** | Project title |
| **Abstract** | Short summary: the task, the five methods, headline results |
| **1. Project Description** | The problem addressed (handwritten digit classification); motivations and potential applications (e.g., check/postal-code reading, form digitization) |
| **2. AI Techniques** | The main idea behind each of the 5 classifiers: instance-based (KNN), probabilistic/generative (NB), linear model + gradient descent, neural nets (MLP), convolution + pooling (CNN) |
| **3. Datasets** | MNIST: 60,000 28×28 grayscale PNGs in 10 digit folders, class counts, my stratified 80/20 split, preprocessing |
| **4. Implementation Tools** | Platform (macOS, Python version), NumPy, PyTorch, Pillow, Matplotlib, editor, AI tools |
| **5. References and AI Assistance** | Citations for every source used; **AI Use Statement**: what I asked, what I kept/changed/rejected, how I verified it (built from `docs/ai_log.md`) |

Also required somewhere in the report: **show my thinking process** (decisions, wrong
guesses, debugging; drawn from `docs/journal.md`).

Results/comparison: the required list has no explicit Results section. Put the comparison
table, confusion matrices and visualizations under Section 2 or as an added
"Results & Analysis" section before Section 5. Confirm with the professor which one they prefer.

Page budget (≈4 pages): Abstract + §1 ≈ 0.5 · §2 ≈ 1 · §3 + §4 ≈ 0.5 · Results/figures ≈ 1.5 · §5 ≈ 0.5.

## Open questions, to confirm with the professor
- [ ] The CNN "Requirements:" bullet is empty in the assignment. Ask whether anything is expected
      beyond 2+ conv layers with activation + pooling and cross-entropy loss.
- [ ] Where results should go (inside §2, or a separate Results section).
- [ ] Due date and submission format for the code (zip? GitHub link?).

---

## Suggested repo layout
```
mnist_project/
├── CLAUDE.md
├── README.md              # setup + how to run
├── requirements.txt       # numpy, pillow, matplotlib, torch
├── data.py                # load images, split, normalize, flatten
├── knn.py
├── naive_bayes.py
├── linear.py
├── torch_utils.py         # shared loaders + training loop (MLP/CNN)
├── mlp.py
├── cnn.py
├── main.py                # trains all 5 on the same split, writes results
├── data/MNIST/            # the Moodle images (not committed)
├── results/               # metrics JSON, plots, confusion matrices
├── docs/
│   ├── ai_log.md          # every AI interaction: asked / got / did / verified
│   ├── journal.md         # per-step thinking-process notes
│   └── references.md      # every source used, collected as I go
└── report/
    └── report.pdf         # 3–5 pages: Title, Abstract, §1–5
```

## Milestones
- [x] 0. Explore the data (reproduce the facts above in my own `explore.py`)
- [x] 1. `data.py`: load + reproducible 80/20 split + preprocessing
- [ ] 2. KNN (k = 1, 3, 5)
- [ ] 3. Naive Bayes + θ visualization
- [ ] 4. Linear classifier: derive the gradient, gradient check, train
- [ ] 5. MLP with SGD
- [ ] 6. CNN: shape table on paper, then train
- [ ] 7. `main.py` comparison table + confusion matrices
- [ ] 8. Extras: hyperparameter sweeps, alternative loss, W and θ visualizations
- [ ] 9. Report draft (Title, Abstract, §1–5, results, thinking process, AI Use Statement)
- [ ] 10. README + final cleanup; export the report to PDF; check it's 3–5 pages

## Expected results (sanity check)
On the 20% test split: Naive Bayes ≈ 84%, Linear (L2) ≈ 85%, KNN ≈ 97%, MLP ≈ 98%, CNN ≈ 99%.
Something near 10% means labels and images are misaligned.
