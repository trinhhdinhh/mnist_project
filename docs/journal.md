# Thinking-Process Journal

Copy the block below once per step. Wrong first guesses are the most valuable part.

---

### Step __: ______  (date)

**Before coding: my understanding**
In my own words, how this method classifies a digit:
My answers to the "Think first" questions:

**Plan**
Functions I'll write and their input/output shapes:
What I expect the checkpoint result to be, and why:

**Decisions + reasoning**
- Decision: ___ | Options considered: ___ | Why I chose it: ___

**What went wrong**
- Symptom: ___ | What I thought it was: ___ | Actual cause: ___ | How I found it: ___

**Result**
Checkpoint number(s):
Did it match my prediction? If not, why?

**AI used this step?** (yes/no + row # in ai_log.md)

---

### Step 0: Explore the data  (2026-10-20)

**Before coding: my understanding**
Goal: look at the raw images before building any model (folder layout, counts, image format, background colour).

My predictions (written before running anything):
- Images per digit: I think the 10 digits are **equally common**. I know there are 60,000 images in total, so equal classes would mean about 6,000 each.
- What the images will look like: I think they'll look like **blobs**, because the image is so zoomed in (only 28×28 pixels). (Check this against the sample grid.)
- One image (shape / dtype / min / max): I think it will print **0** (no guess for dtype).
- White-on-black or black-on-white: I think the **background is black and the digits are white**.

**Plan**
Functions in `explore.py`:
- `count_per_digit(data_dir)` → dict {digit: count}
- `inspect_one(path)` → prints shape, dtype, min, max
- `check_background(path)` → background value 0 or 255?
- `save_sample_grid(data_dir, out_path)` → 10×10 grid of examples saved to `results/`

**Decisions + reasoning**
- _(todo)_

**What went wrong**
- Symptom: `IndentationError` | Cause: empty function bodies (Python needs at least `pass`) | Fix: removed the unfinished functions for now
- Symptom: `AttributeError: 'PosixPath' object has no attribute 'count'` | Cause: I tried `data_dir.count()`, but `Path` has no count method, and I was counting the top folder rather than the digit folder | Fix: loop over `item.iterdir()` and count the `.png` files
- Symptom: `min=<built-in method min of numpy.ndarray ...>` instead of a number | Cause: wrote `img.min`, not `img.min()`. Without `()` Python prints the method itself instead of calling it | Fix: add `()` (methods need them; attributes like `.shape` don't)
- Symptom: sample grid came out completely empty, with **no error** | Cause: wrote `"str(digit)"` in quotes, so Python looked for a folder literally named `str(digit)`; `glob` on a missing folder silently returns nothing | Lesson: quotes make text, not code. A silent empty result is more dangerous than a crash, so check that the list isn't empty
- Got stuck on the counting line; asked Claude to write the finished `count_per_digit` (see ai_log)

**Result**
- Counts per digit: 0: 5,923 · 1: 6,742 · 2: 5,958 · 3: 6,131 · 4: 5,842 · 5: 5,421 · 6: 5,918 · 7: 6,265 · 8: 5,851 · 9: 5,949 · total 60,000
- Did it match my prediction? I had assumed the dataset had equal amounts of each digit 0–9. It doesn't: counts range from 5,421 (digit 5) to 6,742 (digit 1).
- Why uneven classes matter for the split. My first guess: "I'm not sure, but I'm guessing it adds variety." _(Revisit after the explanation; write the real reason in my own words.)_
- One image (`3/037271.png`): shape = (28, 28), dtype = uint8, fraction of black (0) pixels = 0.73 → **black background confirmed** (73% of pixels are exactly 0). After the fix: min = 0, max = 255. **My "it will print 0" prediction was right for the min.** The image uses the full 0–255 range.
- Sample grid saved to `results/sample_grid.png` (first 10 images per digit, sorted). Every row shows the right digit, so folder labels match the images. Lots of handwriting variety: slanted vs straight 1s, open vs closed 4s, 7s with and without a crossbar, and a 9 (row 9, last column) that looks a lot like an 8.
- "Blobs" prediction: **wrong**. The grid looks clearer than I expected.

**AI used this step?** Yes: guidance and hints only, no code (ai_log rows from 2026-10-20, Step 0).

---

### Step 1: data.py (load, split, preprocess)  (2026-10-21)

**Before coding: my understanding**
Goal: one shared, reproducible setup for all five models. Load all 60,000 images with labels, make a stratified 80/20 split with a fixed seed (saving the indices), and convert to float32 in [0, 1] (flattened for 4 of the models).

Why stratify (in my own words): _(todo)_

My predictions (written before running anything):
- Memory for X as uint8 / as float32: _(todo)_
- Time to load all 60,000 PNGs: _(todo)_

**Plan**
Functions in `data.py`:
- `load_images(data_dir, max_per_digit=None)` → `X` (N, 28, 28) uint8, `y` (N,) int
- `stratified_split(y, test_frac=0.2, seed=0)` → `train_idx`, `test_idx`
- `preprocess(X, flatten)` → float32 in [0, 1], shape (N, 784) or (N, 28, 28)

**Decisions + reasoning**
- Split returns *indices*, not copies of images, so they can be saved and every model uses exactly the same test set
- Files are `sorted` before loading, so index *i* always points to the same image (otherwise a "reproducible" split would silently change)
- Commit `results/split.npz` or rely on seed=0? _(todo: my choice + why)_

**What went wrong**
- `images = [], labels = []`: Python reads it as a chained assignment, so it fails with `ValueError` (only visible once the function is actually called) | Fix: `images, labels = [], []`
- Lines after `files = ...` were outside the `for digit` loop, so `files` gets overwritten each pass and only digit 9 would load | Fix: indent everything that happens once per digit inside the loop
- `if max_per_digit == None` was backwards (it cut the list when *no* limit was given) | Fix: `is not None`
- Stray `return True` inside the loop would exit after the first digit | Fix: removed
- Empty `.append()` calls caused a TypeError | Fix: `np.array(Image.open(path))` and `digit` (given by Claude)

**Result**
- `load_images(..., max_per_digit=50)` → `(500, 28, 28) uint8`, `y` `(500,) int64`, first labels `[0 0 0 0 0]`, last `[9 9 9 9 9]` ✔ matches the expected output
- Full load (60,000) time: about 5.0 s (measured by Claude while checking the split). My prediction: _(todo)_
- `stratified_split` bugs: copied the REPL demo `np.array([3,1,3]) == 3` instead of `y == digit`; appended the count `n_test` instead of index slices; first gave train the test slice. Fixed: test = `idx[:n_test]`, train = `idx[n_test:]`
- `preprocess` first draft copied Claude's toy demo array instead of using the parameter `X`; `b / 255` and `b.reshape(...)` computed results but threw them away; no `return`, so the function returned `None` (`AttributeError: 'NoneType' object has no attribute 'shape'`)
- `preprocess` final version (given by Claude, see ai_log): `preprocess(X)` → `(500, 784) float32`, min 0.0, max 1.0; `preprocess(X, flatten=False)` → `(500, 28, 28)` ✔
- `get_data` bugs: named `load_images` without calling it (no `()`); then called all three functions with no arguments (`TypeError: missing 'data_dir'`); `DATA_DIR` was not defined in data.py; missing `return`. Lesson: parameters (names in `def`) vs arguments (values passed in); `get_data` passes `flatten`/`max_per_digit`/`seed` down to the functions that use them
- `get_data()` full run: `(48000, 784) (48000,) (12000, 784) (12000,)`; test per digit [1185 1348 1192 1226 1168 1084 1184 1253 1170 1190], **identical** to the earlier split check, so the split is reproducible end to end ✔. Indices saved to `results/split.npz`
- **Pattern I keep hitting:** (1) copying an *example* literally instead of adapting it to my own variables (`[3,1,3]`, the toy array); (2) calling something without saving its result (`iterdir()`, `b / 255`)
- Split checks, 500 images: train 400 / test 100, 10 test images per digit, overlap 0, same seed gives the same split ✔
- Split checks, all 60,000: train **48,000** / test **12,000**; test per digit [1185 1348 1192 1226 1168 1084 1184 1253 1170 1190] (each ≈ 20% of its class, e.g. digit 5: 1084 of 5,421); overlap 0; reproducible ✔

**AI used this step?** Yes: guidance and hints (ai_log rows from 2026-10-21, Step 1).

---

### Step 2: KNN  (2026-10-21)

**Before coding: my understanding**
In my own words, how KNN classifies a digit: _(todo, after the explanation)_

My predictions (written before running anything):
- What k means and how it affects accuracy: "My guess is that k equals epoch, and with k being higher I expect better results."
- Number of distances to classify one test image / all test images: "I'm not sure."
- Most-confused digit pair: **6 and 9**, "since they are similar shape but flipped."

**Plan**
- `euclidean_distances(A, B)` → (len(A), len(B)): naive loop version first, then vectorized
- `predict(dists, y_train, k)` → predicted labels via the k nearest + majority vote
- Run k = 1, 3, 5 (compute distances once, reuse for every k)

**Decisions + reasoning**
- _(todo: tie-breaking rule, batch size)_

**What went wrong**
- _(todo)_

**Result**
- _(todo)_

**AI used this step?** Yes: ai_log rows from 2026-10-21, Step 2.
