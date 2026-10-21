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
