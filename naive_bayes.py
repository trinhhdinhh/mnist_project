"""naive_bayes.py: Naive Bayes classifier on binarized pixels. NumPy ONLY.

Requirements:
- Binarize each pixel: 1 if pixel > 0.5 (after normalizing to [0, 1]), else 0.
- Estimate P(pixel j is on | digit c) for every pixel and class.
- Classify with Bayes' rule under the pixel-independence assumption.

Decisions to record in docs/journal.md:
- smoothing (alpha) and why it is needed
- working in log space and why

TODO:
- [ ] binarize
- [ ] class priors P(c)
- [ ] conditional probabilities P(x_j = 1 | c)
- [ ] predict (log posterior, argmax)
- [ ] visualize the probability maps as 10 digit-like images
"""
