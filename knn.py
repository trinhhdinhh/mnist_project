"""knn.py: K-Nearest Neighbor classifier. NumPy ONLY.

Requirements:
- Input: flattened 784-d vectors from data.py.
- Euclidean distance (required).
- Classify by majority vote among the k nearest training images.
- Test at least k = 1, 3, 5.
- No scikit-learn (no KNeighborsClassifier).

Decisions to record in docs/journal.md:
- how to break vote ties
- batch size for distance computation (memory)

TODO:
- [ ] distance computation (naive first, then vectorized)
- [ ] find k nearest neighbors
- [ ] majority vote
- [ ] predict for k = 1, 3, 5
"""
