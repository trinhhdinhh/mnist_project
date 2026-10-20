"""linear.py: linear classifier y = Wx. NumPy or PyTorch (choose one).

Requirements:
- Model: y = Wx (decide how to handle the bias).
- Train with L2 loss and gradient descent.
- NumPy option: write forward, backward and update myself.
- PyTorch option: nn.Linear + optimizer + autograd.

Chosen implementation: ______  (reason: ______)

TODO:
- [ ] derive dL/dW on paper (goes in the report)
- [ ] forward + loss
- [ ] gradient (+ numerical gradient check if NumPy)
- [ ] training loop, record loss per epoch
- [ ] visualize the rows of W as 28x28 images
- [ ] (extra) compare L2 vs. a different loss, e.g. cross-entropy
"""
