"""cnn.py: Convolutional Neural Network. PyTorch REQUIRED.

Requirements:
- At least two convolutional layers, each with activation + pooling.
- Example: Conv(1->32, 3x3) -> ReLU -> MaxPool -> Conv(32->64, 3x3) -> ReLU
           -> MaxPool -> FC -> Softmax
- Train with cross-entropy loss.
- Input stays 2D: (batch, 1, 28, 28).

TODO:
- [ ] fill in the shape table on paper first (see the guide doc)
- [ ] model class (nn.Module)
- [ ] loss + optimizer
- [ ] train using torch_utils
"""
