import torch
from cnn_model import EmotionCNN

model = EmotionCNN(in_features=526, num_classes=8)
dummy = torch.randn(4, 1, 526)
out = model(dummy)
print("CNN Output Shape:", out.shape)
assert out.shape == (4, 8), f"Expected (4, 8), got {out.shape}"
print("CNN Test Passed!")