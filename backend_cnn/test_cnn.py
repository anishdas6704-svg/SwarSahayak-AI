import torch
from cnn_model import EmotionCNN

model = EmotionCNN(
    num_classes=8
)

dummy = torch.randn(
    4,
    1,
    64
)

output = model(dummy)

print("Output Shape:")
print(output.shape)