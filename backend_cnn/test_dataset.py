import os
from dataset_pytorch import EmotionDataset

base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(base_dir, "dataset")
dataset = EmotionDataset(dataset_dir, cache=False)
print("Samples:", len(dataset))
x, y = dataset[0]
print("Input Shape :", x.shape)
print("Label :", y)