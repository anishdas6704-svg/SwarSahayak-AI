from dataset_pytorch import EmotionDataset

dataset = EmotionDataset("dataset")

print("Samples:", len(dataset))

x, y = dataset[0]

print("Input Shape :", x.shape)
print("Label :", y)