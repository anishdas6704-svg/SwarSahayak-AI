import os
import torch
from torch.utils.data import Dataset
from feature_extract import extract_features
import numpy as np


class EmotionDataset(Dataset):

    def __init__(self, dataset_path="dataset"):

        self.dataset_path = dataset_path

        self.labels = sorted([
            d for d in os.listdir(dataset_path)
            if os.path.isdir(
                os.path.join(dataset_path, d)
            )
        ])

        self.label_map = {
            label: idx
            for idx, label in enumerate(self.labels)
        }

        self.samples = []

        print("\nLoading Dataset...\n")

        for label in self.labels:

            folder = os.path.join(
                dataset_path,
                label
            )

            for file in os.listdir(folder):

                if file.endswith(".wav"):

                    self.samples.append(
                        (
                            os.path.join(folder, file),
                            self.label_map[label]
                        )
                    )

        print("Classes :", self.labels)
        print("Total Samples :", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        filepath, label = self.samples[idx]

        features = extract_features(filepath)

        # Standardization
        features = (
            features - np.mean(features)
        ) / (
            np.std(features) + 1e-8
        )

        features = torch.tensor(
            features,
            dtype=torch.float32
        )
        features = features.unsqueeze(0)

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return features, label


if __name__ == "__main__":

    dataset = EmotionDataset("dataset")

    x, y = dataset[0]

    print("\nFeature Shape =", x.shape)

    print("Label =", y)