import os
import torch
import numpy as np
from torch.utils.data import Dataset
from feature_extract import extract_features


class EmotionDataset(Dataset):
    """
    PyTorch Dataset for Speech Emotion Recognition.
    Supports in-memory caching and per-feature dataset scaling.
    """
    def __init__(self, dataset_path="dataset", cache=True, scaler=None):
        if not os.path.exists(dataset_path):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            alt_path = os.path.join(base_dir, dataset_path)
            if os.path.exists(alt_path):
                dataset_path = alt_path

        self.dataset_path = dataset_path
        self.cache = cache
        self.scaler = scaler

        self.labels = sorted([
            d for d in os.listdir(dataset_path)
            if os.path.isdir(os.path.join(dataset_path, d))
        ])

        self.label_map = {
            label: idx
            for idx, label in enumerate(self.labels)
        }

        self.samples = []
        for label in self.labels:
            folder = os.path.join(dataset_path, label)
            for file in sorted(os.listdir(folder)):
                if file.endswith(".wav"):
                    self.samples.append(
                        (
                            os.path.join(folder, file),
                            self.label_map[label]
                        )
                    )

        print(f"Loaded dataset from: '{dataset_path}'", flush=True)
        print(f"Classes ({len(self.labels)}): {self.labels}", flush=True)
        print(f"Total Samples: {len(self.samples)}", flush=True)

        self.cached_features = None
        self.cached_labels = None

        if self.cache:
            self._preload_features()

    def _preload_features(self):
        feat_list = []
        label_list = []
        total = len(self.samples)
        print(f"Preloading and extracting features for {total} audio samples...", flush=True)
        for idx, (filepath, label) in enumerate(self.samples):
            feat = extract_features(filepath)
            feat_list.append(feat)
            label_list.append(label)
            if (idx + 1) % 500 == 0 or idx == total - 1:
                print(f"  Extracted [{idx + 1}/{total}] audio samples...", flush=True)

        self.cached_features = np.nan_to_num(
            np.array(feat_list, dtype=np.float32),
            nan=0.0, posinf=0.0, neginf=0.0
        )
        self.cached_labels = np.array(label_list, dtype=np.int64)

        if self.scaler is not None:
            self.cached_features = self.scaler.transform(self.cached_features).astype(np.float32)

    def apply_scaler(self, scaler):
        self.scaler = scaler
        if self.cached_features is not None:
            self.cached_features = scaler.transform(self.cached_features).astype(np.float32)

    def get_raw_features(self):
        if self.cached_features is not None:
            return self.cached_features, self.cached_labels
        feat_list = []
        label_list = []
        for filepath, label in self.samples:
            feat = extract_features(filepath)
            feat_list.append(feat)
            label_list.append(label)
        return (
            np.nan_to_num(np.array(feat_list, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0),
            np.array(label_list, dtype=np.int64)
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.cached_features is not None:
            feat = self.cached_features[idx]
            label = self.cached_labels[idx]
        else:
            filepath, label = self.samples[idx]
            feat = extract_features(filepath)
            if self.scaler is not None:
                feat = self.scaler.transform(feat.reshape(1, -1)).flatten().astype(np.float32)

        feature_tensor = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return feature_tensor, label_tensor


if __name__ == "__main__":
    dataset = EmotionDataset("dataset", cache=False)
    x, y = dataset[0]
    print("\nFeature Shape =", x.shape)
    print("Label =", y)