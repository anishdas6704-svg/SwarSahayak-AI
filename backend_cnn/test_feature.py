import os
from feature_extract import extract_features

base_dir = os.path.dirname(os.path.abspath(__file__))
sample_path = os.path.join(base_dir, "dataset", "sad", "1.wav")

feature = extract_features(sample_path)
print("Shape =", feature.shape)
print("Finite:", bool(feature.shape[0] > 0 and not any(map(lambda x: not (x == x), feature))))