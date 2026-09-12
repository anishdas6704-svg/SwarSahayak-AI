from feature_extract import extract_features

feature = extract_features(
    "dataset/sad/1.wav"
)

print("Shape =", feature.shape)
print(feature)