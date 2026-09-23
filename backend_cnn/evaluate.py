import os
import torch
import joblib
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

from dataset_pytorch import EmotionDataset
from cnn_model import EmotionCNN

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_model(model_path=None, dataset_path=None, scaler_path=None, batch_size=32):
    if model_path is None:
        model_path = os.path.join(BASE_DIR, "emotion_cnn.pth")
    if scaler_path is None:
        scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
    if dataset_path is None:
        dataset_path = (
            os.path.join(BASE_DIR, "dataset_augmented")
            if os.path.exists(os.path.join(BASE_DIR, "dataset_augmented"))
            else os.path.join(BASE_DIR, "dataset")
        )

    if not os.path.exists(model_path):
        print(f"Model file not found at: {model_path}. Train model first using train_cnn.py.")
        return 0.0

    scaler = None
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print(f"Loaded Scaler from: {scaler_path}")

    dataset = EmotionDataset(dataset_path, cache=True, scaler=scaler)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = EmotionCNN(in_features=526, num_classes=len(dataset.labels)).to(DEVICE)
    state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    y_true = []
    y_pred = []

    print(f"\nEvaluating model on {len(dataset)} samples from '{dataset_path}'...")

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    print("\n" + "=" * 50)
    print(f"Overall Evaluation Accuracy: {acc * 100:.2f}%")
    print("=" * 50)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=dataset.labels, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=dataset.labels,
        yticklabels=dataset.labels
    )
    plt.title(f"Evaluation Confusion Matrix (Accuracy: {acc * 100:.2f}%)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    cm_path = os.path.join(BASE_DIR, "eval_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Evaluation confusion matrix saved to: {cm_path}")

    return acc


if __name__ == "__main__":
    evaluate_model()
