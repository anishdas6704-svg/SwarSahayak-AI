import os
import copy
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

from dataset_pytorch import EmotionDataset
from cnn_model import EmotionCNN


# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = (
    os.path.join(BASE_DIR, "dataset_augmented")
    if os.path.exists(os.path.join(BASE_DIR, "dataset_augmented"))
    else os.path.join(BASE_DIR, "dataset")
)

NUM_CLASSES = 8
BATCH_SIZE = 32
EPOCHS = 70
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.001
LABEL_SMOOTHING = 0.05
PATIENCE = 20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 60)
print(f"Speech Emotion Recognition Training Pipeline | Device: {DEVICE}")
print(f"Dataset Path: {DATASET_PATH}")
print("=" * 60)


def train():
    # 1. Load Dataset and Extract Features
    raw_dataset = EmotionDataset(DATASET_PATH, cache=True)
    all_features, all_labels = raw_dataset.get_raw_features()
    feature_dim = all_features.shape[1]
    class_names = raw_dataset.labels

    print(f"\nExtracted Feature Matrix: {all_features.shape}")
    print(f"Labels Distribution: {np.bincount(all_labels)}")

    # 2. Compute Global Scaler & Save
    global_scaler = StandardScaler()
    global_scaler.fit(all_features)
    scaler_save_path = os.path.join(BASE_DIR, "scaler.pkl")
    joblib.dump(global_scaler, scaler_save_path)
    print(f"Global Scaler saved to: {scaler_save_path}")

    # 3. Compute Class Weights
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(all_labels),
        y=all_labels
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    print(f"Computed Class Weights: {np.round(class_weights, 3)}")

    # 4. Stratified 5-Fold Cross Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_accuracies = []
    best_overall_acc = 0.0
    best_overall_model = None
    best_true_all = []
    best_pred_all = []
    best_train_losses = []
    best_val_losses = []
    best_train_accs = []
    best_val_accs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_features, all_labels)):
        print("\n" + "-" * 50)
        print(f"FOLD {fold + 1} / 5")
        print("-" * 50)

        X_train_raw, y_train = all_features[train_idx], all_labels[train_idx]
        X_val_raw, y_val = all_features[val_idx], all_labels[val_idx]

        fold_scaler = StandardScaler()
        X_train = fold_scaler.fit_transform(X_train_raw)
        X_val = fold_scaler.transform(X_val_raw)

        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        )
        val_ds = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long)
        )

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

        model = EmotionCNN(in_features=feature_dim, num_classes=NUM_CLASSES).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=LABEL_SMOOTHING)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

        best_fold_acc = 0.0
        best_fold_weights = None
        best_fold_preds = None
        patience_counter = 0

        fold_train_losses = []
        fold_val_losses = []
        fold_train_accs = []
        fold_val_accs = []

        for epoch in range(EPOCHS):
            # Training Phase
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for bx, by in train_loader:
                bx = bx.to(DEVICE)
                by = by.to(DEVICE)

                optimizer.zero_grad()
                outputs = model(bx)
                loss = criterion(outputs, by)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()

                running_loss += loss.item() * bx.size(0)
                _, predicted = torch.max(outputs, 1)
                total += by.size(0)
                correct += (predicted == by).sum().item()

            scheduler.step()
            train_loss = running_loss / total
            train_acc = correct / total

            # Validation Phase
            model.eval()
            val_loss_sum = 0.0
            val_correct = 0
            val_total = 0
            epoch_preds = []

            with torch.no_grad():
                for bx, by in val_loader:
                    bx = bx.to(DEVICE)
                    by = by.to(DEVICE)

                    outputs = model(bx)
                    loss = criterion(outputs, by)

                    val_loss_sum += loss.item() * bx.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_total += by.size(0)
                    val_correct += (predicted == by).sum().item()
                    epoch_preds.extend(predicted.cpu().numpy())

            val_loss = val_loss_sum / val_total
            val_acc = val_correct / val_total

            fold_train_losses.append(train_loss)
            fold_val_losses.append(val_loss)
            fold_train_accs.append(train_acc)
            fold_val_accs.append(val_acc)

            if (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
                print(
                    f"Epoch [{epoch+1:02d}/{EPOCHS}] | "
                    f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
                    f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%"
                )

            if val_acc > best_fold_acc:
                best_fold_acc = val_acc
                best_fold_weights = copy.deepcopy(model.state_dict())
                best_fold_preds = epoch_preds
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}.")
                break

        print(f"--> Fold {fold + 1} Best Validation Accuracy: {best_fold_acc * 100:.2f}%")
        fold_accuracies.append(best_fold_acc)
        best_true_all.extend(y_val)
        best_pred_all.extend(best_fold_preds)

        if best_fold_acc > best_overall_acc:
            best_overall_acc = best_fold_acc
            best_overall_model = best_fold_weights
            best_train_losses = fold_train_losses
            best_val_losses = fold_val_losses
            best_train_accs = fold_train_accs
            best_val_accs = fold_val_accs

    # 5. Save Final Model
    model_save_path = os.path.join(BASE_DIR, "emotion_cnn.pth")
    if best_overall_model is not None:
        torch.save(best_overall_model, model_save_path)
        print(f"\nFinal Best Model weights saved to: {model_save_path}")

    # 6. Overall Performance Summary
    mean_acc = np.mean(fold_accuracies)
    print("\n" + "=" * 60)
    print("5-FOLD CROSS VALIDATION RESULTS")
    print("=" * 60)
    for i, acc in enumerate(fold_accuracies):
        print(f"  Fold {i+1}: {acc * 100:.2f}%")
    print("-" * 60)
    print(f"Mean Validation Accuracy: {mean_acc * 100:.2f}% (Std: {np.std(fold_accuracies) * 100:.2f}%)")
    print("=" * 60)

    # 7. Classification Report
    print("\nCombined Out-of-Fold Classification Report:")
    print(classification_report(best_true_all, best_pred_all, target_names=class_names, digits=4))

    # 8. Save Artifacts: Confusion Matrix, Accuracy Curve, Loss Curve
    cm = confusion_matrix(best_true_all, best_pred_all)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(f"Confusion Matrix (Mean 5-Fold Acc: {mean_acc * 100:.2f}%)", fontsize=14)
    plt.xlabel("Predicted Emotion", fontsize=12)
    plt.ylabel("Actual Emotion", fontsize=12)
    plt.tight_layout()
    cm_path = os.path.join(BASE_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Saved: {cm_path}")

    if best_train_accs and best_val_accs:
        plt.figure(figsize=(8, 5))
        plt.plot(best_train_accs, label="Train Accuracy", linewidth=2)
        plt.plot(best_val_accs, label="Validation Accuracy", linewidth=2)
        plt.title("Training & Validation Accuracy Curve", fontsize=14)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Accuracy", fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        acc_path = os.path.join(BASE_DIR, "accuracy_curve.png")
        plt.savefig(acc_path, dpi=150)
        plt.close()
        print(f"Saved: {acc_path}")

    if best_train_losses and best_val_losses:
        plt.figure(figsize=(8, 5))
        plt.plot(best_train_losses, label="Train Loss", linewidth=2)
        plt.plot(best_val_losses, label="Validation Loss", linewidth=2)
        plt.title("Training & Validation Loss Curve", fontsize=14)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        loss_path = os.path.join(BASE_DIR, "loss_curve.png")
        plt.savefig(loss_path, dpi=150)
        plt.close()
        print(f"Saved: {loss_path}")


if __name__ == "__main__":
    train()