import os

import copy

import numpy as np

import torch

import torch.nn as nn

import torch.optim as optim


from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from sklearn.metrics import (

    confusion_matrix,

    classification_report,

    accuracy_score
)


from torch.utils.data import DataLoader, Subset


import matplotlib.pyplot as plt

import seaborn as sns


from dataset_pytorch import EmotionDataset

from cnn_model import EmotionCNN



# ==========================================

# CONFIG

# ==========================================


DATASET_PATH = "dataset_augmented"

NUM_CLASSES = 8

BATCH_SIZE = 4

EPOCHS = 100

LEARNING_RATE = 0.0003

PATIENCE = 15


DEVICE = torch.device(

    "cuda" if torch.cuda.is_available() else "cpu"
)


print("Using Device:", DEVICE)


# ==========================================

# LOAD DATASET

# ==========================================


dataset = EmotionDataset(DATASET_PATH)
all_features = []

for i in range(len(dataset)):

    feature, _ = dataset[i]

    all_features.append(
        feature.squeeze(0).numpy()
    )

all_features = np.array(all_features)

scaler = StandardScaler()

scaler.fit(all_features)

labels = []


for i in range(len(dataset)):

    _, label = dataset[i]

    labels.append(label.item())


labels = np.array(labels)

# ==================================
# CLASS WEIGHTS
# ==================================

from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels),
    y=labels
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
).to(DEVICE)

print("\nClass Weights:")
print(class_weights)


# ==========================================

# K-FOLD

# ==========================================


kfold = StratifiedKFold(

    n_splits=5,

    shuffle=True,

    random_state=42
)


fold_results = []


best_overall_acc = 0

best_overall_model = None


# ==========================================

# TRAINING LOOP

# ==========================================


for fold, (train_ids, val_ids) in enumerate(

        kfold.split(np.zeros(len(labels)), labels)):


    print("\n" + "="*50)

    print(f"FOLD {fold+1}/5")

    print("="*50)


    train_dataset = Subset(dataset, train_ids)

    val_dataset = Subset(dataset, val_ids)


    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True
    )


    val_loader = DataLoader(

        val_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False
    )


    model = EmotionCNN(

        num_classes=NUM_CLASSES

    ).to(DEVICE)


    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5
    )

    best_fold_acc = 0

    patience_counter = 0

    best_model_weights = copy.deepcopy(
        model.state_dict()
    )

    train_losses = []

    val_losses = []

    train_accs = []

    val_accs = []


    # ======================================

    # EPOCH LOOP

    # ======================================


    for epoch in range(EPOCHS):


        # TRAIN

        model.train()


        running_loss = 0

        correct = 0

        total = 0


        for inputs, targets in train_loader:

            inputs = inputs.to(DEVICE)

            targets = targets.to(DEVICE)


            optimizer.zero_grad()


            outputs = model(inputs)

            loss = criterion(

                outputs,

                targets
            )


            loss.backward()

            optimizer.step()


            running_loss += loss.item()


            _, predicted = torch.max(

                outputs.data,

                1
            )


            total += targets.size(0)


            correct += (

                predicted == targets

            ).sum().item()


        train_loss = running_loss / len(train_loader)


        train_acc = correct / total


        # VALIDATION


        model.eval()


        val_loss = 0


        correct = 0

        total = 0


        y_true = []

        y_pred = []


        with torch.no_grad():


            for inputs, targets in val_loader:


                inputs = inputs.to(DEVICE)

                targets = targets.to(DEVICE)


                outputs = model(inputs)

                loss = criterion(

                    outputs,

                    targets
                )


                val_loss += loss.item()


                _, predicted = torch.max(

                    outputs.data,

                    1
                )


                total += targets.size(0)


                correct += (

                    predicted == targets

                ).sum().item()


                y_true.extend(

                    targets.cpu().numpy()
                )


                y_pred.extend(

                    predicted.cpu().numpy()
                )


        val_loss /= len(val_loader)


        val_acc = correct / total

        train_losses.append(train_loss)

        val_losses.append(val_loss)

        train_accs.append(train_acc)

        val_accs.append(val_acc)

        scheduler.step(val_acc)

        print(

            f"Epoch {epoch+1}/{EPOCHS} | "

            f"Train Acc={train_acc:.4f} | "

            f"Val Acc={val_acc:.4f}"
        )


        # ==================================

        # EARLY STOPPING

        # ==================================


        if val_acc > best_fold_acc:


            best_fold_acc = val_acc


            best_model_weights = copy.deepcopy(

                model.state_dict()
            )


            patience_counter = 0


        else:

            patience_counter += 1


        if patience_counter >= PATIENCE:

            print(

                "Early stopping triggered."
            )


            break


    # ======================================

    # SAVE BEST FOLD

    # ======================================


    model.load_state_dict(

        best_model_weights
    )


    fold_results.append(best_fold_acc)

    print(

        f"\nFold {fold+1} Best Accuracy = "

        f"{best_fold_acc:.4f}"
    )


    if best_fold_acc > best_overall_acc:


        best_overall_acc = best_fold_acc


        best_overall_model = copy.deepcopy(

            model.state_dict()
        )


        best_true = y_true

        best_pred = y_pred


        best_train_loss = train_losses

        best_val_loss = val_losses


        best_train_acc = train_accs

        best_val_acc = val_accs


# ==========================================

# SAVE FINAL MODEL

# ==========================================


torch.save(

    best_overall_model,

    "emotion_cnn.pth"
)

print(

    "\nBest Model Saved as emotion_cnn.pth"
)


# ==========================================

# FINAL RESULTS

# ==========================================


print("\nFold Results:")


for i, acc in enumerate(fold_results):
    print(

        f"Fold {i+1}: {acc:.4f}"
    )

print(

    f"\nAverage Accuracy = "

    f"{np.mean(fold_results):.4f}"
)


# ==========================================

# ACCURACY GRAPH

# ==========================================


plt.figure(figsize=(8,5))

plt.plot(

    best_train_acc,

    label="Train Accuracy"
)

plt.plot(

    best_val_acc,

    label="Validation Accuracy"
)

plt.title(

    "Accuracy Curve"
)


plt.xlabel("Epoch")

plt.ylabel("Accuracy")


plt.legend()


plt.savefig(

    "accuracy_curve.png"
)


plt.show()


# ==========================================

# LOSS GRAPH

# ==========================================


plt.figure(figsize=(8,5))

plt.plot(

    best_train_loss,

    label="Train Loss"
)

plt.plot(

    best_val_loss,

    label="Validation Loss"
)

plt.title(

    "Loss Curve"
)


plt.xlabel("Epoch")

plt.ylabel("Loss")


plt.legend()


plt.savefig(

    "loss_curve.png"
)


plt.show()


# ==========================================

# CONFUSION MATRIX

# ==========================================


cm = confusion_matrix(

    best_true,

    best_pred
)


class_names = dataset.labels


plt.figure(figsize=(10,8))


sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    cmap="Blues",

    xticklabels=class_names,

    yticklabels=class_names
)

plt.title(

    "Confusion Matrix"
)


plt.xlabel(

    "Predicted"
)


plt.ylabel(

    "Actual"
)


plt.savefig(

    "confusion_matrix.png"
)


plt.show()


# ==========================================

# REPORT

# ==========================================


print("\nClassification Report\n")

print(

    classification_report(

        best_true,

        best_pred,

        target_names=class_names
    )
)
jls_extract_var = train_loader
for x, y in jls_extract_var:

    print(x.shape)

    break
for x, y in train_loader:
    print(x.shape)
    break

import joblib

joblib.dump(
    scaler,
    "scaler.pkl"
)

print("Scaler Saved")