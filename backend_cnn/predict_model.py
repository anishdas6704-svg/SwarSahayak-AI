import os
import torch
import joblib
import numpy as np
from cnn_model import EmotionCNN
from feature_extract import extract_features

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CLASSES = [
    'acknowledge',
    'afraid',
    'angry',
    'confused',
    'discomfort',
    'glad',
    'monotony',
    'sad'
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_cached_model = None
_cached_scaler = None


def load_model(model_path=None):
    global _cached_model
    if model_path is None:
        model_path = os.path.join(BASE_DIR, "emotion_cnn.pth")

    if _cached_model is not None and model_path == os.path.join(BASE_DIR, "emotion_cnn.pth"):
        return _cached_model

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}. Train model first using train_cnn.py.")

    model = EmotionCNN(in_features=526, num_classes=len(CLASSES)).to(DEVICE)
    state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    _cached_model = model
    return model


def load_scaler(scaler_path=None):
    global _cached_scaler
    if scaler_path is None:
        scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

    if _cached_scaler is not None and scaler_path == os.path.join(BASE_DIR, "scaler.pkl"):
        return _cached_scaler

    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        _cached_scaler = scaler
        return scaler
    return None


def predict_emotion(file_path_or_audio, model=None, scaler=None):
    """
    Predict emotion from audio file or audio array.
    Returns dictionary with predicted emotion, confidence, and class probabilities.
    """
    if model is None:
        model = load_model()
    if scaler is None:
        scaler = load_scaler()

    features = extract_features(file_path_or_audio)

    if scaler is not None:
        features = scaler.transform(features.reshape(1, -1)).flatten()

    features_tensor = torch.tensor(
        features, dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        outputs = model(features_tensor)
        probabilities = torch.softmax(outputs, dim=1).squeeze(0).cpu().numpy()
        predicted_idx = int(np.argmax(probabilities))
        predicted_label = CLASSES[predicted_idx]
        confidence = float(probabilities[predicted_idx])

    return {
        "emotion": predicted_label,
        "confidence": confidence,
        "probabilities": {CLASSES[i]: float(probabilities[i]) for i in range(len(CLASSES))}
    }


if __name__ == "__main__":
    sample_file = os.path.join(BASE_DIR, "dataset", "sad", "1.wav")
    if os.path.exists(sample_file):
        try:
            result = predict_emotion(sample_file)
            print("\nPrediction Result:")
            print(f"Predicted Emotion : {result['emotion']}")
            print(f"Confidence        : {result['confidence']:.2%}")
            print("\nClass Probabilities:")
            for emotion, prob in result["probabilities"].items():
                print(f"  {emotion:12s}: {prob:.4f}")
        except Exception as e:
            print("Error running prediction:", e)
    else:
        print("Sample audio file not found.")
