import os
import librosa
import soundfile as sf
import numpy as np

INPUT_DATASET = "dataset"
OUTPUT_DATASET = "dataset_augmented"

os.makedirs(OUTPUT_DATASET, exist_ok=True)


def add_noise(y):
    noise = np.random.randn(len(y))
    return y + 0.005 * noise


def pitch_up(y, sr):
    return librosa.effects.pitch_shift(
        y=y,
        sr=sr,
        n_steps=2
    )


def pitch_down(y, sr):
    return librosa.effects.pitch_shift(
        y=y,
        sr=sr,
        n_steps=-2
    )


def stretch(y):
    return librosa.effects.time_stretch(
        y=y,
        rate=1.1
    )


for emotion in os.listdir(INPUT_DATASET):

    emotion_path = os.path.join(
        INPUT_DATASET,
        emotion
    )

    if not os.path.isdir(emotion_path):
        continue

    save_emotion_path = os.path.join(
        OUTPUT_DATASET,
        emotion
    )

    os.makedirs(
        save_emotion_path,
        exist_ok=True
    )

    for file in os.listdir(emotion_path):

        if not file.endswith(".wav"):
            continue

        filepath = os.path.join(
            emotion_path,
            file
        )

        y, sr = librosa.load(
            filepath,
            sr=22050
        )

        filename = os.path.splitext(file)[0]

        # Original
        sf.write(
            os.path.join(
                save_emotion_path,
                f"{filename}_orig.wav"
            ),
            y,
            sr
        )

        # Noise
        sf.write(
            os.path.join(
                save_emotion_path,
                f"{filename}_noise.wav"
            ),
            add_noise(y),
            sr
        )

        # Pitch Up
        sf.write(
            os.path.join(
                save_emotion_path,
                f"{filename}_pitchup.wav"
            ),
            pitch_up(y, sr),
            sr
        )

        # Pitch Down
        sf.write(
            os.path.join(
                save_emotion_path,
                f"{filename}_pitchdown.wav"
            ),
            pitch_down(y, sr),
            sr
        )

        # Stretch
        stretched = stretch(y)

        sf.write(
            os.path.join(
                save_emotion_path,
                f"{filename}_stretch.wav"
            ),
            stretched,
            sr
        )

print("Dataset Augmentation Completed.")