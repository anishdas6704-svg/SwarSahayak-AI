import os
import librosa
import soundfile as sf
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DATASET = os.path.join(BASE_DIR, "dataset")
OUTPUT_DATASET = os.path.join(BASE_DIR, "dataset_augmented")


def add_noise(y, factor=0.005):
    noise = np.random.randn(len(y))
    return y + factor * noise


def pitch_shift(y, sr, n_steps):
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)


def time_stretch(y, rate):
    return librosa.effects.time_stretch(y=y, rate=rate)


def change_volume(y, gain):
    return np.clip(y * gain, -1.0, 1.0)


def augment_all():
    print("Starting audio augmentation pipeline...")
    os.makedirs(OUTPUT_DATASET, exist_ok=True)

    emotions = sorted([
        d for d in os.listdir(INPUT_DATASET)
        if os.path.isdir(os.path.join(INPUT_DATASET, d))
    ])

    total_created = 0

    for emotion in emotions:
        emotion_path = os.path.join(INPUT_DATASET, emotion)
        save_emotion_path = os.path.join(OUTPUT_DATASET, emotion)
        os.makedirs(save_emotion_path, exist_ok=True)

        files = [f for f in os.listdir(emotion_path) if f.endswith(".wav")]
        is_minority = (len(files) <= 10)  # e.g. glad

        print(f"Processing '{emotion}' ({len(files)} base files, minority={is_minority})...")

        for file in files:
            filepath = os.path.join(emotion_path, file)
            y, sr = librosa.load(filepath, sr=22050)
            filename = os.path.splitext(file)[0]

            variants = [
                (f"{filename}_orig.wav", y),
                (f"{filename}_pitch_up1.wav", pitch_shift(y, sr, 1.0)),
                (f"{filename}_pitch_up2.wav", pitch_shift(y, sr, 2.0)),
                (f"{filename}_pitch_dn1.wav", pitch_shift(y, sr, -1.0)),
                (f"{filename}_pitch_dn2.wav", pitch_shift(y, sr, -2.0)),
                (f"{filename}_stretch_slow.wav", time_stretch(y, 0.9)),
                (f"{filename}_stretch_fast.wav", time_stretch(y, 1.1)),
                (f"{filename}_noise_light.wav", add_noise(y, 0.003)),
                (f"{filename}_noise_med.wav", add_noise(y, 0.007)),
                (f"{filename}_vol_up.wav", change_volume(y, 1.25)),
                (f"{filename}_vol_dn.wav", change_volume(y, 0.8)),
                (f"{filename}_combo_p1_n.wav", add_noise(pitch_shift(y, sr, 1.5), 0.003)),
            ]

            # If minority class, generate additional variations to equalize class distribution
            if is_minority:
                variants.extend([
                    (f"{filename}_pitch_up3.wav", pitch_shift(y, sr, 3.0)),
                    (f"{filename}_pitch_dn3.wav", pitch_shift(y, sr, -3.0)),
                    (f"{filename}_pitch_up15.wav", pitch_shift(y, sr, 1.5)),
                    (f"{filename}_pitch_dn15.wav", pitch_shift(y, sr, -1.5)),
                    (f"{filename}_pitch_up05.wav", pitch_shift(y, sr, 0.5)),
                    (f"{filename}_pitch_dn05.wav", pitch_shift(y, sr, -0.5)),
                    (f"{filename}_stretch_085.wav", time_stretch(y, 0.85)),
                    (f"{filename}_stretch_095.wav", time_stretch(y, 0.95)),
                    (f"{filename}_stretch_105.wav", time_stretch(y, 1.05)),
                    (f"{filename}_stretch_115.wav", time_stretch(y, 1.15)),
                    (f"{filename}_stretch_125.wav", time_stretch(y, 1.25)),
                    (f"{filename}_noise_subtle.wav", add_noise(y, 0.002)),
                    (f"{filename}_noise_high.wav", add_noise(y, 0.010)),
                    (f"{filename}_vol_070.wav", change_volume(y, 0.70)),
                    (f"{filename}_vol_090.wav", change_volume(y, 0.90)),
                    (f"{filename}_vol_115.wav", change_volume(y, 1.15)),
                    (f"{filename}_vol_135.wav", change_volume(y, 1.35)),
                    (f"{filename}_combo_p2_s.wav", time_stretch(pitch_shift(y, sr, 2.0), 0.95)),
                    (f"{filename}_combo_pd2_s.wav", time_stretch(pitch_shift(y, sr, -2.0), 1.05)),
                    (f"{filename}_combo_p1_s11.wav", time_stretch(pitch_shift(y, sr, 1.0), 1.1)),
                    (f"{filename}_combo_pd1_s09.wav", time_stretch(pitch_shift(y, sr, -1.0), 0.9)),
                    (f"{filename}_combo_p05_n.wav", add_noise(pitch_shift(y, sr, 0.5), 0.004)),
                    (f"{filename}_combo_pd05_n.wav", add_noise(pitch_shift(y, sr, -0.5), 0.004)),
                    (f"{filename}_combo_s11_n.wav", add_noise(time_stretch(y, 1.1), 0.004)),
                    (f"{filename}_combo_s09_n.wav", add_noise(time_stretch(y, 0.9), 0.004)),
                    (f"{filename}_combo_p2_v12.wav", change_volume(pitch_shift(y, sr, 2.0), 1.2)),
                    (f"{filename}_combo_pd2_v08.wav", change_volume(pitch_shift(y, sr, -2.0), 0.8)),
                ])

            for out_name, out_audio in variants:
                out_path = os.path.join(save_emotion_path, out_name)
                sf.write(out_path, out_audio, sr)
                total_created += 1

        print(f"  -> Generated {len(os.listdir(save_emotion_path))} files in {save_emotion_path}")

    print(f"\nDataset Augmentation Completed! Total files created: {total_created}")


if __name__ == "__main__":
    augment_all()