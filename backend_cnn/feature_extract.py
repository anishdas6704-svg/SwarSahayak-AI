import warnings
import librosa
import numpy as np

warnings.filterwarnings("ignore")


def extract_features(file_path_or_audio, sr=22050):
    """
    Extract comprehensive 526-dimensional acoustic features for Speech Emotion Recognition (SER).
    Accepts either an audio file path (str) or a pre-loaded 1D numpy array of audio samples.
    Optimized for high-accuracy emotion classification and ultra-fast feature extraction.
    """
    if isinstance(file_path_or_audio, str):
        y, sample_rate = librosa.load(file_path_or_audio, sr=sr, duration=5.0)
    else:
        y = np.asarray(file_path_or_audio, dtype=np.float32)
        sample_rate = sr

    # Ensure audio is not empty
    if len(y) == 0 or np.max(np.abs(y)) < 1e-6:
        y = np.pad(y, (0, max(0, int(sample_rate * 0.5) - len(y))), mode='constant') + 1e-6

    # Trim leading/trailing silence
    y_trimmed, _ = librosa.effects.trim(y, top_db=30)
    if len(y_trimmed) >= int(sample_rate * 0.3):
        y = y_trimmed

    n_fft = 1024
    hop_length = 512
    stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))

    # 1. MFCCs (40 coefficients) - mean, std, max, min
    mfcc = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=40, n_fft=n_fft, hop_length=hop_length)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    mfcc_max = np.max(mfcc, axis=1)
    mfcc_min = np.min(mfcc, axis=1)

    # 2. Delta & Delta-Delta MFCCs - mean, std
    delta_mfcc = librosa.feature.delta(mfcc)
    delta_mean = np.mean(delta_mfcc, axis=1)
    delta_std = np.std(delta_mfcc, axis=1)

    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    delta2_mean = np.mean(delta2_mfcc, axis=1)
    delta2_std = np.std(delta2_mfcc, axis=1)

    # 3. Log-Mel Spectrogram (64 frequency bands) - mean, std
    mel_spec = librosa.feature.melspectrogram(S=stft**2, sr=sample_rate, n_mels=64)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    mel_mean = np.mean(log_mel, axis=1)
    mel_std = np.std(log_mel, axis=1)

    # 4. Chroma STFT (12 pitch classes) - mean, std
    chroma_stft = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
    chroma_mean = np.mean(chroma_stft, axis=1)
    chroma_std = np.std(chroma_stft, axis=1)

    # 5. Tonnetz (6 tonal centroid features) - mean, std
    try:
        tonnetz = librosa.feature.tonnetz(chroma=chroma_stft)
        tonnetz_mean = np.mean(tonnetz, axis=1)
        tonnetz_std = np.std(tonnetz, axis=1)
    except Exception:
        tonnetz_mean = np.zeros(6, dtype=np.float32)
        tonnetz_std = np.zeros(6, dtype=np.float32)

    # 6. Spectral Contrast across 7 subbands - mean, std
    contrast = librosa.feature.spectral_contrast(S=stft, sr=sample_rate)
    contrast_mean = np.mean(contrast, axis=1)
    contrast_std = np.std(contrast, axis=1)

    # 7. Spectral Descriptors (Centroid, Bandwidth, Rolloff 85%, Rolloff 95%, Flatness)
    centroid = librosa.feature.spectral_centroid(S=stft, sr=sample_rate)
    bandwidth = librosa.feature.spectral_bandwidth(S=stft, sr=sample_rate)
    rolloff_85 = librosa.feature.spectral_rolloff(S=stft, sr=sample_rate, roll_percent=0.85)
    rolloff_95 = librosa.feature.spectral_rolloff(S=stft, sr=sample_rate, roll_percent=0.95)
    flatness = librosa.feature.spectral_flatness(S=stft)

    spectral_stats = np.array([
        np.mean(centroid), np.std(centroid), np.max(centroid), np.min(centroid),
        np.mean(bandwidth), np.std(bandwidth), np.max(bandwidth), np.min(bandwidth),
        np.mean(rolloff_85), np.std(rolloff_85),
        np.mean(rolloff_95), np.std(rolloff_95),
        np.mean(flatness), np.std(flatness)
    ], dtype=np.float32)

    # 8. Zero Crossing Rate & RMS Energy
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    energy_stats = np.array([
        np.mean(zcr), np.std(zcr), np.max(zcr),
        np.mean(rms), np.std(rms), np.max(rms), np.min(rms)
    ], dtype=np.float32)

    # 9. Spectral Flux & Peak Dynamics
    flux = np.sqrt(np.sum(np.diff(stft, axis=1)**2, axis=0)) if stft.shape[1] > 1 else np.zeros(1)
    peak_ratio = np.max(stft) / (np.mean(stft) + 1e-8)
    flux_stats = np.array([
        np.mean(flux), np.std(flux), np.max(flux), np.min(flux),
        float(peak_ratio), float(np.percentile(rms, 90)), float(np.percentile(rms, 10))
    ], dtype=np.float32)

    # Concatenate full feature vector (526 dimensions)
    feature_vector = np.hstack([
        mfcc_mean,       # 40
        mfcc_std,        # 40
        mfcc_max,        # 40
        mfcc_min,        # 40
        delta_mean,      # 40
        delta_std,       # 40
        delta2_mean,     # 40
        delta2_std,      # 40
        mel_mean,        # 64
        mel_std,         # 64
        chroma_mean,     # 12
        chroma_std,      # 12
        tonnetz_mean,    # 6
        tonnetz_std,     # 6
        contrast_mean,   # 7
        contrast_std,    # 7
        spectral_stats,  # 14
        energy_stats,    # 7
        flux_stats       # 7
    ])

    return np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


if __name__ == "__main__":
    import os
    import time
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sample_path = os.path.join(base_dir, "dataset", "sad", "1.wav")
    if os.path.exists(sample_path):
        t0 = time.time()
        feats = extract_features(sample_path)
        dt = (time.time() - t0) * 1000
        print(f"Extracted feature vector shape: {feats.shape} in {dt:.2f} ms")