import librosa
import numpy as np


def extract_features(file_path):

    y, sr = librosa.load(
        file_path,
        sr=22050,
        duration=5.0
    )

    # -------------------------
    # MFCC
    # -------------------------
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=40
    )

    mfcc_mean = np.mean(mfcc.T, axis=0)

    # -------------------------
    # Delta MFCC
    # -------------------------
    delta_mfcc = librosa.feature.delta(mfcc)

    delta_mean = np.mean(
        delta_mfcc.T,
        axis=0
    )

    # -------------------------
    # Delta Delta MFCC
    # -------------------------
    delta2_mfcc = librosa.feature.delta(
        mfcc,
        order=2
    )

    delta2_mean = np.mean(
        delta2_mfcc.T,
        axis=0
    )

    # -------------------------
    # Chroma
    # -------------------------
    stft = np.abs(librosa.stft(y))

    chroma = librosa.feature.chroma_stft(
        S=stft,
        sr=sr
    )

    chroma_mean = np.mean(
        chroma.T,
        axis=0
    )

    # -------------------------
    # Spectral Features
    # -------------------------
    centroid = np.mean(
        librosa.feature.spectral_centroid(
            y=y,
            sr=sr
        )
    )

    bandwidth = np.mean(
        librosa.feature.spectral_bandwidth(
            y=y,
            sr=sr
        )
    )

    contrast = np.mean(
        librosa.feature.spectral_contrast(
            S=stft,
            sr=sr
        )
    )

    rolloff = np.mean(
        librosa.feature.spectral_rolloff(
            y=y,
            sr=sr
        )
    )

    zcr = np.mean(
        librosa.feature.zero_crossing_rate(y)
    )

    rms = np.mean(
        librosa.feature.rms(y=y)
    )

    # -------------------------
    # Pitch
    # -------------------------
    pitches, magnitudes = librosa.piptrack(
        y=y,
        sr=sr
    )

    pitch = np.mean(
        pitches[pitches > 0]
    )

    if np.isnan(pitch):
        pitch = 0

    # -------------------------
    # Final Feature Vector
    # -------------------------
    feature_vector = np.hstack([

        mfcc_mean,          # 40

        delta_mean,         # 40

        delta2_mean,        # 40

        chroma_mean,        # 12

        centroid,
        bandwidth,
        contrast,
        rolloff,
        zcr,
        rms,
        pitch

    ])

    return feature_vector