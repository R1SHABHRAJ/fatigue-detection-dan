import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from numpy.fft import rfft, rfftfreq


# ── Frequency band power ─────────────────────────────────────────────────

def bandpower(signal, fs, low, high):
    """Compute average power in a frequency band using FFT."""
    n = len(signal)

    freqs = rfftfreq(n, d=1.0 / fs)
    fft_mag = np.abs(rfft(signal)) ** 2 / n

    band_mask = (freqs >= low) & (freqs < high)

    return fft_mag[band_mask].mean() if band_mask.sum() > 0 else 0.0


# ── Peak counting ────────────────────────────────────────────────────────

def count_peaks(signal, threshold_factor=1.5):
    """Count local peaks above mean + threshold_factor * std."""
    threshold = signal.mean() + threshold_factor * signal.std()

    peaks = 0

    for i in range(1, len(signal) - 1):
        if signal[i] > signal[i - 1] and signal[i] > signal[i + 1] and signal[i] > threshold:
            peaks += 1

    return peaks


# ── Feature extraction ───────────────────────────────────────────────────

def extract_features(window, fs=250):
    """
    Extract 23 features from ECG, EEG, and GSR signals.

    Input:
        window → DataFrame containing ECG, EEG, GSR

    Output:
        dictionary of features
    """

    ecg = window["ECG"].values
    eeg = window["EEG"].values
    gsr = window["GSR"].values

    features = {}

    # ── ECG features ───────────────────────────────────────────────────

    features["ECG_mean"] = ecg.mean()
    features["ECG_std"] = ecg.std()
    features["ECG_rms"] = np.sqrt(np.mean(ecg ** 2))
    features["ECG_iqr"] = np.percentile(ecg, 75) - np.percentile(ecg, 25)
    features["ECG_skew"] = skew(ecg)
    features["ECG_kurt"] = kurtosis(ecg)
    features["ECG_ptp"] = ecg.max() - ecg.min()

    lf = bandpower(ecg, fs, 0.04, 0.15)
    hf = bandpower(ecg, fs, 0.15, 0.40)

    features["ECG_lf_hf"] = lf / (hf + 1e-8)

    # ── EEG features ───────────────────────────────────────────────────

    features["EEG_mean"] = eeg.mean()
    features["EEG_std"] = eeg.std()

    features["EEG_delta"] = bandpower(eeg, fs, 1, 4)
    features["EEG_theta"] = bandpower(eeg, fs, 4, 8)
    features["EEG_alpha"] = bandpower(eeg, fs, 8, 13)
    features["EEG_beta"] = bandpower(eeg, fs, 13, 30)

    features["EEG_ab_ratio"] = features["EEG_alpha"] / (features["EEG_beta"] + 1e-8)
    features["EEG_ta_ratio"] = features["EEG_theta"] / (features["EEG_alpha"] + 1e-8)

    # ── GSR features ───────────────────────────────────────────────────

    features["GSR_mean"] = gsr.mean()
    features["GSR_std"] = gsr.std()
    features["GSR_max"] = gsr.max()
    features["GSR_iqr"] = np.percentile(gsr, 75) - np.percentile(gsr, 25)

    gsr_diff = np.diff(gsr)

    features["GSR_roc_mean"] = gsr_diff.mean()
    features["GSR_roc_std"] = gsr_diff.std()

    features["GSR_peaks"] = count_peaks(gsr)

    return features


# ── Window creation ─────────────────────────────────────────────────────

def create_feature_windows(data, window_seconds=10, overlap=0.5, fs=250):
    """
    Segment signals into windows and extract features.

    Returns:
        DataFrame with features per window
    """

    window_size = window_seconds * fs
    step = int(window_size * (1 - overlap))

    rows = []

    for driver in sorted(data["driver"].unique()):

        df_d = data[data["driver"] == driver].reset_index(drop=True)

        label = df_d["label"].iloc[0]

        for start in range(0, len(df_d) - window_size, step):

            window = df_d.iloc[start:start + window_size]

            feat = extract_features(window, fs)

            feat["driver"] = driver
            feat["label"] = label

            rows.append(feat)

    return pd.DataFrame(rows)