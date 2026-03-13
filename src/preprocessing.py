import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch

FS = 250  # Sampling frequency


# ── Filter functions ─────────────────────────────────────────────────────

def bandpass_filter(signal, lowcut, highcut, fs=250, order=4):
    """Apply a Butterworth bandpass filter."""
    nyquist = 0.5 * fs
    b, a = butter(order, [lowcut / nyquist, highcut / nyquist], btype="band")
    return filtfilt(b, a, signal)


def notch_filter(signal, notch_freq=50, fs=250, Q=30):
    """Apply notch filter to remove power line interference."""
    b, a = iirnotch(notch_freq / (0.5 * fs), Q)
    return filtfilt(b, a, signal)


def clip_outliers_zscore(series, threshold=3.0):
    """Clip values beyond ±threshold standard deviations."""
    mean_val = series.mean()
    std_val = series.std()
    return series.clip(mean_val - threshold * std_val,
                       mean_val + threshold * std_val)


# ── Main preprocessing function ──────────────────────────────────────────

def preprocess_signals(data):
    """
    Apply preprocessing to ECG, EEG and GSR signals per driver.

    Steps:
    - Notch filter
    - Bandpass filter
    - Outlier clipping
    """

    data = data.copy()

    for driver in sorted(data["driver"].unique()):

        mask = data["driver"] == driver

        # ECG
        ecg = data.loc[mask, "ECG"].values
        ecg = notch_filter(ecg)
        ecg = bandpass_filter(ecg, 0.5, 40)
        data.loc[mask, "ECG"] = ecg

        # EEG
        eeg = data.loc[mask, "EEG"].values
        if not np.all(eeg == 0):
            eeg = notch_filter(eeg)
            eeg = bandpass_filter(eeg, 1.0, 45.0)
        data.loc[mask, "EEG"] = eeg

        # GSR
        gsr = data.loc[mask, "GSR"].values
        gsr = bandpass_filter(gsr, 0.05, 5.0)
        data.loc[mask, "GSR"] = gsr

    return data