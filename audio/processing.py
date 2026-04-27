import numpy as np
import noisereduce as nr
from scipy.signal import butter, lfilter


def butter_bandpass_filter(
    data: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: int,
    order: int = 4,
) -> np.ndarray:
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return lfilter(b, a, data)


def normalize_audio(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    peak = np.max(np.abs(audio)) + 1e-9
    return audio * (target_peak / peak)


def enhance_audio(audio: np.ndarray, sr: int) -> np.ndarray:
    reduced_noise = nr.reduce_noise(
        y=audio,
        sr=sr,
        stationary=False,
        prop_decrease=0.8,
    )
    filtered = butter_bandpass_filter(
        reduced_noise,
        lowcut=80.0,
        highcut=3800.0,
        fs=sr,
    )
    return normalize_audio(filtered).astype(np.float32)