import numpy as np
import sounddevice as sd


def record_audio(seconds: int, sr: int, channels: int) -> np.ndarray:
    recording = sd.rec(
        int(seconds * sr),
        samplerate=sr,
        channels=channels,
        dtype="float32",
    )
    sd.wait()

    if channels == 1:
        return recording.flatten()

    return np.mean(recording, axis=1)


def play_audio(audio, sr: int) -> None:
    sd.play(audio, sr)