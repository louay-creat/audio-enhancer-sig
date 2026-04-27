import librosa
import numpy as np
from models import AudioAnalysis


def estimate_snr_db(audio: np.ndarray, sr: int) -> float:
    if len(audio) < sr:
        return 0.0

    noise_len = min(len(audio), int(0.5 * sr))
    noise_part = audio[:noise_len]

    signal_power = np.mean(audio ** 2) + 1e-12
    noise_power = np.mean(noise_part ** 2) + 1e-12

    return float(10 * np.log10(signal_power / noise_power))


def analyze_audio(audio: np.ndarray, sr: int) -> AudioAnalysis:
    duration = len(audio) / sr
    rms = np.sqrt(np.mean(audio ** 2) + 1e-12)
    rms_db = float(20 * np.log10(rms + 1e-12))
    snr_est = estimate_snr_db(audio, sr)
    clipping_ratio = float(np.mean(np.abs(audio) >= 0.99))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=audio)[0]))

    comments = []

    if snr_est < 5:
        comments.append("Beaucoup de bruit detecte. Essaie un micro plus proche ou un environnement plus calme.")
    elif snr_est < 12:
        comments.append("Le signal est moyen. L'amelioration est utile mais la qualite reste limitee.")
    else:
        comments.append("Le niveau de bruit semble acceptable apres traitement.")

    if rms_db < -35:
        comments.append("Le volume est faible. Parle plus pres du micro ou augmente le gain d'entree.")
    elif rms_db > -10:
        comments.append("Le volume est eleve. Verifie qu'il n'y a pas de saturation.")
    else:
        comments.append("Le volume est globalement correct.")

    if clipping_ratio > 0.01:
        comments.append("Presence probable de saturation. Reduis le niveau d'enregistrement.")
    else:
        comments.append("Pas de saturation importante detectee.")

    if zcr > 0.18:
        comments.append("Le signal contient beaucoup de bruit residuel.")

    score = 100.0
    score -= max(0.0, (8.0 - snr_est) * 4.0)
    score -= max(0.0, (-22.0 - rms_db) * 1.2)
    score -= min(25.0, clipping_ratio * 1200.0)
    score -= max(0.0, (zcr - 0.15) * 120.0)
    score = float(max(0.0, min(100.0, score)))

    return AudioAnalysis(
        duration_sec=float(duration),
        rms_db=rms_db,
        snr_est_db=float(snr_est),
        clipping_ratio=clipping_ratio,
        zcr=zcr,
        quality_score=score,
        feedback="\n".join(f"- {comment}" for comment in comments),
    )


def compare_analyses(before: AudioAnalysis, after: AudioAnalysis) -> dict:
    return {
        "snr_gain_db": after.snr_est_db - before.snr_est_db,
        "rms_gain_db": after.rms_db - before.rms_db,
        "clipping_reduction_pct": (before.clipping_ratio - after.clipping_ratio) * 100.0,
        "zcr_variation": after.zcr - before.zcr,
        "score_gain": after.quality_score - before.quality_score,
    }