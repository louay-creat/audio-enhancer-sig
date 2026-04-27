from models import AudioAnalysis


def build_report(
    before_analysis: AudioAnalysis,
    after_analysis: AudioAnalysis,
    comparison: dict,
    raw_path,
    enhanced_path,
) -> str:
    return (
        f"=== RESULTAT DU TRAITEMENT ===\n\n"
        f"Score qualite global : {after_analysis.quality_score:.2f} / 100\n"
        f"Gain de score        : {comparison['score_gain']:+.2f} points\n"
        f"Duree                : {after_analysis.duration_sec:.2f} s\n\n"
        f"--- Comparaison avant / apres ---\n"
        f"RMS avant            : {before_analysis.rms_db:.2f} dB\n"
        f"RMS apres            : {after_analysis.rms_db:.2f} dB\n"
        f"Gain RMS             : {comparison['rms_gain_db']:+.2f} dB\n\n"
        f"SNR avant            : {before_analysis.snr_est_db:.2f} dB\n"
        f"SNR apres            : {after_analysis.snr_est_db:.2f} dB\n"
        f"Gain SNR             : {comparison['snr_gain_db']:+.2f} dB\n\n"
        f"Clipping avant       : {before_analysis.clipping_ratio * 100:.2f} %\n"
        f"Clipping apres       : {after_analysis.clipping_ratio * 100:.2f} %\n"
        f"Reduction clipping   : {comparison['clipping_reduction_pct']:+.2f} %\n\n"
        f"ZCR avant            : {before_analysis.zcr:.4f}\n"
        f"ZCR apres            : {after_analysis.zcr:.4f}\n"
        f"Variation ZCR        : {comparison['zcr_variation']:+.4f}\n\n"
        f"=== FEEDBACK ===\n{after_analysis.feedback}\n\n"
        f"Audio brut     : {raw_path}\n"
        f"Audio ameliore : {enhanced_path}\n"
    )