from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np

@dataclass
class AudioAnalysis:
    duration_sec: float
    rms_db: float
    snr_est_db: float
    clipping_ratio: float
    zcr: float
    quality_score: float
    feedback: str

@dataclass
class PipelineResult:
    report_text: str
    raw_path: Path
    enhanced_path: Path
    raw_audio: np.ndarray
    enhanced_audio: np.ndarray
    before_analysis: AudioAnalysis
    after_analysis: AudioAnalysis
    comparison: dict
    plot_path: Optional[Path] = None