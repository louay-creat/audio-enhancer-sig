from pathlib import Path
import tempfile

APP_TITLE = "Audio Enhancer - SIG"
RECORD_SECONDS = 10
SAMPLE_RATE = 16000
CHANNELS = 1

TEMP_DIR = Path(tempfile.gettempdir()) / "audio_enhancer_sig"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

RAW_AUDIO_PATH = TEMP_DIR / "recorded_raw.wav"
ENHANCED_AUDIO_PATH = TEMP_DIR / "recorded_enhanced.wav"
PLOT_PATH = TEMP_DIR / "audio_report_plots.png"
TXT_REPORT_PATH = TEMP_DIR / "rapport_traitement_audio.txt"
PDF_REPORT_PATH = TEMP_DIR / "rapport_traitement_audio.pdf"