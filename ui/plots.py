from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def style_axis(ax):
    ax.set_facecolor("#0f172a")
    ax.tick_params(colors="#94a3b8")

    for spine in ax.spines.values():
        spine.set_color("#334155")


def clear_plots(fig, canvas):
    fig.clear()
    fig.patch.set_facecolor("#111a2b")

    axes = fig.subplots(2, 2)

    titles = [
        "Signal avant traitement",
        "Signal apres traitement",
        "Spectrogramme avant traitement",
        "Spectrogramme apres traitement",
    ]

    for ax, title in zip(axes.flatten(), titles):
        ax.clear()
        ax.set_title(title, color="#e2e8f0")
        ax.grid(True, alpha=0.2)
        style_axis(ax)

    fig.tight_layout(pad=2.5)
    canvas.draw_idle()

    return axes


def plot_audio_comparison(
    fig,
    canvas,
    raw_audio: np.ndarray,
    enhanced_audio: np.ndarray,
    sample_rate: int,
    comparison: dict | None,
    plot_path: Path,
):
    fig.clear()
    fig.patch.set_facecolor("#111a2b")

    axes = fig.subplots(2, 2)

    raw_audio = np.asarray(raw_audio, dtype=np.float32).flatten()
    enhanced_audio = np.asarray(enhanced_audio, dtype=np.float32).flatten()

    t_raw = np.linspace(0, len(raw_audio) / sample_rate, num=len(raw_audio))
    t_enh = np.linspace(0, len(enhanced_audio) / sample_rate, num=len(enhanced_audio))

    raw_peak = max(float(np.max(np.abs(raw_audio))), 1e-6)
    enh_peak = max(float(np.max(np.abs(enhanced_audio))), 1e-6)
    common_ylim = max(raw_peak, enh_peak) * 1.05

    ax1 = axes[0, 0]
    ax1.plot(t_raw, raw_audio, linewidth=0.8, color="#94a3b8")
    ax1.set_title("Signal avant traitement", color="#e2e8f0")
    ax1.set_xlabel("Temps (s)", color="#94a3b8")
    ax1.set_ylabel("Amplitude", color="#94a3b8")
    ax1.set_ylim(-common_ylim, common_ylim)
    ax1.grid(True, alpha=0.2)
    style_axis(ax1)

    ax2 = axes[0, 1]
    ax2.plot(t_enh, enhanced_audio, linewidth=1.2, color="#38bdf8")
    ax2.set_title("Signal apres traitement", color="#e2e8f0")
    ax2.set_xlabel("Temps (s)", color="#94a3b8")
    ax2.set_ylabel("Amplitude", color="#94a3b8")
    ax2.set_ylim(-common_ylim, common_ylim)
    ax2.grid(True, alpha=0.2)
    style_axis(ax2)

    # Spectrogramme audio brut
    ax3 = axes[1, 0]
    spec_raw = librosa.stft(raw_audio, n_fft=1024, hop_length=256)
    spec_raw_db = librosa.amplitude_to_db(np.abs(spec_raw), ref=np.max)

    img1 = librosa.display.specshow(
        spec_raw_db,
        sr=sample_rate,
        hop_length=256,
        x_axis="time",
        y_axis="hz",
        ax=ax3,
        cmap="magma",
    )

    ax3.set_title("Spectrogramme avant traitement", color="#e2e8f0")
    ax3.set_xlabel("Temps (s)", color="#94a3b8")
    ax3.set_ylabel("Frequence (Hz)", color="#94a3b8")
    style_axis(ax3)

    # Spectrogramme audio ameliore
    ax4 = axes[1, 1]
    spec_enh = librosa.stft(enhanced_audio, n_fft=1024, hop_length=256)
    spec_enh_db = librosa.amplitude_to_db(np.abs(spec_enh), ref=np.max)

    img2 = librosa.display.specshow(
        spec_enh_db,
        sr=sample_rate,
        hop_length=256,
        x_axis="time",
        y_axis="hz",
        ax=ax4,
        cmap="magma",
    )

    ax4.set_title("Spectrogramme apres traitement", color="#e2e8f0")
    ax4.set_xlabel("Temps (s)", color="#94a3b8")
    ax4.set_ylabel("Frequence (Hz)", color="#94a3b8")
    style_axis(ax4)

    cbar1 = fig.colorbar(img1, ax=ax3, format="%+2.0f dB")
    cbar2 = fig.colorbar(img2, ax=ax4, format="%+2.0f dB")

    cbar1.ax.tick_params(colors="#94a3b8")
    cbar2.ax.tick_params(colors="#94a3b8")

    if comparison:
        gain = comparison.get("snr_gain_db", 0.0)
        fig.suptitle(
            f"Comparaison avant / apres | Gain SNR: {gain:+.2f} dB",
            fontsize=15,
            fontweight="bold",
            color="#f8fafc",
        )

    fig.tight_layout(rect=[0, 0, 1, 0.95], pad=2.2)

    fig.savefig(
        plot_path,
        dpi=180,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )

    canvas.draw_idle()

    return axes