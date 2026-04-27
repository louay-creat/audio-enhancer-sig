from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas as pdf_canvas
from models import AudioAnalysis


def export_txt(report_text: str, out_path: Path) -> Path:
    out_path.write_text(report_text, encoding="utf-8")
    return out_path


def export_pdf(
    report_text: str,
    out_path: Path,
    plot_path: Path | None = None,
    after_analysis: AudioAnalysis | None = None,
    comparison: dict | None = None,
) -> Path:
    pdf = pdf_canvas.Canvas(str(out_path), pagesize=A4)
    _, height = A4

    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(50, height - 50, "Projet de traitement du signal audio")
    pdf.setFont("Helvetica", 11)
    pdf.drawString(50, height - 70, "Rapport d'analyse et d'amelioration audio")

    y = height - 100
    for line in report_text.splitlines():
        pdf.drawString(50, y, line[:110])
        y -= 15
        if y < 270:
            break

    if plot_path and Path(plot_path).exists():
        pdf.drawImage(
            str(plot_path),
            45,
            40,
            width=500,
            height=210,
            preserveAspectRatio=True,
            mask="auto",
        )

    pdf.showPage()
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(50, height - 50, "Resume de performance")
    pdf.setFont("Helvetica", 11)

    y = height - 85
    if after_analysis and comparison:
        lines = [
            f"Score global final : {after_analysis.quality_score:.2f} / 100",
            f"Gain de score      : {comparison['score_gain']:+.2f} points",
            f"Gain SNR           : {comparison['snr_gain_db']:+.2f} dB",
            f"Gain RMS           : {comparison['rms_gain_db']:+.2f} dB",
            f"Reduction clipping : {comparison['clipping_reduction_pct']:+.2f} %",
            f"Variation ZCR      : {comparison['zcr_variation']:+.4f}",
        ]
        for line in lines:
            pdf.drawString(50, y, line)
            y -= 18

    pdf.save()
    return out_path