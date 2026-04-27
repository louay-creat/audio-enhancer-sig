import queue
import threading

import customtkinter as ctk
import matplotlib.pyplot as plt
import soundfile as sf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from audio.analysis import analyze_audio, compare_analyses
from audio.processing import enhance_audio
from audio.recorder import record_audio, play_audio
from config import (
    APP_TITLE,
    CHANNELS,
    ENHANCED_AUDIO_PATH,
    PDF_REPORT_PATH,
    PLOT_PATH,
    RAW_AUDIO_PATH,
    RECORD_SECONDS,
    SAMPLE_RATE,
    TXT_REPORT_PATH,
)
from reports.exporter import export_pdf as save_pdf
from reports.exporter import export_txt as save_txt
from reports.formatter import build_report
from ui.plots import clear_plots, plot_audio_comparison


class AudioEnhancerApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title(APP_TITLE)
        self.geometry("1320x960")

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.configure(fg_color="#0b1220")

        self.worker_queue = queue.Queue()
        self.is_processing = False
        self.progress_value = 0

        self.last_raw_path = None
        self.last_enhanced_path = None
        self.last_report_text = ""
        self.last_plot_path = None
        self.last_after_analysis = None
        self.last_comparison = None

        self._record_anim_phase = 0
        self._record_anim_job = None

        self._build_ui()
        self.after(150, self._poll_queue)

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(5, weight=1)
        self.grid_rowconfigure(6, weight=2)

        header = ctk.CTkFrame(
            self,
            corner_radius=28,
            fg_color="#111a2b",
            border_width=1,
            border_color="#22304a",
        )
        header.grid(row=0, column=0, padx=24, pady=(20, 12), sticky="ew")
        header.grid_columnconfigure(1, weight=1)

        logo = ctk.CTkLabel(
            header,
            text="SIG",
            width=74,
            height=74,
            corner_radius=37,
            font=ctk.CTkFont(size=24, weight="bold"),
            fg_color="#2563eb",
            text_color="white",
        )
        logo.grid(row=0, column=0, padx=18, pady=16)

        title_block = ctk.CTkFrame(header, fg_color="transparent")
        title_block.grid(row=0, column=1, padx=(0, 18), pady=14, sticky="ew")

        title = ctk.CTkLabel(
            title_block,
            text="Audio Enhancement Studio",
            font=ctk.CTkFont(size=30, weight="bold"),
            text_color="#f8fafc",
        )
        title.grid(row=0, column=0, sticky="w")

        subtitle = ctk.CTkLabel(
            title_block,
            text="Enregistrement 10 s • Denoising • Analyse • Rapport PDF/TXT",
            font=ctk.CTkFont(size=14),
            text_color="#94a3b8",
        )
        subtitle.grid(row=1, column=0, sticky="w", pady=(5, 0))

        self.score_card = ctk.CTkFrame(
            self,
            corner_radius=26,
            fg_color="#111a2b",
            border_width=1,
            border_color="#22304a",
        )
        self.score_card.grid(row=1, column=0, padx=24, pady=(0, 12), sticky="ew")
        self.score_card.grid_columnconfigure((0, 1, 2, 3), weight=1)

        self.score_label = ctk.CTkLabel(
            self.score_card,
            text="Score qualite\n-- / 100",
            font=ctk.CTkFont(size=28, weight="bold"),
            justify="center",
            text_color="#f8fafc",
        )
        self.score_label.grid(row=0, column=0, padx=15, pady=16, sticky="nsew")

        self.delta_label = ctk.CTkLabel(
            self.score_card,
            text="Gain\n--",
            font=ctk.CTkFont(size=24, weight="bold"),
            justify="center",
            text_color="#38bdf8",
        )
        self.delta_label.grid(row=0, column=1, padx=15, pady=16, sticky="nsew")

        self.compare_label = ctk.CTkLabel(
            self.score_card,
            text="Comparaison\nSNR: -- dB",
            font=ctk.CTkFont(size=18),
            justify="center",
            text_color="#cbd5e1",
        )
        self.compare_label.grid(row=0, column=2, padx=15, pady=16, sticky="nsew")

        self.quality_badge = ctk.CTkLabel(
            self.score_card,
            text="Qualite\n--",
            font=ctk.CTkFont(size=20, weight="bold"),
            justify="center",
            corner_radius=18,
            width=170,
            height=74,
            fg_color="#475569",
            text_color="white",
        )
        self.quality_badge.grid(row=0, column=3, padx=15, pady=16, sticky="nsew")

        description = ctk.CTkLabel(
            self,
            text="Appuie sur le bouton principal pour enregistrer, ameliorer l'audio et afficher la comparaison avant/apres.",
            justify="left",
            text_color="#cbd5e1",
        )
        description.grid(row=2, column=0, padx=24, pady=(0, 10), sticky="w")

        controls = ctk.CTkFrame(
            self,
            corner_radius=26,
            fg_color="#111a2b",
            border_width=1,
            border_color="#22304a",
        )
        controls.grid(row=3, column=0, padx=24, pady=10, sticky="ew")

        for i in range(5):
            controls.grid_columnconfigure(i, weight=1)

        self.record_button = ctk.CTkButton(
            controls,
            text="● START RECORDING",
            command=self.start_pipeline,
            height=58,
            corner_radius=29,
            font=ctk.CTkFont(size=18, weight="bold"),
            fg_color="#2563eb",
            hover_color="#1d4ed8",
        )
        self.record_button.grid(row=0, column=0, padx=10, pady=16, sticky="ew")

        self.play_raw_button = ctk.CTkButton(
            controls,
            text="▶ Brut",
            command=self.play_raw,
            state="disabled",
            height=50,
            corner_radius=24,
            fg_color="#1e293b",
            hover_color="#334155",
        )
        self.play_raw_button.grid(row=0, column=1, padx=10, pady=16, sticky="ew")

        self.play_enhanced_button = ctk.CTkButton(
            controls,
            text="▶ Ameliore",
            command=self.play_enhanced,
            state="disabled",
            height=50,
            corner_radius=24,
            fg_color="#1e293b",
            hover_color="#334155",
        )
        self.play_enhanced_button.grid(row=0, column=2, padx=10, pady=16, sticky="ew")

        self.export_txt_button = ctk.CTkButton(
            controls,
            text="⬇ TXT",
            command=self.export_txt,
            state="disabled",
            height=50,
            corner_radius=24,
            fg_color="#1e293b",
            hover_color="#334155",
        )
        self.export_txt_button.grid(row=0, column=3, padx=10, pady=16, sticky="ew")

        self.export_pdf_button = ctk.CTkButton(
            controls,
            text="⬇ PDF",
            command=self.export_pdf,
            state="disabled",
            height=50,
            corner_radius=24,
            fg_color="#1e293b",
            hover_color="#334155",
        )
        self.export_pdf_button.grid(row=0, column=4, padx=10, pady=16, sticky="ew")

        progress_frame = ctk.CTkFrame(
            self,
            corner_radius=26,
            fg_color="#111a2b",
            border_width=1,
            border_color="#22304a",
        )
        progress_frame.grid(row=4, column=0, padx=24, pady=(0, 12), sticky="ew")
        progress_frame.grid_columnconfigure(0, weight=1)

        self.status_label = ctk.CTkLabel(
            progress_frame,
            text="Statut : pret",
            text_color="#cbd5e1",
        )
        self.status_label.grid(row=0, column=0, padx=16, pady=(12, 6), sticky="w")

        self.progress_bar = ctk.CTkProgressBar(
            progress_frame,
            height=18,
            corner_radius=10,
            progress_color="#2563eb",
        )
        self.progress_bar.grid(row=1, column=0, padx=16, pady=(6, 12), sticky="ew")
        self.progress_bar.set(0)

        self.results_box = ctk.CTkTextbox(
            self,
            wrap="word",
            height=220,
            corner_radius=22,
            border_width=1,
            border_color="#22304a",
            fg_color="#111a2b",
            text_color="#e2e8f0",
        )
        self.results_box.grid(row=5, column=0, padx=24, pady=(0, 12), sticky="nsew")
        self.results_box.insert("1.0", "Le feedback apparaitra ici...\n")
        self.results_box.configure(state="disabled")

        plots_frame = ctk.CTkFrame(
            self,
            corner_radius=26,
            fg_color="#111a2b",
            border_width=1,
            border_color="#22304a",
        )
        plots_frame.grid(row=6, column=0, padx=24, pady=(0, 22), sticky="nsew")
        plots_frame.grid_rowconfigure(0, weight=1)
        plots_frame.grid_columnconfigure(0, weight=1)

        self.fig = plt.figure(figsize=(11, 6.8), facecolor="#111a2b")
        self.canvas = FigureCanvasTkAgg(self.fig, master=plots_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        self.axes = clear_plots(self.fig, self.canvas)

    def set_status(self, text: str):
        self.status_label.configure(text=f"Statut : {text}")

    def write_results(self, text: str):
        self.results_box.configure(state="normal")
        self.results_box.delete("1.0", "end")
        self.results_box.insert("1.0", text)
        self.results_box.configure(state="disabled")

    def _set_quality_badge(self, score):
        if score is None:
            self.quality_badge.configure(text="Qualite\n--", fg_color="#475569")
        elif score < 40:
            self.quality_badge.configure(text="Qualite\nFaible", fg_color="#dc2626")
        elif score < 70:
            self.quality_badge.configure(text="Qualite\nMoyenne", fg_color="#f59e0b")
        else:
            self.quality_badge.configure(text="Qualite\nBonne", fg_color="#16a34a")

    def _set_buttons_processing_state(self):
        self.record_button.configure(state="disabled")
        self.play_raw_button.configure(state="disabled")
        self.play_enhanced_button.configure(state="disabled")
        self.export_txt_button.configure(state="disabled")
        self.export_pdf_button.configure(state="disabled")

    def _set_buttons_ready_state(self):
        self.record_button.configure(state="normal")
        self.play_raw_button.configure(state="normal")
        self.play_enhanced_button.configure(state="normal")
        self.export_txt_button.configure(state="normal")
        self.export_pdf_button.configure(state="normal")

    def _reset_score_cards(self):
        self.score_label.configure(text="Score qualite\n-- / 100")
        self.delta_label.configure(text="Gain\n--")
        self.compare_label.configure(text="Comparaison\nSNR: -- dB")
        self._set_quality_badge(None)

    def start_pipeline(self):
        if self.is_processing:
            return

        self.is_processing = True
        self.progress_value = 0

        self.progress_bar.set(0)
        self._set_buttons_processing_state()
        self.write_results("En attente d'enregistrement...")
        self.axes = clear_plots(self.fig, self.canvas)
        self._reset_score_cards()
        self.set_status("enregistrement en cours...")

        self._animate_progress()
        self._start_record_button_animation()

        threading.Thread(target=self._run_pipeline, daemon=True).start()

    def _run_pipeline(self):
        try:
            audio = record_audio(RECORD_SECONDS, SAMPLE_RATE, CHANNELS)

            self.worker_queue.put(("status", "traitement du signal..."))

            enhanced = enhance_audio(audio, SAMPLE_RATE)

            before_analysis = analyze_audio(audio, SAMPLE_RATE)
            after_analysis = analyze_audio(enhanced, SAMPLE_RATE)
            comparison = compare_analyses(before_analysis, after_analysis)

            sf.write(RAW_AUDIO_PATH, audio, SAMPLE_RATE)
            sf.write(ENHANCED_AUDIO_PATH, enhanced, SAMPLE_RATE)

            report = build_report(
                before_analysis=before_analysis,
                after_analysis=after_analysis,
                comparison=comparison,
                raw_path=RAW_AUDIO_PATH,
                enhanced_path=ENHANCED_AUDIO_PATH,
            )

            self.worker_queue.put(
                (
                    "done",
                    report,
                    RAW_AUDIO_PATH,
                    ENHANCED_AUDIO_PATH,
                    audio,
                    enhanced,
                    before_analysis,
                    after_analysis,
                    comparison,
                )
            )

        except Exception as exc:
            self.worker_queue.put(("error", str(exc)))

    def _poll_queue(self):
        try:
            while True:
                item = self.worker_queue.get_nowait()
                action = item[0]

                if action == "status":
                    self.set_status(item[1])

                elif action == "done":
                    self._handle_done(item)

                elif action == "error":
                    self._handle_error(item[1])

        except queue.Empty:
            pass

        self.after(150, self._poll_queue)

    def _handle_done(self, item):
        (
            _,
            report,
            raw_path,
            enhanced_path,
            raw_audio,
            enhanced_audio,
            before_analysis,
            after_analysis,
            comparison,
        ) = item

        self.last_report_text = report
        self.last_raw_path = raw_path
        self.last_enhanced_path = enhanced_path
        self.last_after_analysis = after_analysis
        self.last_comparison = comparison
        self.last_plot_path = PLOT_PATH

        self.write_results(report)

        self.axes = plot_audio_comparison(
            self.fig,
            self.canvas,
            raw_audio,
            enhanced_audio,
            SAMPLE_RATE,
            comparison,
            PLOT_PATH,
        )

        self.progress_bar.set(1)

        self.score_label.configure(
            text=f"Score qualite\n{after_analysis.quality_score:.1f} / 100"
        )

        self.delta_label.configure(
            text=f"Gain\n{comparison['score_gain']:+.1f} pts"
        )

        self.compare_label.configure(
            text=(
                f"Comparaison\n"
                f"SNR: {comparison['snr_gain_db']:+.2f} dB\n"
                f"Clip: {comparison['clipping_reduction_pct']:+.2f} %"
            )
        )

        self._set_quality_badge(after_analysis.quality_score)
        self._stop_record_button_animation()
        self.set_status("termine")
        self._set_buttons_ready_state()

        self.is_processing = False

    def _handle_error(self, message: str):
        self.write_results(f"Erreur : {message}")
        self._stop_record_button_animation()
        self.set_status("erreur")
        self.record_button.configure(state="normal")
        self.is_processing = False

    def _animate_progress(self):
        if not self.is_processing:
            return

        progress = min(1.0, self.progress_value / (RECORD_SECONDS * 10))
        self.progress_bar.set(progress)
        self.progress_value += 1

        if self.progress_value <= RECORD_SECONDS * 10:
            self.after(100, self._animate_progress)

    def _start_record_button_animation(self):
        self._record_anim_phase = 0
        self._animate_record_button()

    def _animate_record_button(self):
        if not self.is_processing:
            self.record_button.configure(
                text="● START RECORDING",
                fg_color="#2563eb",
                hover_color="#1d4ed8",
            )
            return

        frames = [
            ("● RECORDING...", "#dc2626"),
            ("◉ RECORDING...", "#ef4444"),
            ("● RECORDING...", "#b91c1c"),
        ]

        text, color = frames[self._record_anim_phase % len(frames)]
        self.record_button.configure(text=text, fg_color=color, hover_color=color)

        self._record_anim_phase += 1
        self._record_anim_job = self.after(350, self._animate_record_button)

    def _stop_record_button_animation(self):
        if self._record_anim_job is not None:
            try:
                self.after_cancel(self._record_anim_job)
            except Exception:
                pass

            self._record_anim_job = None

        self.record_button.configure(
            text="● START RECORDING",
            fg_color="#2563eb",
            hover_color="#1d4ed8",
        )

    def play_raw(self):
        if self.last_raw_path and self.last_raw_path.exists():
            audio, sr = sf.read(self.last_raw_path)
            play_audio(audio, sr)

    def play_enhanced(self):
        if self.last_enhanced_path and self.last_enhanced_path.exists():
            audio, sr = sf.read(self.last_enhanced_path)
            play_audio(audio, sr)

    def export_txt(self):
        if not self.last_report_text:
            return

        out_path = save_txt(self.last_report_text, TXT_REPORT_PATH)
        self.set_status(f"rapport TXT exporte : {out_path}")

    def export_pdf(self):
        if not self.last_report_text:
            return

        out_path = save_pdf(
            report_text=self.last_report_text,
            out_path=PDF_REPORT_PATH,
            plot_path=self.last_plot_path,
            after_analysis=self.last_after_analysis,
            comparison=self.last_comparison,
        )

        self.set_status(f"rapport PDF exporte : {out_path}")