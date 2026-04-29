from __future__ import annotations

import threading
import time
from queue import Empty, Queue

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

from orquestrador.adapters.stt import LocalSTT
from orquestrador.config import settings

RESET_SIM_CMD = "__reset_simulation__"


class UnifiedGUI:
    COLORS = {
        "bg": "#0d1117",
        "bg_secondary": "#161b22",
        "bg_tertiary": "#21262d",
        "accent": "#58a6ff",
        "accent_green": "#3fb950",
        "accent_orange": "#d29922",
        "accent_red": "#f85149",
        "accent_purple": "#a371f7",
        "accent_cyan": "#39c5cf",
        "accent_yellow": "#e3c55e",
        "text": "#c9d1d9",
        "text_muted": "#8b949e",
        "border": "#30363d",
    }

    SHORTCUTS = {
        "VISAO": [
            ("Detectar", "detectar objetos"),
            ("Loc. Cubo", "localizar cubo"),
            ("Loc. Esfera", "localizar esfera"),
            ("Limpar", "limpar deteccoes"),
        ],
        "MOVIMENTO": [
            ("Home", "ir para home"),
            ("Centro", "mover para [0, 0.3, 0.3]"),
            ("Esquerda", "mover para [-0.2, 0.3, 0.25]"),
            ("Direita", "mover para [0.2, 0.3, 0.25]"),
        ],
        "JUNTAS": [
            ("Base +30", "girar base 30 graus"),
            ("Base -30", "girar base -30 graus"),
            ("Ombro +15", "girar ombro 15 graus"),
        ],
        "GARRA": [
            ("Abrir", "abrir garra"),
            ("Fechar", "fechar garra"),
        ],
        "MANIPULACAO": [
            ("Pegar Cubo", "pegue o cubo"),
            ("Pegar Esfera", "pegue a esfera"),
            ("Depositar", "depositar no centro"),
        ],
        "INFO": [
            ("Estado", "qual e o estado atual do robo?"),
        ],
    }

    def __init__(self, cmd_queue: Queue, log_queue: Queue, frame_queue: Queue, running: list[bool]):
        self.cmd_queue = cmd_queue
        self.log_queue = log_queue
        self.frame_queue = frame_queue
        self.running = running
        self.history: list[str] = []
        self.history_idx = 0

        self.photo1 = None
        self.photo2 = None
        self._stt: LocalSTT | None = None
        self._stt_lock = threading.Lock()
        self._mic_state = "idle"
        self._mic_recording = False
        self._mic_chunks: list[np.ndarray] = []
        self._mic_lock = threading.Lock()
        self._mic_worker: threading.Thread | None = None

        self._build()
        threading.Thread(target=self._preload_stt, daemon=True).start()

    def _build(self) -> None:
        self.root = tk.Tk()
        self.root.title("UR3 Robot Controller")
        self.root.geometry("1150x750")
        self.root.configure(bg=self.COLORS["bg"])
        self.root.resizable(True, True)

        try:
            self.root.state("zoomed")
        except Exception:
            try:
                self.root.attributes("-zoomed", True)
            except Exception:
                pass

        main_frame = tk.Frame(self.root, bg=self.COLORS["bg"])
        main_frame.pack(fill="both", expand=True, padx=5, pady=5)

        self._build_title_bar(main_frame)

        content_frame = tk.Frame(main_frame, bg=self.COLORS["bg"])
        content_frame.pack(fill="both", expand=True, pady=(5, 0))

        left_panel = tk.Frame(content_frame, bg=self.COLORS["bg"])
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 5))

        self._build_input_panel(left_panel)
        self._build_terminal_panel(left_panel)
        self._build_button_panel(left_panel)

        right_panel = tk.Frame(content_frame, bg=self.COLORS["bg_secondary"])
        right_panel.pack(side="right", fill="y", padx=(5, 0))

        self._build_camera_panel(right_panel)
        self._configure_tags()

        self.root.after(30, self._poll_frames)
        self.root.after(50, self._poll_logs)

        self.root.bind("<Escape>", lambda e: self._close())
        self.root.protocol("WM_DELETE_WINDOW", self._close)

    def _build_title_bar(self, parent: tk.Widget) -> None:
        title_bar = tk.Frame(parent, bg=self.COLORS["bg_tertiary"])
        title_bar.pack(fill="x", pady=(0, 5))

        tk.Label(
            title_bar,
            text="UR3 ROBOT CONTROLLER",
            font=("Consolas", 14, "bold"),
            bg=self.COLORS["bg_tertiary"],
            fg=self.COLORS["accent"],
        ).pack(side="left", padx=15, pady=5)

        tk.Label(
            title_bar,
            text="Powered by Gemini Robotics-ER",
            font=("Consolas", 9),
            bg=self.COLORS["bg_tertiary"],
            fg=self.COLORS["text_muted"],
        ).pack(side="right", padx=15, pady=5)

    def _build_camera_panel(self, parent: tk.Widget) -> None:
        cam_title = tk.Frame(parent, bg=self.COLORS["bg_tertiary"])
        cam_title.pack(fill="x")

        tk.Label(
            cam_title,
            text="CAMERAS",
            font=("Consolas", 10, "bold"),
            bg=self.COLORS["bg_tertiary"],
            fg=self.COLORS["accent_cyan"],
        ).pack(pady=4)

        cams_container = tk.Frame(parent, bg=self.COLORS["bg_secondary"])
        cams_container.pack(fill="both", expand=True, padx=3, pady=3)

        cam1_frame = tk.Frame(cams_container, bg=self.COLORS["border"], bd=1)
        cam1_frame.pack(pady=(0, 3))

        self.cam1_label = tk.Label(
            cam1_frame,
            bg="black",
            width=settings.display_width,
            height=settings.display_height,
        )
        self.cam1_label.pack()

        cam2_frame = tk.Frame(cams_container, bg=self.COLORS["border"], bd=1)
        cam2_frame.pack(pady=(3, 0))

        self.cam2_label = tk.Label(
            cam2_frame,
            bg="black",
            width=settings.display_width,
            height=settings.display_height,
        )
        self.cam2_label.pack()

    def _build_input_panel(self, parent: tk.Widget) -> None:
        input_frame = tk.Frame(parent, bg=self.COLORS["bg_secondary"])
        input_frame.pack(fill="x", pady=(0, 5))

        input_header = tk.Frame(input_frame, bg=self.COLORS["bg_tertiary"])
        input_header.pack(fill="x")

        tk.Label(
            input_header,
            text="COMANDO",
            font=("Consolas", 10, "bold"),
            bg=self.COLORS["bg_tertiary"],
            fg=self.COLORS["accent_green"],
        ).pack(side="left", padx=10, pady=4)

        entry_container = tk.Frame(input_frame, bg=self.COLORS["bg_secondary"])
        entry_container.pack(fill="x", padx=5, pady=8)

        tk.Label(
            entry_container,
            text=">",
            font=("Consolas", 16, "bold"),
            bg=self.COLORS["bg_secondary"],
            fg=self.COLORS["accent_green"],
        ).pack(side="left", padx=(5, 8))

        self.entry = tk.Entry(
            entry_container,
            font=("Consolas", 13),
            bg=self.COLORS["bg"],
            fg=self.COLORS["text"],
            insertbackground=self.COLORS["accent"],
            relief="flat",
            highlightthickness=2,
            highlightcolor=self.COLORS["accent"],
            highlightbackground=self.COLORS["border"],
        )
        self.entry.pack(side="left", fill="x", expand=True, padx=5, ipady=10)

        self.entry.bind("<Return>", self._on_enter)
        self.entry.bind("<Up>", lambda e: self._navigate_history(-1))
        self.entry.bind("<Down>", lambda e: self._navigate_history(1))
        self.entry.focus_set()

        tk.Button(
            entry_container,
            text="RESET",
            font=("Consolas", 11, "bold"),
            bg=self.COLORS["accent_orange"],
            fg=self.COLORS["bg"],
            relief="flat",
            cursor="hand2",
            command=self._send_reset,
        ).pack(side="right", padx=(0, 6), ipadx=12, ipady=8)

        self.mic_button = tk.Button(
            entry_container,
            text="MIC",
            font=("Consolas", 11, "bold"),
            bg=self.COLORS["accent_cyan"],
            fg=self.COLORS["bg"],
            relief="flat",
            cursor="hand2",
            command=self._toggle_mic,
        )
        self.mic_button.pack(side="right", padx=(0, 6), ipadx=15, ipady=8)

        tk.Button(
            entry_container,
            text="ENVIAR",
            font=("Consolas", 11, "bold"),
            bg=self.COLORS["accent_green"],
            fg=self.COLORS["bg"],
            relief="flat",
            cursor="hand2",
            command=self._on_enter,
        ).pack(side="right", padx=10, ipadx=15, ipady=8)

    def _build_terminal_panel(self, parent: tk.Widget) -> None:
        terminal_frame = tk.Frame(parent, bg=self.COLORS["bg_secondary"])
        terminal_frame.pack(fill="both", expand=True)

        header = tk.Frame(terminal_frame, bg=self.COLORS["bg_tertiary"])
        header.pack(fill="x")

        tk.Label(
            header,
            text="TERMINAL",
            font=("Consolas", 10, "bold"),
            bg=self.COLORS["bg_tertiary"],
            fg=self.COLORS["text"],
        ).pack(side="left", padx=10, pady=4)

        tk.Button(
            header,
            text="Limpar",
            font=("Consolas", 8),
            bg=self.COLORS["bg_tertiary"],
            fg=self.COLORS["text_muted"],
            relief="flat",
            cursor="hand2",
            command=self._clear_terminal,
        ).pack(side="right", padx=8, pady=2)

        text_frame = tk.Frame(terminal_frame, bg=self.COLORS["bg"])
        text_frame.pack(fill="both", expand=True, padx=2, pady=2)

        self.terminal = tk.Text(
            text_frame,
            font=("Consolas", 10),
            bg=self.COLORS["bg"],
            fg=self.COLORS["text"],
            insertbackground=self.COLORS["accent"],
            relief="flat",
            wrap="word",
            state="disabled",
            padx=10,
            pady=8,
        )

        scrollbar = tk.Scrollbar(text_frame, command=self.terminal.yview)
        self.terminal.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        self.terminal.pack(side="left", fill="both", expand=True)

    def _build_button_panel(self, parent: tk.Widget) -> None:
        button_frame = tk.Frame(parent, bg=self.COLORS["bg_secondary"])
        button_frame.pack(fill="x", pady=(5, 0))

        header = tk.Frame(button_frame, bg=self.COLORS["bg_tertiary"])
        header.pack(fill="x")

        tk.Label(
            header,
            text="ATALHOS RAPIDOS",
            font=("Consolas", 9, "bold"),
            bg=self.COLORS["bg_tertiary"],
            fg=self.COLORS["accent_cyan"],
        ).pack(pady=3)

        buttons_container = tk.Frame(button_frame, bg=self.COLORS["bg_secondary"])
        buttons_container.pack(fill="x", padx=2, pady=3)

        category_colors = {
            "VISAO": self.COLORS["accent_cyan"],
            "MOVIMENTO": self.COLORS["accent"],
            "JUNTAS": self.COLORS["accent_orange"],
            "GARRA": self.COLORS["accent_purple"],
            "MANIPULACAO": self.COLORS["accent_green"],
            "INFO": self.COLORS["accent_red"],
        }

        col = 0
        for category, buttons in self.SHORTCUTS.items():
            cat_frame = tk.Frame(buttons_container, bg=self.COLORS["bg_secondary"])
            cat_frame.grid(row=0, column=col, padx=3, sticky="n")

            tk.Label(
                cat_frame,
                text=category,
                font=("Consolas", 8, "bold"),
                bg=self.COLORS["bg_secondary"],
                fg=category_colors.get(category, self.COLORS["text"]),
            ).pack(anchor="w", pady=(2, 3))

            for label, cmd in buttons:
                btn = tk.Button(
                    cat_frame,
                    text=label,
                    font=("Consolas", 8),
                    bg=self.COLORS["bg_tertiary"],
                    fg=self.COLORS["text"],
                    activebackground=category_colors.get(category, self.COLORS["accent"]),
                    activeforeground=self.COLORS["bg"],
                    relief="flat",
                    cursor="hand2",
                    width=12,
                    command=lambda c=cmd: self._send_command(c),
                )
                btn.pack(fill="x", pady=1)

                color = category_colors.get(category, self.COLORS["accent"])
                btn.bind("<Enter>", lambda e, b=btn, c=color: b.configure(bg=c, fg=self.COLORS["bg"]))
                btn.bind(
                    "<Leave>",
                    lambda e, b=btn: b.configure(bg=self.COLORS["bg_tertiary"], fg=self.COLORS["text"]),
                )

            col += 1

        for i in range(col):
            buttons_container.columnconfigure(i, weight=1)

    def _configure_tags(self) -> None:
        tags = {
            "user": self.COLORS["accent_green"],
            "success": self.COLORS["accent_green"],
            "error": self.COLORS["accent_red"],
            "warning": self.COLORS["accent_orange"],
            "info": self.COLORS["text"],
            "robot": self.COLORS["accent"],
            "vision": self.COLORS["accent_cyan"],
            "system": self.COLORS["accent_purple"],
            "action": self.COLORS["accent_orange"],
            "timing": self.COLORS["accent_yellow"],
            "timestamp": self.COLORS["text_muted"],
        }

        for tag, color in tags.items():
            self.terminal.tag_configure(tag, foreground=color)

        self.terminal.tag_configure("bold", font=("Consolas", 10, "bold"))

    def _log(self, message: str, tag: str = "info") -> None:
        self.terminal.configure(state="normal")

        timestamp = time.strftime("%H:%M:%S")
        prefixes = {
            "user": "USR",
            "robot": "BOT",
            "vision": "VIS",
            "success": "OK",
            "error": "ERR",
            "warning": "WRN",
            "action": "ACT",
            "system": "SYS",
            "info": "INF",
            "timing": "TMR",
        }
        prefix = prefixes.get(tag, "INF")

        if tag != "system":
            self.terminal.insert("end", f"[{timestamp}] ", "timestamp")

        self.terminal.insert("end", f"[{prefix}] ", (tag, "bold"))
        self.terminal.insert("end", f"{message}\n", tag)
        self.terminal.see("end")
        self.terminal.configure(state="disabled")

    def _clear_terminal(self) -> None:
        self.terminal.configure(state="normal")
        self.terminal.delete("1.0", "end")
        self.terminal.configure(state="disabled")

    def _send_command(self, cmd: str) -> None:
        if not cmd.strip():
            return

        clean = cmd.strip()
        self._log(clean, "user")
        self.cmd_queue.put(clean)

        if clean not in self.history or (self.history and self.history[-1] != clean):
            self.history.append(clean)
        self.history_idx = len(self.history)

    def _ensure_stt(self) -> LocalSTT:
        with self._stt_lock:
            if self._stt is None:
                self._stt = LocalSTT(
                    model_size=settings.stt_model_size,
                    device=settings.stt_device,
                    compute_type=settings.stt_compute_type,
                    beam_size=settings.stt_beam_size,
                    vad_filter=settings.stt_vad_filter,
                )
        return self._stt

    def _preload_stt(self) -> None:
        try:
            self._ensure_stt()
            self.root.after(0, lambda: self._log("STT local pronto.", "system"))
        except Exception as exc:
            msg = f"STT indisponivel: {exc}"
            self.root.after(0, lambda m=msg: self._log(m, "warning"))

    def _refresh_mic_button(self) -> None:
        if self._mic_state == "recording":
            self.mic_button.configure(text="PARAR", bg=self.COLORS["accent_red"], state="normal")
            return
        if self._mic_state == "processing":
            self.mic_button.configure(text="...", bg=self.COLORS["bg_tertiary"], state="disabled")
            return
        self.mic_button.configure(text="MIC", bg=self.COLORS["accent_cyan"], state="normal")

    def _toggle_mic(self) -> None:
        if self._mic_state == "recording":
            self._mic_recording = False
            self._mic_state = "processing"
            self._refresh_mic_button()
            self._log("Processando audio...", "system")
            return

        if self._mic_state != "idle":
            return

        self._mic_state = "recording"
        self._mic_recording = True
        self._mic_chunks = []
        self._refresh_mic_button()
        self._log("Gravacao iniciada. Clique PARAR para transcrever.", "system")

        self._mic_worker = threading.Thread(target=self._record_and_transcribe, daemon=True)
        self._mic_worker.start()

    def _record_and_transcribe(self) -> None:
        try:
            try:
                import sounddevice as sd
            except ImportError as exc:
                raise RuntimeError("Dependencia ausente para microfone. Instale: pip install '.[stt]'") from exc

            def on_audio(indata: np.ndarray, frames: int, stream_time: object, status: object) -> None:
                del frames, stream_time
                if status:
                    return
                with self._mic_lock:
                    self._mic_chunks.append(indata.copy())

            with sd.InputStream(
                samplerate=settings.mic_sample_rate,
                channels=settings.mic_channels,
                dtype="float32",
                callback=on_audio,
            ):
                while self.running[0] and self._mic_recording:
                    sd.sleep(100)

            with self._mic_lock:
                chunks = list(self._mic_chunks)
                self._mic_chunks = []

            if not chunks:
                self.root.after(0, lambda: self._log("Nenhum audio capturado.", "warning"))
                return

            audio = np.concatenate(chunks, axis=0)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio = audio.astype(np.float32, copy=False)
            if audio.size < settings.mic_sample_rate // 5:
                self.root.after(0, lambda: self._log("Audio muito curto para transcricao.", "warning"))
                return

            transcribed = self._ensure_stt().transcribe_array(
                audio=audio,
                sample_rate=settings.mic_sample_rate,
                language=settings.stt_language,
            )
            if not transcribed:
                self.root.after(0, lambda: self._log("Nao foi possivel transcrever o audio.", "warning"))
                return

            self.root.after(0, lambda txt=transcribed: self._on_voice_transcribed(txt))
        except Exception as exc:
            msg = f"Erro no microfone/STT: {exc}"
            self.root.after(0, lambda m=msg: self._log(m, "error"))
        finally:
            self.root.after(0, self._set_mic_idle)

    def _on_voice_transcribed(self, text: str) -> None:
        self._log(f"Transcricao: {text}", "system")
        self._send_command(text)

    def _set_mic_idle(self) -> None:
        self._mic_recording = False
        self._mic_state = "idle"
        self._refresh_mic_button()

    def _send_reset(self) -> None:
        self._log("Reset da simulacao solicitado", "system")
        self.cmd_queue.put(RESET_SIM_CMD)

    def _on_enter(self, event: object | None = None) -> None:
        cmd = self.entry.get().strip()
        self.entry.delete(0, "end")

        if cmd.lower() in ["sair", "exit", "quit"]:
            self._close()
        elif cmd:
            self._send_command(cmd)

    def _navigate_history(self, delta: int) -> str:
        if not self.history:
            return "break"

        self.history_idx = max(0, min(len(self.history), self.history_idx + delta))
        self.entry.delete(0, "end")
        if self.history_idx < len(self.history):
            self.entry.insert(0, self.history[self.history_idx])
        return "break"

    def _poll_frames(self) -> None:
        if not self.running[0]:
            self.root.quit()
            return

        try:
            frame1, frame2 = self.frame_queue.get_nowait()

            if frame1 is not None:
                frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
                frame1_resized = cv2.resize(frame1_rgb, (settings.display_width, settings.display_height))
                img1 = Image.fromarray(frame1_resized)
                self.photo1 = ImageTk.PhotoImage(img1)
                self.cam1_label.configure(image=self.photo1)

            if frame2 is not None:
                frame2_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                frame2_resized = cv2.resize(frame2_rgb, (settings.display_width, settings.display_height))
                img2 = Image.fromarray(frame2_resized)
                self.photo2 = ImageTk.PhotoImage(img2)
                self.cam2_label.configure(image=self.photo2)

        except Empty:
            pass

        self.root.after(30, self._poll_frames)

    def _poll_logs(self) -> None:
        if not self.running[0]:
            return

        try:
            for _ in range(20):
                message, tag = self.log_queue.get_nowait()
                self._log(message, tag)
        except Empty:
            pass

        self.root.after(50, self._poll_logs)

    def _close(self) -> None:
        self.running[0] = False
        self._mic_recording = False
        self.root.quit()

    def run(self) -> None:
        self.root.mainloop()
        self.root.destroy()
