#!/usr/bin/env python3
"""Transparent always-on-top subtitle overlay for live Japanese → English translation."""

from __future__ import annotations

import argparse
import sys

from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QGuiApplication, QMouseEvent
from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

from translate import (
    BUFFER_SECONDS,
    SAMPLE_RATE,
    find_capture_device,
    is_silent,
    list_devices,
    load_model,
    to_mono_f32,
    translate_audio,
)

SUBTITLE_HOLD_MS = 8_000


class TranslatorWorker(QThread):
    text_ready = pyqtSignal(str)
    status_ready = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(
        self,
        model_name: str,
        device_name: str | None,
        use_mic: bool,
        buffer_seconds: float,
    ) -> None:
        super().__init__()
        self._model_name = model_name
        self._device_name = device_name
        self._use_mic = use_mic
        self._buffer_seconds = buffer_seconds
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        try:
            self.status_ready.emit("Finding audio device…")
            mic = find_capture_device(self._device_name, self._use_mic)
            self.status_ready.emit(f"Loading model ({self._model_name})…")
            model = load_model(self._model_name)
            frames = int(self._buffer_seconds * SAMPLE_RATE)
            self.status_ready.emit(f"Listening · {mic.name}")

            with mic.recorder(samplerate=SAMPLE_RATE, channels=1) as recorder:
                while not self._stop:
                    audio = to_mono_f32(recorder.record(numframes=frames))
                    if self._stop or is_silent(audio):
                        continue
                    text = translate_audio(model, audio)
                    if text:
                        self.text_ready.emit(text)
        except Exception as exc:
            self.failed.emit(str(exc))


class SubtitleOverlay(QWidget):
    def __init__(self, font_size: int, click_through: bool) -> None:
        super().__init__()
        self._drag_offset = None
        flags = (
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        if click_through:
            flags |= Qt.WindowType.WindowTransparentForInput
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setWindowTitle("JP Live Translate")

        self.status = QLabel("Starting…")
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status.setStyleSheet(
            "color: rgba(255,255,255,180); background: transparent; "
            "font-size: 13px; padding: 0 8px;"
        )

        self.subtitle = QLabel("Waiting for Japanese audio…")
        self.subtitle.setWordWrap(True)
        self.subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.subtitle.setStyleSheet(
            f"""
            QLabel {{
                color: #fff;
                background-color: rgba(0, 0, 0, 165);
                border-radius: 14px;
                padding: 14px 22px;
                font-size: {font_size}px;
                font-weight: 600;
            }}
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 8, 16, 16)
        layout.setSpacing(8)
        layout.addWidget(self.status)
        layout.addWidget(self.subtitle)

        self._clear_timer = QTimer(self)
        self._clear_timer.setSingleShot(True)
        self._clear_timer.timeout.connect(self._clear_subtitle)

        screen = QGuiApplication.primaryScreen().availableGeometry()
        width = min(900, int(screen.width() * 0.7))
        self.resize(width, 140)
        self.move(
            screen.x() + (screen.width() - width) // 2,
            screen.y() + screen.height() - 180,
        )

    def show_status(self, text: str) -> None:
        self.status.setText(text)

    def show_subtitle(self, text: str) -> None:
        self.subtitle.setText(text)
        self.adjustSize()
        self._clear_timer.start(SUBTITLE_HOLD_MS)

    def show_error(self, text: str) -> None:
        self.status.setText("Error")
        self.subtitle.setText(text)
        self.adjustSize()
        self._clear_timer.stop()

    def _clear_subtitle(self) -> None:
        self.subtitle.setText("")
        self.adjustSize()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = (
                event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            )

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_offset is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_offset)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = None

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            QApplication.quit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transparent overlay for local Japanese → English translation."
    )
    parser.add_argument(
        "--model",
        default="small",
        choices=("small", "medium"),
        help="faster-whisper model size (default: small).",
    )
    parser.add_argument(
        "--device",
        help="Capture device name substring, e.g. BlackHole.",
    )
    parser.add_argument(
        "--mic",
        action="store_true",
        help="Use the default microphone instead of system-audio loopback.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Print capture devices and exit.",
    )
    parser.add_argument(
        "--buffer",
        type=float,
        default=BUFFER_SECONDS,
        help="Audio buffer length in seconds (default: 3).",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=28,
        help="Subtitle font size in pixels (default: 28).",
    )
    parser.add_argument(
        "--click-through",
        action="store_true",
        help="Let mouse clicks pass through the overlay (not draggable).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list_devices:
        list_devices()
        return 0
    if args.buffer <= 0:
        raise SystemExit("--buffer must be positive")

    app = QApplication(sys.argv)
    app.setFont(QFont(".AppleSystemUIFont" if sys.platform == "darwin" else "Sans Serif"))
    overlay = SubtitleOverlay(args.font_size, args.click_through)
    overlay.show()

    worker = TranslatorWorker(args.model, args.device, args.mic, args.buffer)
    worker.text_ready.connect(overlay.show_subtitle)
    worker.status_ready.connect(overlay.show_status)
    worker.failed.connect(overlay.show_error)
    app.aboutToQuit.connect(worker.stop)
    worker.start()

    code = app.exec()
    worker.stop()
    worker.wait(4_000)
    return code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(0)
