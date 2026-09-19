#!/usr/bin/env python3
"""Transparent always-on-top subtitle overlay for live Japanese → English translation."""

from __future__ import annotations

import argparse
import base64
import ctypes
import ctypes.util
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QGuiApplication, QMouseEvent, QPainter
from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

from cues_store import cue_at, load_cues, sidecar_path
from translate import find_capture_device, list_devices
from engine import (
    BUFFER_SECONDS,
    HOP_SECONDS,
    SAMPLE_RATE,
    iter_live_utterances,
    load_pipeline,
    normalize_target,
    to_mono_f32,
)

SUBTITLE_HOLD_MS = 6_000


def _macos_ignores_mouse(widget: QWidget, ignore: bool) -> None:
    """Qt's WindowTransparentForInput does not reliably pass clicks on macOS."""
    if sys.platform != "darwin":
        return
    try:
        lib = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
        sel_registerName = lib.sel_registerName
        sel_registerName.restype = ctypes.c_void_p
        sel_registerName.argtypes = [ctypes.c_char_p]
        msg = lib.objc_msgSend
        nsview = ctypes.c_void_p(int(widget.winId()))
        msg.restype = ctypes.c_void_p
        msg.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        nswindow = msg(nsview, sel_registerName(b"window"))
        if not nswindow:
            return
        msg.restype = None
        msg.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool]
        msg(
            ctypes.c_void_p(nswindow),
            sel_registerName(b"setIgnoresMouseEvents:"),
            bool(ignore),
        )
    except (OSError, ValueError, AttributeError, TypeError):
        return


class TranslatorWorker(QThread):
    text_ready = pyqtSignal(str, str, int)
    status_ready = pyqtSignal(str)
    failed = pyqtSignal(str)
    finished_ok = pyqtSignal()

    def __init__(
        self,
        model_name: str,
        device_name: str | None,
        use_mic: bool,
        buffer_seconds: float,
        hop_seconds: float = HOP_SECONDS,
        backend: str = "auto",
        mt: str = "cascade",
        target: str = "en",
    ) -> None:
        super().__init__()
        self._model_name = model_name
        self._backend = backend
        self._mt = mt
        self._target = target
        self._device_name = device_name
        self._use_mic = use_mic
        self._buffer_seconds = buffer_seconds
        self._hop_seconds = hop_seconds
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        try:
            self._run_live()
        except KeyboardInterrupt:
            return
        except Exception as exc:
            self.failed.emit(str(exc))

    def _run_live(self) -> None:
        self.status_ready.emit("Finding audio device…")
        mic = find_capture_device(self._device_name, self._use_mic)
        self.status_ready.emit("Loading models…")
        pipeline = load_pipeline(
            self._model_name,
            backend=self._backend,
            mt=self._mt,
            target=self._target,
        )
        self.status_ready.emit("")

        last_pair = ("", "")
        with mic.recorder(samplerate=SAMPLE_RATE, channels=1) as recorder:
            def record_hop(frames: int):
                return to_mono_f32(recorder.record(numframes=frames))

            for utterance in iter_live_utterances(
                record_hop,
                self._hop_seconds,
                self._buffer_seconds,
                lambda: self._stop,
            ):
                if self._stop:
                    break
                japanese, english = pipeline.run(utterance)
                pair = (japanese, english)
                if pair != last_pair and (japanese or english):
                    last_pair = pair
                    self.text_ready.emit(japanese, english, SUBTITLE_HOLD_MS)


def _vlc_bin() -> str:
    app = Path("/Applications/VLC.app/Contents/MacOS/VLC")
    if app.is_file():
        return str(app)
    found = shutil.which("vlc")
    if found:
        return found
    raise RuntimeError(
        "VLC.app not found in /Applications. Install VLC from https://www.videolan.org/"
    )


PLAYER_BUNDLES = frozenset(
    {
        "org.videolan.vlc",
        "com.apple.QuickTimePlayerX",
        "com.apple.quicktimeplayer",
    }
)


def _frontmost_app() -> tuple[int, str] | None:
    """Frontmost macOS app (pid, bundle id). None if unknown."""
    if sys.platform != "darwin":
        return None
    try:
        asn = subprocess.check_output(
            ["lsappinfo", "front"], text=True, timeout=0.3
        ).strip()
        info = subprocess.check_output(
            ["lsappinfo", "info", "-only", "bundleid,pid", asn],
            text=True,
            timeout=0.3,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    pid = None
    bundle = ""
    for line in info.splitlines():
        match = re.search(r'pid"?\s*=\s*(\d+)', line, re.I)
        if match:
            pid = int(match.group(1))
        match = re.search(r'=\s*"([^"]+)"', line)
        if match and "bundle" in line.lower():
            bundle = match.group(1)
    if pid is None:
        return None
    return pid, bundle


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class FileDemoPlayer:
    """Play in the real VLC.app window; poll its HTTP status for caption time."""

    def __init__(self, overlay: SubtitleOverlay, path: str, cues: list, on_done) -> None:
        self.overlay = overlay
        self.cues = cues
        self._on_done = on_done
        self._shown = None
        self._ended = False
        self._started = False
        self._proc: subprocess.Popen | None = None
        self._http_password = "jplt"
        self._http_port = _free_port()
        self._status_url = f"http://127.0.0.1:{self._http_port}/requests/status.xml"
        self._path = path
        self._tick = QTimer()
        self._tick.setInterval(80)
        self._tick.timeout.connect(self._poll)

    def start(self) -> None:
        cmd = [
            _vlc_bin(),
            "--intf",
            "macosx" if sys.platform == "darwin" else "qt",
            "--extraintf",
            "http",
            "--http-host",
            "127.0.0.1",
            "--http-port",
            str(self._http_port),
            "--http-password",
            self._http_password,
            "--play-and-exit",
            "--no-video-title-show",
            self._path,
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if not self._wait_http():
            self.stop()
            raise RuntimeError(
                "VLC started but its control interface did not come up. "
                "Quit any extra VLC windows and try again."
            )
        if self._proc is not None:
            self.overlay.set_follow_pid(self._proc.pid)
        self._tick.start()

    def _wait_http(self) -> bool:
        deadline = time.time() + 8
        while time.time() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                return False
            if self._vlc_status() is not None:
                return True
            time.sleep(0.1)
        return False

    def _vlc_status(self) -> tuple[str, float] | None:
        req = urllib.request.Request(self._status_url)
        token = base64.b64encode(f":{self._http_password}".encode()).decode()
        req.add_header("Authorization", f"Basic {token}")
        try:
            with urllib.request.urlopen(req, timeout=0.4) as resp:
                body = resp.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, TimeoutError, OSError):
            return None
        state_m = re.search(r"<state>([^<]+)</state>", body)
        time_m = re.search(r"<time>([^<]+)</time>", body)
        if not state_m:
            return None
        seconds = float(time_m.group(1)) if time_m else 0.0
        return state_m.group(1).strip(), seconds

    def stop(self) -> None:
        self.overlay.set_follow_pid(None)
        self._tick.stop()
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None

    def _finish(self) -> None:
        if self._ended:
            return
        self._ended = True
        self._tick.stop()
        self.overlay._clear_subtitle()
        self.stop()
        self._on_done()

    def _poll(self) -> None:
        if self._proc is not None and self._proc.poll() is not None:
            self._finish()
            return
        status = self._vlc_status()
        if status is None:
            return
        state, t = status
        if state == "playing":
            self._started = True
        if self._started and state in {"stopped", "ended"}:
            self._finish()
            return
        if not self._started:
            return
        cue = cue_at(self.cues, t)
        if cue == self._shown:
            return
        self._shown = cue
        if cue is None:
            self.overlay._clear_subtitle()
            return
        _t0, t1, japanese, english = cue
        self.overlay.show_subtitle(japanese, english, 60_000)

    def _cue_at(self, t: float):
        return cue_at(self.cues, t)


class OutlinedLabel(QLabel):
    """Caption text with a dark outline so it stays readable without a blocking card."""

    def __init__(self, color: str, size: int, bold: bool = True) -> None:
        super().__init__("")
        self._fill = QColor(color)
        self._outline = QColor(0, 0, 0, 235)
        self.setWordWrap(True)
        self.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        font = QFont(
            ".AppleSystemUIFont" if sys.platform == "darwin" else "Helvetica Neue",
            size,
        )
        font.setBold(bold)
        self.setFont(font)
        self.setStyleSheet("background: transparent; color: transparent;")

    def paintEvent(self, event) -> None:  # noqa: ANN001
        text = self.text()
        if not text:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        flags = int(self.alignment() | Qt.TextFlag.TextWordWrap)
        rect = self.rect()
        width = 2
        painter.setPen(self._outline)
        for dx in (-width, 0, width):
            for dy in (-width, 0, width):
                if dx or dy:
                    painter.drawText(rect.translated(dx, dy), flags, text)
        painter.setPen(self._fill)
        painter.drawText(rect, flags, text)


class SubtitleOverlay(QWidget):
    def __init__(
        self,
        font_size: int,
        click_through: bool,
        screenshot_dir: str | None = None,
    ) -> None:
        super().__init__()
        self._drag_offset = None
        self._click_through = click_through
        self._screenshot_dir = Path(screenshot_dir) if screenshot_dir else None
        self._shot_index = 0
        if self._screenshot_dir:
            self._screenshot_dir.mkdir(parents=True, exist_ok=True)
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
        self.setAttribute(Qt.WidgetAttribute.WA_MacAlwaysShowToolWindow, True)
        self.setWindowTitle("JP Live Translate")

        ja_size = max(font_size - 4, 16)
        self.japanese = OutlinedLabel("#ffe082", ja_size, bold=False)
        self.english = OutlinedLabel("#ffffff", font_size, bold=True)
        self.japanese.hide()
        self.english.hide()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 8, 24, 12)
        layout.setSpacing(2)
        layout.addWidget(self.japanese)
        layout.addWidget(self.english)

        self._clear_timer = QTimer(self)
        self._clear_timer.setSingleShot(True)
        self._clear_timer.timeout.connect(self._clear_subtitle)

        self._lift_timer = QTimer(self)
        self._lift_timer.timeout.connect(self._lift)
        self._lift_timer.start(400)
        self._follow_pid: int | None = None

        self._reposition()

    def set_follow_pid(self, pid: int | None) -> None:
        """None = always on top (live). 0 = hidden watcher idle. >0 = follow that player."""
        self._follow_pid = pid
        self._lift()

    def _has_caption(self) -> bool:
        return bool(self.japanese.text().strip() or self.english.text().strip())

    def _apply_click_through(self) -> None:
        if self._click_through:
            _macos_ignores_mouse(self, True)

    def _player_is_focused(self) -> bool:
        if self._follow_pid is None:
            return True
        if self._follow_pid <= 0:
            return False
        front = _frontmost_app()
        if front is None:
            return True
        pid, bundle = front
        if pid in {self._follow_pid, os.getpid()}:
            return True
        return bundle in PLAYER_BUNDLES

    def show_status(self, text: str) -> None:
        if text.lower().startswith("error"):
            self.show_error(text)

    def show_subtitle(self, japanese: str, english: str, hold_ms: int = SUBTITLE_HOLD_MS) -> None:
        if japanese:
            self.japanese.setText(japanese)
            self.japanese.show()
        else:
            self.japanese.clear()
            self.japanese.hide()
        if english:
            self.english.setText(english)
            self.english.show()
        else:
            self.english.clear()
            self.english.hide()
        self._reposition()
        if self._player_is_focused():
            self._lift()
        self._clear_timer.start(max(400, int(hold_ms)))
        self._save_shot()

    def show_error(self, text: str) -> None:
        self.japanese.hide()
        self.english.setText(text)
        self.english.show()
        self._reposition()
        self._clear_timer.stop()
        self._lift()
        self._save_shot()

    def _clear_subtitle(self) -> None:
        self.japanese.clear()
        self.english.clear()
        self.japanese.hide()
        self.english.hide()
        self._reposition()
        self.hide()

    def _lift(self) -> None:
        try:
            self._lift_body()
        except KeyboardInterrupt:
            QApplication.quit()

    def _lift_body(self) -> None:
        if not self._player_is_focused() or not self._has_caption():
            if self.isVisible():
                self.hide()
            return
        appearing = not self.isVisible()
        if appearing:
            self.show()
            self._apply_click_through()
        # Live: do not raise on a timer — that steals clicks from the browser.
        # File/watch: keep above VLC when the player is focused.
        if appearing or (self._follow_pid is not None and self._follow_pid > 0):
            self.raise_()
            handle = self.windowHandle()
            if handle is not None:
                handle.raise_()
            self._apply_click_through()

    def _reposition(self) -> None:
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return
        geom = screen.availableGeometry()
        max_w = int(geom.width() * 0.7)
        inner_w = max_w - 48
        self.japanese.setMaximumWidth(inner_w)
        self.english.setMaximumWidth(inner_w)
        self.adjustSize()
        hint = self.sizeHint()
        width = min(max_w, max(hint.width(), 80))
        height = max(hint.height(), 40)
        x = geom.x() + (geom.width() - width) // 2
        y = geom.y() + geom.height() - height - 36
        self.setGeometry(x, y, width, height)

    def _save_shot(self) -> None:
        if not self._screenshot_dir:
            return
        path = self._screenshot_dir / f"subtitle-{self._shot_index:02d}.png"
        self._shot_index += 1
        QTimer.singleShot(50, lambda p=path: self.grab().save(str(p)))

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
        description="Transparent overlay for local Japanese → English/Chinese translation."
    )
    parser.add_argument(
        "--model",
        default="small",
        choices=("small", "medium"),
        help="Whisper model size (default: small).",
    )
    parser.add_argument(
        "--backend",
        default="auto",
        choices=("auto", "mlx", "faster-whisper"),
        help="ASR backend (default: auto = MLX on Apple Silicon).",
    )
    parser.add_argument(
        "--mt",
        default="cascade",
        choices=("cascade", "nanosakura", "hymt", "whisper"),
        help="cascade = opus-mt; nanosakura = ja→zh 0.2B; hymt = Hy-MT2 1.8B; whisper = English-only.",
    )
    parser.add_argument(
        "--to",
        dest="target",
        default="en",
        help="Caption language: en (default) or zh (Simplified Chinese).",
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
        help="Max utterance length in seconds (default: 8).",
    )
    parser.add_argument(
        "--hop",
        type=float,
        default=HOP_SECONDS,
        help="Live VAD frame length in seconds (default: 0.5).",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=28,
        help="Subtitle font size in pixels (default: 28).",
    )
    parser.add_argument(
        "--drag",
        action="store_true",
        help="Make the overlay draggable (clicks will not pass through to the video).",
    )
    parser.add_argument(
        "--file",
        dest="audio_file",
        help="Play a local file using captions from prepare.py (does not transcribe).",
    )
    parser.add_argument(
        "--screenshot-dir",
        help="Save overlay snapshots when subtitles update.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list_devices:
        list_devices()
        return 0
    if args.buffer <= 0 or args.hop <= 0:
        raise SystemExit("--buffer and --hop must be positive")
    if args.audio_file and not Path(args.audio_file).is_file():
        raise SystemExit(f"File not found: {args.audio_file}")
    try:
        args.target = normalize_target(args.target)
    except ValueError as exc:
        raise SystemExit(str(exc))

    if args.audio_file:
        media = str(Path(args.audio_file).resolve())
        cues = load_cues(media, args.target)
        if not cues:
            stored = sidecar_path(media, args.target)
            raise SystemExit(
                f"No stored captions for this file ({args.target}).\n"
                f"Expected: {stored}\n"
                f'Prepare first: python prepare.py --file "{args.audio_file}" --to {args.target}'
            )

    app = QApplication(sys.argv)
    app.setFont(QFont(".AppleSystemUIFont" if sys.platform == "darwin" else "Sans Serif"))
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    signal.signal(signal.SIGTERM, lambda *_: app.quit())
    overlay = SubtitleOverlay(
        args.font_size,
        click_through=not args.drag,
        screenshot_dir=args.screenshot_dir,
    )

    demo_player = {"obj": None}
    worker = None

    def shutdown() -> None:
        if worker is not None:
            worker.stop()
        if demo_player["obj"] is not None:
            demo_player["obj"].stop()

    app.aboutToQuit.connect(shutdown)

    if args.audio_file:
        media = str(Path(args.audio_file).resolve())
        cues = load_cues(media, args.target) or []
        print(f"Playing with {len(cues)} stored caption lines", flush=True)
        try:
            player = FileDemoPlayer(
                overlay,
                media,
                cues,
                on_done=QApplication.quit,
            )
        except Exception as exc:
            overlay.show_error(str(exc))
            return app.exec()
        demo_player["obj"] = player
        try:
            player.start()
        except Exception as exc:
            overlay.show_error(str(exc))
            return app.exec()
        code = app.exec()
        shutdown()
        return code

    worker = TranslatorWorker(
        args.model,
        args.device,
        args.mic,
        args.buffer,
        hop_seconds=args.hop,
        backend=args.backend,
        mt=args.mt,
        target=args.target,
    )
    worker.text_ready.connect(overlay.show_subtitle)
    worker.status_ready.connect(overlay.show_status)
    worker.failed.connect(overlay.show_error)
    worker.finished_ok.connect(lambda: QTimer.singleShot(2500, QApplication.quit))
    worker.start()

    code = app.exec()
    shutdown()
    worker.wait(4_000)
    return code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(0) from None
