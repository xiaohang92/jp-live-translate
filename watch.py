#!/usr/bin/env python3
"""Watch VLC/QuickTime. If that video already has stored captions, show them. Otherwise do nothing."""

from __future__ import annotations

import argparse
import signal
import sys

from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication, QMessageBox

from cues_store import cue_at, load_cues_auto
from engine import normalize_target
from overlay import SubtitleOverlay
from player_attach import pause_player, poll_open_player, resume_player


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Leave this running, then open any video in VLC or QuickTime. "
            "If prepare.py already saved captions for that video (or a same-size copy) "
            "into this project's cues/ folder, the overlay appears. If not, this script stays silent."
        )
    )
    parser.add_argument(
        "--to",
        dest="target",
        default="auto",
        help="zh, en, or auto (use whichever sidecar exists; default auto).",
    )
    parser.add_argument(
        "--ask",
        action="store_true",
        help="When captions exist, pause the player and ask before showing them.",
    )
    parser.add_argument("--font-size", type=int, default=28)
    parser.add_argument("--drag", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    preferred = args.target.strip().lower()
    if preferred != "auto":
        try:
            preferred = normalize_target(preferred)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    app = QApplication(sys.argv)
    app.setFont(QFont(".AppleSystemUIFont" if sys.platform == "darwin" else "Sans Serif"))
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    signal.signal(signal.SIGTERM, lambda *_: app.quit())
    overlay = SubtitleOverlay(
        args.font_size,
        click_through=not args.drag,
    )
    overlay.set_follow_pid(0)

    state = {"path": None, "cues": None, "denied": set(), "announced": None}

    def tick() -> None:
        try:
            _tick()
        except KeyboardInterrupt:
            app.quit()

    def _tick() -> None:
        player = poll_open_player()
        if player is None:
            overlay.set_follow_pid(0)
            overlay._clear_subtitle()
            state["path"] = None
            state["cues"] = None
            return
        media = player.path
        if media != state["path"]:
            cues, _lang = load_cues_auto(media, preferred)
            state["path"] = media
            state["cues"] = cues
            if cues and args.ask and media not in state["denied"]:
                pause_player(player)
                reply = QMessageBox.question(
                    None,
                    "JP Live Translate",
                    f"Subtitles are available for:\n{media}\n\nShow them?",
                )
                resume_player(player)
                if reply != QMessageBox.StandardButton.Yes:
                    state["denied"].add(media)
                    state["cues"] = None
            if state["cues"] and state["announced"] != media:
                print(f"Subtitles on  {media}  ({len(state['cues'])} lines)", flush=True)
                state["announced"] = media
        if not state["cues"]:
            overlay.set_follow_pid(0)
            overlay._clear_subtitle()
            return
        overlay.set_follow_pid(player.pid)
        cue = cue_at(state["cues"], player.time_sec)
        if cue is None:
            overlay._clear_subtitle()
            return
        _t0, _t1, japanese, translated = cue
        overlay.show_subtitle(japanese, translated, 8_000)

    timer = QTimer()
    timer.timeout.connect(tick)
    timer.start(200)
    print(
        "Watching VLC and QuickTime. Open a video yourself.\n"
        "Captions show if that video was prepared into cues/ (same-size copies count).\n"
        "Ctrl+C or Esc to quit.",
        flush=True,
    )
    return app.exec()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(0) from None
