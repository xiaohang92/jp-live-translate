"""Detect the video currently open in VLC or QuickTime (macOS)."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass


@dataclass
class OpenPlayer:
    app: str
    pid: int
    path: str
    time_sec: float


def poll_open_player() -> OpenPlayer | None:
    return _poll_vlc() or _poll_quicktime()


def _pgrep_pid(name: str) -> int | None:
    try:
        out = subprocess.check_output(
            ["pgrep", "-x", name],
            text=True,
            timeout=0.3,
            start_new_session=True,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired, KeyboardInterrupt):
        return None
    for line in out.splitlines():
        line = line.strip()
        if line.isdigit():
            return int(line)
    return None


def _osascript(source: str) -> str | None:
    try:
        result = subprocess.run(
            ["osascript", "-e", source],
            capture_output=True,
            text=True,
            timeout=0.8,
            start_new_session=True,
        )
    except (OSError, subprocess.TimeoutExpired, KeyboardInterrupt):
        return None
    if result.returncode != 0:
        return None
    text = result.stdout.strip()
    return text or None


def _poll_vlc() -> OpenPlayer | None:
    pid = _pgrep_pid("VLC")
    if pid is None:
        return None
    path = _osascript('tell application "VLC" to get path of current item')
    raw_t = _osascript('tell application "VLC" to get current time')
    if not path or raw_t is None:
        return None
    try:
        time_sec = float(raw_t)
    except ValueError:
        return None
    if not path.startswith("/"):
        return None
    return OpenPlayer(app="VLC", pid=pid, path=path, time_sec=time_sec)


def _poll_quicktime() -> OpenPlayer | None:
    pid = _pgrep_pid("QuickTime Player")
    if pid is None:
        return None
    raw = _osascript(
        'tell application "QuickTime Player"\n'
        "if (count of documents) is 0 then return \"\"\n"
        "set d to front document\n"
        "return (POSIX path of (file of d as alias)) & tab & (current time of d as text)\n"
        "end tell"
    )
    if not raw or "\t" not in raw:
        return None
    path, _, clock = raw.partition("\t")
    path = path.strip()
    try:
        time_sec = float(clock.strip())
    except ValueError:
        return None
    if not path.startswith("/"):
        return None
    return OpenPlayer(app="QuickTime", pid=pid, path=path, time_sec=time_sec)


def pause_player(player: OpenPlayer) -> None:
    if player.app == "VLC":
        _osascript(
            'tell application "VLC"\n'
            "if playing then play\n"
            "end tell"
        )
        return
    _osascript('tell application "QuickTime Player" to pause front document')


def resume_player(player: OpenPlayer) -> None:
    if player.app == "VLC":
        _osascript(
            'tell application "VLC"\n'
            "if not playing then play\n"
            "end tell"
        )
        return
    _osascript('tell application "QuickTime Player" to play front document')
