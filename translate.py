#!/usr/bin/env python3
"""Capture system audio and translate Japanese to English or Chinese (console).

On macOS, true system-audio capture needs a virtual device such as BlackHole.
Use --mic to test with the default microphone.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings

import numpy as np
import soundcard as sc

from engine import (
    BUFFER_SECONDS,
    HOP_SECONDS,
    SAMPLE_RATE,
    iter_live_utterances,
    load_pipeline,
    normalize_target,
    to_mono_f32,
)

# SoundCard cannot tap speakers on macOS; include_loopback only works on Windows.
if sys.platform == "darwin":
    warnings.filterwarnings(
        "ignore",
        message="macOS does not support loopback recording functionality",
        category=Warning,
    )

LOOPBACK_HINTS = (
    "blackhole",
    "loopback",
    "soundflower",
    "vb-audio",
    "cable",
    "stereo mix",
    "what u hear",
    "wave out mix",
)


def capture_microphones():
    """Return capture devices. Loopback inclusion is Windows-only in SoundCard."""
    return sc.all_microphones(include_loopback=(sys.platform == "win32"))


def list_devices() -> None:
    print("Capture devices:")
    mics = capture_microphones()
    if not mics:
        print("  (none found)")
        return
    for i, mic in enumerate(mics):
        print(f"  [{i}] {mic.name}")
    if sys.platform == "darwin":
        print(
            "\nmacOS cannot tap speaker output directly. Install BlackHole "
            "(brew install blackhole-2ch), create a Multi-Output Device that "
            "includes your speakers + BlackHole, then capture from BlackHole."
        )


def find_capture_device(name: str | None, use_mic: bool):
    mics = capture_microphones()
    if not mics:
        raise RuntimeError("No audio capture devices found.")

    if name:
        needle = name.lower()
        matches = [mic for mic in mics if needle in mic.name.lower()]
        if not matches:
            raise RuntimeError(
                f"No capture device matching {name!r}. Run with --list-devices."
            )
        return matches[0]

    if use_mic:
        return sc.default_microphone()

    for mic in mics:
        if any(hint in mic.name.lower() for hint in LOOPBACK_HINTS):
            return mic

    if sys.platform == "win32":
        try:
            speaker = sc.default_speaker()
            loopback = sc.get_microphone(
                id=str(speaker.name), include_loopback=True
            )
            if loopback is not None:
                return loopback
        except Exception:
            pass

    raise RuntimeError(
        "No system-audio loopback device found.\n"
        "macOS: install BlackHole, send output to a Multi-Output Device "
        "(speakers + BlackHole), then rerun with --device BlackHole.\n"
        "To test with your microphone instead: python translate.py --mic\n"
        "To see device names: python translate.py --list-devices"
    )


def load_audio_file(path: str, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Decode any ffmpeg-readable file to mono float32 at Whisper's sample rate."""
    import subprocess

    result = subprocess.run(
        [
            "ffmpeg",
            "-i",
            path,
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-v",
            "error",
            "pipe:1",
        ],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(detail or f"ffmpeg failed to decode {path}")
    audio = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    if audio.size == 0:
        raise RuntimeError(f"No audio decoded from {path}")
    return audio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local Japanese → English/Chinese live translation (console)."
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
        help="Capture device name substring, e.g. BlackHole or 'Stereo Mix'.",
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list_devices:
        list_devices()
        return 0

    if args.buffer <= 0 or args.hop <= 0:
        raise SystemExit("--buffer and --hop must be positive")
    try:
        args.target = normalize_target(args.target)
    except ValueError as exc:
        raise SystemExit(str(exc))

    mic = find_capture_device(args.device, args.mic)
    print("Loading models (first run downloads weights)...")
    pipeline = load_pipeline(
        args.model, backend=args.backend, mt=args.mt, target=args.target
    )
    print(pipeline.label)

    print(f"Capturing from: {mic.name}")
    print(
        f"Pause-cut live captions  hop={args.hop:g}s  max={args.buffer:g}s  "
        f"@{SAMPLE_RATE} Hz"
    )
    print("Ctrl+C to stop.\n")

    last_pair = ("", "")
    with mic.recorder(samplerate=SAMPLE_RATE, channels=1) as recorder:
        def record_hop(frames: int):
            return to_mono_f32(recorder.record(numframes=frames))

        for utterance in iter_live_utterances(
            record_hop, args.hop, args.buffer, lambda: False
        ):
            japanese, english = pipeline.run(utterance)
            pair = (japanese, english)
            if pair == last_pair or not (japanese or english):
                continue
            last_pair = pair
            stamp = time.strftime("%H:%M:%S")
            if japanese:
                print(f"[{stamp}] JA  {japanese}", flush=True)
            if english:
                print(f"         {pipeline.target_tag}  {english}", flush=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped.")
        raise SystemExit(0)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
