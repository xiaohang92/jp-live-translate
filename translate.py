#!/usr/bin/env python3
"""Capture system audio in 3-second buffers and translate Japanese to English.

Uses a local faster-whisper model (small or medium). Output is printed to the
console. On macOS, true system-audio capture needs a virtual device such as
BlackHole: route sound through a Multi-Output Device (speakers + BlackHole),
then record from BlackHole. Use --mic to test with the default microphone.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings

import numpy as np
import soundcard as sc
from faster_whisper import WhisperModel

# SoundCard cannot tap speakers on macOS; include_loopback only works on Windows.
if sys.platform == "darwin":
    warnings.filterwarnings(
        "ignore",
        message="macOS does not support loopback recording functionality",
        category=Warning,
    )

SAMPLE_RATE = 16_000
BUFFER_SECONDS = 3.0
SILENCE_RMS = 0.008
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


def to_mono_f32(audio: np.ndarray) -> np.ndarray:
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return np.ascontiguousarray(audio, dtype=np.float32)


def is_silent(audio: np.ndarray, threshold: float = SILENCE_RMS) -> bool:
    if audio.size == 0:
        return True
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float32))))
    return rms < threshold


def load_model(name: str) -> WhisperModel:
    return WhisperModel(name, device="cpu", compute_type="int8")


def translate_audio(model: WhisperModel, audio: np.ndarray) -> str:
    segments, _info = model.transcribe(
        audio,
        language="ja",
        task="translate",
        beam_size=1,
        condition_on_previous_text=False,
    )
    return " ".join(segment.text.strip() for segment in segments).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local Japanese → English live translation (console)."
    )
    parser.add_argument(
        "--model",
        default="small",
        choices=("small", "medium"),
        help="faster-whisper model size (default: small).",
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
        help="Audio buffer length in seconds (default: 3).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list_devices:
        list_devices()
        return 0

    if args.buffer <= 0:
        raise SystemExit("--buffer must be positive")

    mic = find_capture_device(args.device, args.mic)
    frames = int(args.buffer * SAMPLE_RATE)

    print(f"Loading faster-whisper '{args.model}' on CPU (int8)...")
    print("First run downloads the model weights; later runs are local-only.")
    model = load_model(args.model)

    print(f"Capturing from: {mic.name}")
    print(
        f"Buffer: {args.buffer:g}s @ {SAMPLE_RATE} Hz | "
        "language=ja | task=translate (Japanese → English)"
    )
    print("Ctrl+C to stop.\n")

    with mic.recorder(samplerate=SAMPLE_RATE, channels=1) as recorder:
        while True:
            audio = to_mono_f32(recorder.record(numframes=frames))
            if is_silent(audio):
                continue

            text = translate_audio(model, audio)
            if text:
                print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped.")
        raise SystemExit(0)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
