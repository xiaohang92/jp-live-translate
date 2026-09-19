#!/usr/bin/env python3
"""Transcribe + translate a video and store captions in cues/. Does not play."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cues_store import load_cues, save_cues, sidecar_path
from engine import (
    MT_CHOICES,
    build_translator,
    is_junk,
    iter_file_cues,
    load_pipeline,
    normalize_target,
)
from translate import load_audio_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare stored Japanese captions (and EN/ZH) into the project cues/ folder."
    )
    parser.add_argument("--file", dest="audio_file", required=True, help="Audio/video file.")
    parser.add_argument(
        "--to",
        dest="target",
        default="en",
        help="Caption language: en (default) or zh (Simplified Chinese).",
    )
    parser.add_argument("--model", default="small", choices=("small", "medium"))
    parser.add_argument(
        "--backend", default="auto", choices=("auto", "mlx", "faster-whisper")
    )
    parser.add_argument(
        "--mt",
        default="cascade",
        choices=MT_CHOICES,
        help="cascade = opus-mt; nanosakura = ja→zh 0.2B; hymt = Hy-MT2 1.8B; whisper = English-only.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild from audio even if captions already exist.",
    )
    parser.add_argument(
        "--retranslate",
        action="store_true",
        help="Reuse stored Japanese lines; only run the translation model (no ffmpeg/Whisper).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    media = Path(args.audio_file)
    if not media.is_file():
        raise SystemExit(f"File not found: {args.audio_file}")
    try:
        target = normalize_target(args.target)
    except ValueError as exc:
        raise SystemExit(str(exc))

    media_s = str(media.resolve())
    existing = load_cues(media_s, target)
    out = sidecar_path(media_s, target)

    if args.retranslate:
        if not existing:
            raise SystemExit(
                f"No stored Japanese captions to reuse.\n"
                f"Run a full prepare first: python prepare.py --file \"{args.audio_file}\" --to {target}"
            )
        print(f"Retranslating {len(existing)} Japanese lines (no audio)…", flush=True)
        print("Loading translation model…", flush=True)
        translator = build_translator(args.mt, target)
        if translator is None:
            raise SystemExit("--retranslate needs a text MT model (cascade or nanosakura).")
        print(translator.label, flush=True)
        total = len(existing)
        all_cues = []
        for i, (t0, t1, japanese, _old) in enumerate(existing, 1):
            zh = translator.translate(japanese)
            if is_junk(zh):
                zh = ""
            all_cues.append((t0, t1, japanese, zh))
            if i == 1 or i == total or i % 10 == 0:
                print(f"Translated {i:4d} / {total}", flush=True)
        saved = save_cues(
            media_s,
            target,
            all_cues,
            model=args.model,
            backend=args.backend,
            mt=args.mt,
        )
        print(f"Saved {len(all_cues)} lines → {saved}")
        print(f'Restart watch.py, then open the video.')
        return 0

    if existing is not None and not args.force:
        print(f"Already prepared: {out}")
        print(f"{len(existing)} caption lines stored.")
        print(
            "Same audio, new translator: "
            f'python prepare.py --file "{args.audio_file}" --to {target} --mt nanosakura --retranslate'
        )
        return 0

    print("Loading models…", flush=True)
    pipeline = load_pipeline(
        args.model, backend=args.backend, mt=args.mt, target=target
    )
    print(pipeline.label, flush=True)
    print("Decoding audio…", flush=True)
    audio = load_audio_file(media_s)
    all_cues: list = []
    for end, total, batch in iter_file_cues(pipeline, audio):
        print(f"Captions {end:5.0f}s / {total:.0f}s", flush=True)
        all_cues.extend(batch)
    if not all_cues:
        raise SystemExit("No captions produced (silent file or ASR failed).")
    saved = save_cues(
        media_s,
        target,
        all_cues,
        model=args.model,
        backend=args.backend,
        mt=args.mt,
    )
    print(f"Saved {len(all_cues)} lines → {saved}")
    print(f'Play: python overlay.py --file "{args.audio_file}" --to {target}')
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
        raise SystemExit(130)
