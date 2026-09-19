# JP Live Translate

Local Japanese → English or Simplified Chinese captions on macOS. Speech recognition and translation run on-device. A transparent overlay sits on top of VLC or QuickTime.

There is no cloud API and no `.env`. First run downloads model weights from Hugging Face into the local cache.

## Install

Needs Python 3.11+, [ffmpeg](https://ffmpeg.org/), and [VLC](https://www.videolan.org/) (file playback).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Apple Silicon, ASR uses MLX Whisper by default. `--to zh` also needs PyTorch:

```bash
pip install torch
```

## Use

Put a video on disk, then:

```bash
# 1) Transcribe + translate once. Writes cues/*.en.cues.json (or .zh).
python prepare.py --file video.mp4 --to en

# 2a) This script launches VLC and overlays the stored captions.
python overlay.py --file video.mp4 --to en

# 2b) Or leave a watcher running and open the file yourself in VLC / QuickTime.
python watch.py
```

Live mic (captions appear after a pause, not while someone is still speaking):

```bash
python overlay.py --mic
python translate.py --mic   # console, no overlay
```

macOS cannot tap speaker output unless you install a virtual cable such as BlackHole, then `python overlay.py --device BlackHole`.

Useful flags: `--model small|medium`, `--backend auto|mlx|faster-whisper`, `--mt cascade|nanosakura|hymt|whisper`, `--to en|zh`.

See [HOW_IT_WORKS.md](HOW_IT_WORKS.md) for the pipeline, timing rules, and file layout. Local videos and generated captions stay out of git (`file/`, `cues/`).
