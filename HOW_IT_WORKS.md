# How JP Live Translate works

Local Japanese → English captions on macOS. Audio is recognized on-device, translated on-device, and drawn as a transparent overlay on top of whatever you are watching.

This note is the map of the code: files, data flow, and the timing rules that keep subtitles on the spoken sentence.

---

## 1. What you are looking at

There are two ways to run the same brain:

| Mode | Command | What happens |
| --- | --- | --- |
| **Prepare** | `python prepare.py --file video.mp4 --to zh` | Transcribe + translate once; write `cues/*.zh.cues.json`; no playback |
| **Play (launches VLC)** | `python overlay.py --file video.mp4 --to zh` | Load stored captions, open VLC, overlay on VLC’s clock |
| **Watch (you open the video)** | `python watch.py` | Leave running; open any file in VLC/QuickTime; overlay if that file (or a same-size copy) was prepared, else silent |
| **Live** | `python overlay.py --mic` (or BlackHole) | Listen in 0.5s hops, cut on a pause, then transcribe + translate that utterance |

`translate.py` is the same live pipeline without the GUI: it prints `JA` / `EN` to the terminal.

The file demo is the path that is actually usable today. macOS cannot tap speaker output unless you install a virtual cable (BlackHole). Without that, “live YouTube” has nothing to hear.

---

## 2. Files

```
jp-live-translate/
├── prepare.py      Transcribe + translate a file; save captions; no playback
├── overlay.py      GUI overlay; --file launches VLC with stored captions
├── watch.py        Background: detect VLC/QuickTime; overlay if cues exist
├── player_attach.py  Read open-player path + time via AppleScript
├── cues_store.py   Caption JSON in cues/ (*.en.cues.json / *.zh.cues.json)
├── cues/           All prepared caption files (gitignored)
├── engine.py       ASR, translation, sentence timing, live pause-cut
├── translate.py    Audio I/O (devices, ffmpeg decode) + console live mode
├── requirements.txt
```

Responsibilities stay split on purpose:

- **`translate.py`** talks to the OS (microphones, ffmpeg).
- **`engine.py`** turns samples into Japanese + English, and (for files) into timed cues.
- **`overlay.py`** never does ML itself. It shows text, and for files it asks “what caption belongs at *this* playback time?”

---

## 3. The translation pipeline

Everything goes through `load_pipeline()` in `engine.py`.

```
                 Japanese audio (16 kHz, mono, float32)
                              │
                              ▼
                    ┌─────────────────┐
                    │  ASR (Whisper)  │  language="ja", task="transcribe"
                    │  MLX on Apple   │  or faster-whisper CPU int8
                    │  Silicon        │
                    └────────┬────────┘
                             │ Japanese text
                             ▼
                    ┌─────────────────┐
                    │  MT (cascade)   │  opus-mt-ja-en via CTranslate2
                    │  default        │
                    └────────┬────────┘
                             │ English text
                             ▼
                      overlay / console
```

**Cascade** (`--mt cascade`, the default) is two models:

1. Whisper writes Japanese.
2. Helsinki opus-mt (CTranslate2 int8) translates that text to **English** (`--to en`, default) or **Simplified Chinese** (`--to zh`).

`--to zh` uses `shun89/opus-mt-ja-zh` through Transformers (local, no API). First Chinese run downloads the weights. Needs `torch` and `transformers` in the venv (`pip install torch transformers`).

Showing both languages is intentional. If JA is wrong, the error is ASR (e.g. 紫外線 → 市外線). If JA is right and EN is weird, the error is MT.

**Whisper-only MT** (`--mt whisper`) skips opus-mt and runs Whisper again with `task="translate"`. Worse Japanese fidelity, useful as a fallback if the CT2 model fails to load.

### ASR backend

`build_asr()` picks:

- **`mlx-whisper`** on Darwin ARM (`mlx-community/whisper-small-mlx` or `medium`).
- **`faster-whisper`** otherwise (CPU, int8).

`--backend auto|mlx|faster-whisper` overrides that. First run downloads weights from Hugging Face.

Shared constants:

| Name | Value | Role |
| --- | --- | --- |
| `SAMPLE_RATE` | 16 000 | Whisper’s native rate. All audio is resampled here. |
| `BUFFER_SECONDS` | 8 | Max live utterance length. **Not** used to slice file captions. |
| `HOP_SECONDS` | 0.5 | Live analysis frame. |
| `JA_PROMPT` | news / weather Japanese | Nudges Whisper toward the right domain. |
| `HALLUCINATIONS` | “thanks for watching”, ご視聴, … | Drop typical Whisper junk. |

---

## 4. File demo flow (the synced-subtitle path)

This is **two programs**. Prepare waits until captions are stored. Play does no ML.

```
  1) python prepare.py --file video.mp4 --to zh
           │
           ▼
  Whisper + opus-mt over the whole file (slow, once)
           │
           ▼
  Save in the project:
  cues/video.mp4.zh.cues.json
           │
           ▼
  2) python overlay.py --file video.mp4 --to zh
           │
           ▼
  Load that JSON (no Whisper) → VLC + overlay
  Same file again? Skip prepare; just play.
```

Two clocks used to fight each other. That was the subtitle-sync bug.

| Clock | What it is | Why it is wrong for captions |
| --- | --- | --- |
| Wall clock | `time.time()` after “start ffplay” | ffplay starts late; captions run **ahead** of speech |
| Whisper `segment.end` | Model’s guess that the phrase ended | Often **early**; next sentence appears while this one is still spoken |
| **Media `position()`** | Where the player actually is | The clock the ear is hearing |

File mode now uses only the media clock.

### 4.1 From audio to cues

`MlxAsr.transcribe_timed` / `FasterWhisperAsr.transcribe_timed`:

1. Transcribe the **whole file** with `word_timestamps=True`.
2. Flatten every word to `(start, end, token)`.
3. `_sentences_from_words` walks those tokens and cuts whenever a token contains `。`, `！`, `？`, `!`, or `?`.
4. If word times are missing, `_sentences_from_segments` splits the coarse Whisper segment by `。` and shares the segment’s duration by character count.
5. `_merge_timed_segments` only glues fragments shorter than 1s. It will not merge two finished sentences just because they abut.

Each sentence is then translated on its own (opus-mt already splits on `。` internally too).

A cue is a tuple:

```text
(t0, t1, japanese, english)
```

- `t0` — first word of that sentence (seconds from the start of the file).
- `t1` — **display** end, not the raw last-word time (see below).

### 4.2 Why the next line used to appear too soon

Whisper’s last-word time is optimistic. Example from the weather clip:

```text
sentence 1 words:  0.60 → 4.2     「…天気を伝えます。」
sentence 2 words:  4.08 → …      「高気圧に…」
```

If you switch at 4.08, sentence 2 is on screen while sentence 1 is still being said.

`_hold_captions_until_spoken` therefore:

1. Sets display end to `last_word_end + 0.65s` (`CAPTION_HOLD_PAD`).
2. If there is a gap before the next sentence, keeps the current line until that next `t0` (no flicker).
3. Lets cues **overlap**. The overlay picks the **first** matching cue.

```
time →    0.6          4.2     4.85        9.5
          |-- sentence 1 ---------|
                       |---- sentence 2 ---------|
          ^^^^^^^^^^^^^^^^ shown 1
                               ^^^ still 1 (first match)
                                  ^^^^^^ shown 2
```

`FileDemoPlayer._cue_at(t)`:

```python
for t0, t1, ja, en in self.cues:
    if t0 <= t < t1:
        return that cue   # first match wins
```

So the next sentence cannot steal the overlay until the current hold is over. Slightly late on the switch is better than cutting the speaker off.

The 0.65s pad is the knob if a clip still feels early or sticky.

### 4.3 Playback details

- Poll interval is 50 ms (20 Hz). That is finer than subtitle changes, cheap enough for a timer.
- `show_subtitle(..., hold_ms=60_000)` so the overlay’s own clear-timer does not wipe a line mid-cue. The player clears when `t` falls in a gap or the file ends.
- Overlay is frameless, always-on-top, click-through by default (`--drag` if you need to move it). Text is drawn with a dark outline instead of a black card, so the video stays visible.

---

## 5. Live flow (mic / BlackHole)

Live cannot wait for the whole show. It also cannot show a line at the *start* of a sentence: Whisper needs a finished chunk.

```
  0.5s hop from the capture device
           │
           ▼
  iter_live_utterances()
           │  RMS > 0.008  → speech
           │  ~0.4s trailing silence → flush utterance
           │  or 8s max → flush anyway (so one breath cannot stall forever)
           │
           ▼
  pipeline.run(utterance)
           │  transcribe Japanese (no word times; the chunk is short)
           │  opus-mt → English
           │
           ▼
  text_ready(ja, en, 6000)  → overlay holds ~6s or until the next utterance
```

Implications:

- Captions appear at the **end** of a phrase (after the pause), not while it is being spoken. That is the live-latency tradeoff. File mode does not have this problem because it precomputes.
- Duplicate identical `(ja, en)` pairs are dropped so a repeated decode does not flicker.
- `split_speech()` (Silero VAD) still exists in `engine.py` but file captions no longer use those 8s windows. Cutting on VAD max-length was the other “next sentence too early” bug.

### Capturing system audio on macOS

SoundCard cannot loop back speakers on Darwin.

1. `brew install blackhole-2ch`
2. Audio MIDI Setup → Multi-Output Device (speakers + BlackHole)
3. Play YouTube (or anything) through that device
4. `python overlay.py --device BlackHole`

`--mic` records the built-in microphone instead (useful for talking at the laptop).

---

## 6. Overlay structure

```
QApplication
├── SubtitleOverlay          always-on-top, translucent, bottom-center
│   ├── OutlinedLabel JA     gold
│   └── OutlinedLabel EN     white
├── TranslatorWorker         QThread: ML + file decode
└── FileDemoPlayer           only in --file mode
    └── VLC.app window       real VLC (HTTP clock for captions)
```

Signals (Qt rule: emit from the worker, slot runs on the GUI thread):

| Signal | Meaning |
| --- | --- |
| `status_ready` | “Loading models…” etc. |
| `text_ready` | Live caption |
| `playback_ready` | File: start VLC now, before captions exist |
| `cues_batch` | File: more timed captions as 30s windows finish |
| `failed` | Exception text on the overlay |

Escape quits. A 400 ms raise-timer keeps the overlay above the video.

---

## 7. Audio decode

`load_audio_file()` always goes through ffmpeg:

```text
ffmpeg -i FILE -ac 1 -ar 16000 -f s16le pipe:1
```

int16 bytes → float32 in `[-1, 1]`. That array is what Whisper sees, whether the source was webm, ogg, or mp4.

Playback is a **second** decode: VLC plays the original file so picture and sound stay on VLC’s clock. ASR does not listen to the player; it already transcribed ffmpeg PCM. Qt/QuickTime is not used for video (some files are AV1, which those cannot decode).

---

## 8. Mental model: two different products in one repo

```
                 same overlay, same pipeline
                           │
            ┌──────────────┴──────────────┐
            │                             │
       FILE DEMO                      LIVE
   know the future                 cannot
   word timestamps                 pause-cut
   captions tied to video          captions tied to silence
   sync problem = timestamps       sync problem = latency
```

When something feels “wrong”, ask which product you are in.

- File: “the line changed before she finished” → hold pad / first-match / media clock (`engine.py` `CAPTION_HOLD_PAD`, `overlay.py` `_cue_at`).
- Live: “the line showed up late” → expected; you only get a line after a pause or 8s.
- “EN is nonsense but JA is fine” → opus-mt.
- “JA is the wrong homophone” → Whisper size / domain (`--model medium`).

---

## 9. Commands

```bash
source .venv/bin/activate

# 1) Prepare captions (slow, once). Does not play.
python prepare.py --file video.mp4 --to zh

# 2a) Play: this script launches VLC (smooth; no Whisper).
python overlay.py --file video.mp4 --to zh

# 2b) Or leave a watcher running and open the video yourself.
python watch.py
# python watch.py --ask    # pause + Yes/No if captions exist

# Live mic
python overlay.py --mic

# Console live
python translate.py --mic --list-devices
```

Useful flags (both `overlay.py` and `translate.py` where they apply):

| Flag | Effect |
| --- | --- |
| `--model small\|medium` | Whisper size |
| `--backend auto\|mlx\|faster-whisper` | ASR implementation |
| `--mt cascade\|nanosakura\|hymt\|whisper` | opus-mt; NanoSakura 0.2B; Hy-MT2 1.8B; Whisper (English-only) |
| `--to en\|zh` | English (default) or Simplified Chinese |
| `--buffer` / `--hop` | Live cut length / frame (not file cue length) |

---

## 10. Where to read next

Start at `prepare.py` for file captions, then `overlay.py`:

1. `prepare.py` — Whisper + MT, writes `cues/*.zh.cues.json`.
2. `overlay.py --file` — load that JSON, VLC + overlay (no models).
3. `FileDemoPlayer._poll` / `_cue_at` — media time → visible line.
4. `iter_live_utterances` — only if you care about mic / BlackHole.

---

## 11. Conclusion — where this actually is

This is **not a live translator yet**, and it is **not a one-stop product**.

**Live is code, not a working setup.** The mic / BlackHole path exists, but macOS cannot tap speaker audio without a virtual cable, and this machine is not capturing a livestream. What you can watch today is a **file demo**: a video already on disk.

**The working process is offline and manual-feeling:**

1. Get a video.
2. `python prepare.py --file that.mp4 --to zh` → wait until `cues/*.zh.cues.json` is saved.
3. Either `python overlay.py --file that.mp4 --to zh` (launches VLC), or leave `python watch.py` running and open the file in VLC yourself.
4. Same video again? Skip step 2 unless you pass `--force`.

That last step is where most of the work went. The 0.65s hold pad, overlapping cues, and “don’t advance until this sentence is done” rules were **tuned on an early weather clip**. They are heuristics, not a model you train — but they are still clip-sensitive. A new video (different speaker, speed, or Whisper timestamp bias) can drift again. Then you re-run the file, watch it, and nudge timing until it matches. That is the same loop, not “drop any video in and it is done.”

So the honest product shape today:

| What it looks like | What it actually is |
| --- | --- |
| Live JP → EN overlay | File playback with precomputed captions |
| One-click on any stream | Decode this file, translate, then sync by hand if needed |
| Finished app | A working demo of the hard parts (local ASR, MT, overlay, clock) |

What is still missing for a complete one-stop tool: system-audio loopback (BlackHole + Multi-Output), captions that appear *while* someone is speaking (not after a pause, and not only after a full-file pass), and timing that generalizes without per-clip adjustment.
