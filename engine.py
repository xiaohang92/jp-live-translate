"""ASR, translation, and speech-segmentation backends."""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16_000
BUFFER_SECONDS = 8.0
HOP_SECONDS = 0.5
SILENCE_RMS = 0.008
JA_PROMPT = "日本語のニュース、会話、天気予報。"
HALLUCINATIONS = (
    "thank you for watching",
    "thanks for watching",
    "please subscribe",
    "please like and subscribe",
    "see you next time",
    "ご視聴ありがとうございました",
    "ご視聴頂きありがとうございます",
    "チャンネル登録",
    "字幕作成者",
    "ご視聴",
)
MLX_REPOS = {
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
}
OPUS_CT2_REPO = "playto-mt/opus-mt-ja-en-ct2"
OPUS_JA_ZH_HF = "shun89/opus-mt-ja-zh"
NANOSAKURA_REPO = "telecomadm1145/NanoSakura-2.2-0.2B"
HYMT_REPO = "tencent/Hy-MT2-1.8B"
MT_CHOICES = ("cascade", "nanosakura", "hymt", "whisper")
TARGET_TAGS = {"en": "EN", "zh": "ZH"}


def _quiet_hub() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _hub_snapshot(repo_id: str) -> str:
    """Use the local Hugging Face cache; download only if missing."""
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(repo_id, local_files_only=True)
    except Exception:
        return snapshot_download(repo_id)


def _from_pretrained(loader, name: str, **kwargs):
    try:
        return loader(name, local_files_only=True, **kwargs)
    except Exception:
        return loader(name, **kwargs)


def normalize_target(value: str) -> str:
    """Map CLI aliases to en | zh."""
    key = value.strip().lower().replace("_", "-")
    if key in {"en", "eng", "english"}:
        return "en"
    if key in {"zh", "cn", "zh-cn", "zh-hans", "chinese", "中文"}:
        return "zh"
    raise ValueError(f"Unsupported target {value!r}. Use en or zh.")


def is_silent(audio: np.ndarray, threshold: float = SILENCE_RMS) -> bool:
    if audio.size == 0:
        return True
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float32))))
    return rms < threshold


def is_junk(text: str) -> bool:
    compact = "".join(text.lower().split())
    if len(compact) < 2:
        return True
    return any(phrase.replace(" ", "") in compact for phrase in HALLUCINATIONS)


def to_mono_f32(audio: np.ndarray) -> np.ndarray:
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return np.ascontiguousarray(audio, dtype=np.float32)


def _join_fw_segments(segments) -> str:
    parts: list[str] = []
    for segment in segments:
        if getattr(segment, "no_speech_prob", 0.0) > 0.65:
            continue
        if getattr(segment, "avg_logprob", 0.0) < -1.0:
            continue
        text = segment.text.strip()
        if text:
            parts.append(text)
    return " ".join(parts).strip()


class FasterWhisperAsr:
    def __init__(self, model_name: str) -> None:
        from faster_whisper import WhisperModel

        threads = max(2, min(8, (os.cpu_count() or 4) // 2))
        self.model = WhisperModel(
            model_name,
            device="cpu",
            compute_type="int8",
            cpu_threads=threads,
        )
        self.label = f"faster-whisper/{model_name} cpu-int8"

    def transcribe(self, audio: np.ndarray) -> str:
        segments, _info = self.model.transcribe(
            audio,
            language="ja",
            task="transcribe",
            initial_prompt=JA_PROMPT,
            beam_size=1,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 400,
                "speech_pad_ms": 200,
            },
            no_speech_threshold=0.65,
            compression_ratio_threshold=2.4,
            without_timestamps=True,
        )
        return _join_fw_segments(segments)

    def transcribe_timed(self, audio: np.ndarray) -> list[tuple[float, float, str]]:
        segments, _info = self.model.transcribe(
            audio,
            language="ja",
            task="transcribe",
            initial_prompt=JA_PROMPT,
            beam_size=1,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 400,
                "speech_pad_ms": 200,
            },
            without_timestamps=False,
            word_timestamps=True,
        )
        words: list[tuple[float, float, str]] = []
        fallback: list[tuple[float, float, str]] = []
        for segment in segments:
            text = segment.text.strip()
            if text:
                fallback.append((float(segment.start), float(segment.end), text))
            for word in getattr(segment, "words", None) or []:
                token = (word.word or "").strip()
                if token:
                    words.append((float(word.start), float(word.end), token))
        timed = _sentences_from_words(words) or _sentences_from_segments(fallback)
        return _merge_timed_segments(timed)

    def translate_audio(self, audio: np.ndarray) -> str:
        segments, _info = self.model.transcribe(
            audio,
            language="ja",
            task="translate",
            beam_size=1,
            condition_on_previous_text=False,
            vad_filter=True,
            without_timestamps=True,
        )
        return _join_fw_segments(segments)


class MlxAsr:
    def __init__(self, model_name: str) -> None:
        _quiet_hub()
        import mlx_whisper  # noqa: F401

        self.repo = _hub_snapshot(MLX_REPOS.get(model_name, MLX_REPOS["small"]))
        self.model_name = model_name
        self.label = f"mlx-whisper/{model_name} metal"
        self._mlx = mlx_whisper

    def transcribe(self, audio: np.ndarray) -> str:
        result = self._mlx.transcribe(
            to_mono_f32(audio),
            path_or_hf_repo=self.repo,
            language="ja",
            task="transcribe",
            initial_prompt=JA_PROMPT,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.4,
            fp16=True,
        )
        return str(result.get("text", "")).strip()

    def transcribe_timed(self, audio: np.ndarray) -> list[tuple[float, float, str]]:
        result = self._mlx.transcribe(
            to_mono_f32(audio),
            path_or_hf_repo=self.repo,
            language="ja",
            task="transcribe",
            initial_prompt=JA_PROMPT,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            word_timestamps=True,
            fp16=True,
        )
        words: list[tuple[float, float, str]] = []
        fallback: list[tuple[float, float, str]] = []
        for segment in result.get("segments") or []:
            text = str(segment.get("text", "")).strip()
            if text:
                fallback.append(
                    (float(segment.get("start", 0.0)), float(segment.get("end", 0.0)), text)
                )
            for word in segment.get("words") or []:
                token = str(word.get("word", "")).strip()
                if token:
                    words.append(
                        (float(word.get("start", 0.0)), float(word.get("end", 0.0)), token)
                    )
        timed = _sentences_from_words(words) or _sentences_from_segments(fallback)
        if not timed:
            text = str(result.get("text", "")).strip()
            if text:
                timed.append((0.0, max(0.1, audio.size / SAMPLE_RATE), text))
        return _merge_timed_segments(timed)

    def translate_audio(self, audio: np.ndarray) -> str:
        result = self._mlx.transcribe(
            to_mono_f32(audio),
            path_or_hf_repo=self.repo,
            language="ja",
            task="translate",
            condition_on_previous_text=False,
            fp16=True,
        )
        return str(result.get("text", "")).strip()


class OpusMtTranslator:
    """Japanese → English (CTranslate2) or Chinese (Marian / Transformers)."""

    def __init__(self, target: str = "en") -> None:
        self.target = normalize_target(target)
        if self.target == "zh":
            self._init_ja_zh()
        else:
            self._init_ja_en()

    def _init_ja_en(self) -> None:
        import sentencepiece as spm
        from ctranslate2 import Translator

        model_dir = _resolve_mt_dir("en")
        self._backend = "ct2"
        self.translator = Translator(str(model_dir), device="cpu", compute_type="int8")
        self.src = spm.SentencePieceProcessor(model_file=str(model_dir / "source.spm"))
        self.tgt = spm.SentencePieceProcessor(model_file=str(model_dir / "target.spm"))
        self.label = "opus-mt-ja-en ct2-int8"

    def _init_ja_zh(self) -> None:
        _quiet_hub()
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self._backend = "hf"
        self._torch = torch
        self.tok = _from_pretrained(AutoTokenizer.from_pretrained, OPUS_JA_ZH_HF)
        self.model = _from_pretrained(AutoModelForSeq2SeqLM.from_pretrained, OPUS_JA_ZH_HF)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.model.eval()
        if getattr(self.model, "generation_config", None) is not None:
            self.model.generation_config.max_length = None
        self.label = "opus-mt-ja-zh transformers"

    def translate(self, japanese: str) -> str:
        pieces: list[str] = []
        for sentence in _split_ja(japanese):
            if self._backend == "hf":
                pieces.append(self._translate_hf(sentence))
            else:
                pieces.append(self._translate_ct2(sentence))
        sep = "" if self.target == "zh" else " "
        return sep.join(p for p in pieces if p)

    def _translate_ct2(self, sentence: str) -> str:
        tokens = self.src.encode(sentence, out_type=str) + ["</s>"]
        result = self.translator.translate_batch([tokens], beam_size=2)
        hyp = [tok for tok in result[0].hypotheses[0] if tok not in ("</s>", "<unk>")]
        return self.tgt.decode(hyp).strip()

    def _translate_hf(self, sentence: str) -> str:
        inp = self.tok(sentence, return_tensors="pt", truncation=True, max_length=256)
        with self._torch.inference_mode():
            out = self.model.generate(**inp, max_new_tokens=96, num_beams=2)
        return self.tok.decode(out[0], skip_special_tokens=True).strip()

    def translate_many(self, texts: list[str]) -> list[str]:
        """Batch when possible; Chinese Marian is slow one-by-one on CPU."""
        if not texts:
            return []
        if self._backend != "hf":
            return [self.translate(text) for text in texts]
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        inp = self.tok(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        with self._torch.inference_mode():
            out = self.model.generate(**inp, max_new_tokens=96, num_beams=2)
        return [
            self.tok.decode(row, skip_special_tokens=True).strip() for row in out
        ]


class NanoSakuraTranslator:
    """Japanese → Chinese specialist (~0.2B). Better sentence meaning than opus-mt."""

    def __init__(self) -> None:
        _quiet_hub()
        import torch
        from transformers import AutoModelForSeq2SeqLM, PreTrainedTokenizerFast

        self.target = "zh"
        self._torch = torch
        self._device = torch.device("cpu")
        self.tok = _from_pretrained(PreTrainedTokenizerFast.from_pretrained, NANOSAKURA_REPO)
        self.model = _from_pretrained(
            AutoModelForSeq2SeqLM.from_pretrained,
            NANOSAKURA_REPO,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
        self.model.to(self._device)
        self.model.eval()
        self.label = "nanosakura-2.2-0.2B ja-zh"

    def translate(self, japanese: str) -> str:
        text = japanese.strip()
        if not text:
            return ""
        return self._translate_one(text)

    def translate_many(self, texts: list[str]) -> list[str]:
        return [self.translate(text) for text in texts]

    def _translate_one(self, sentence: str) -> str:
        ids = self.tok.encode(sentence.rstrip() + "<eos>")
        if not ids:
            return ""
        if len(ids) > 256:
            ids = ids[:256]
        inp = self._torch.tensor([ids], device=self._device)
        with self._torch.inference_mode():
            out = self.model.generate(inp, max_new_tokens=128)
        return self.tok.decode(out[0], skip_special_tokens=True).strip()


class HyMtTranslator:
    """Tencent Hy-MT2 1.8B. Stronger sentence translation; glossary for 日式英语."""

    _GLOSS_ZH = (
        "参考下面的翻译：\n"
        "ゲット 翻译成 搞到\n"
        "ゲットならず 翻译成 没搞到\n"
        "チェック 翻译成 确认\n"
        "ナイス 翻译成 漂亮\n"
    )

    def __init__(self, target: str = "zh") -> None:
        _quiet_hub()
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.target = normalize_target(target)
        self._torch = torch
        if torch.backends.mps.is_available():
            self._device = torch.device("mps")
            dtype = torch.float16
        else:
            self._device = torch.device("cpu")
            dtype = torch.float32
        self.tok = _from_pretrained(
            AutoTokenizer.from_pretrained,
            HYMT_REPO,
            trust_remote_code=True,
        )
        try:
            self.model = _from_pretrained(
                AutoModelForCausalLM.from_pretrained,
                HYMT_REPO,
                trust_remote_code=True,
                torch_dtype=dtype,
            )
            self.model.to(self._device)
        except Exception:
            self._device = torch.device("cpu")
            self.model = _from_pretrained(
                AutoModelForCausalLM.from_pretrained,
                HYMT_REPO,
                trust_remote_code=True,
                torch_dtype=torch.float32,
            )
            self.model.to(self._device)
        self.model.eval()
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.label = f"hy-mt2-1.8B ja-{self.target}"

    def translate(self, japanese: str) -> str:
        text = japanese.strip()
        if not text:
            return ""
        return self._translate_one(text)

    def translate_many(self, texts: list[str]) -> list[str]:
        return [self.translate(text) for text in texts]

    def _prompt(self, source: str) -> str:
        lang = "中文" if self.target == "zh" else "英语"
        head = self._GLOSS_ZH if self.target == "zh" else ""
        return (
            f"{head}将以下文本翻译为{lang}，注意只需要输出翻译后的结果，不要额外解释：\n"
            f"{source}"
        )

    def _translate_one(self, sentence: str) -> str:
        messages = [{"role": "user", "content": self._prompt(sentence)}]
        encoded = self.tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )
        if hasattr(encoded, "items"):
            gen_in = {key: value.to(self._device) for key, value in encoded.items()}
        else:
            ids = encoded.to(self._device)
            gen_in = {"input_ids": ids}
        prompt_len = int(gen_in["input_ids"].shape[-1])
        with self._torch.inference_mode():
            out = self.model.generate(
                **gen_in,
                max_new_tokens=96,
                do_sample=False,
                pad_token_id=self.tok.pad_token_id,
            )
        text = self.tok.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
        return text.split("\n")[0].strip()


def _resolve_mt_dir(target: str):
    from pathlib import Path

    if target != "en":
        raise ValueError(f"CTranslate2 MT dir is English-only, got {target}")
    root = Path(_hub_snapshot(OPUS_CT2_REPO))
    return _resolve_opus_dir(root)


def _resolve_opus_dir(root):
    from pathlib import Path
    import tarfile

    root = Path(root)
    found = next(root.rglob("source.spm"), None)
    if found is not None:
        return found.parent
    tarball = root / "opus-mt-ja-en-ct2.tar.gz"
    if not tarball.is_file():
        raise FileNotFoundError(f"No CTranslate2 model in {root}")
    extract_kw = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
    with tarfile.open(tarball) as tar:
        tar.extractall(root, **extract_kw)
    found = next(root.rglob("source.spm"), None)
    if found is None:
        raise FileNotFoundError(f"source.spm missing after extracting {tarball}")
    return found.parent


def _split_ja(text: str) -> list[str]:
    buf: list[str] = []
    current: list[str] = []
    for char in text:
        current.append(char)
        if char in "。！？!?":
            piece = "".join(current).strip()
            if piece:
                buf.append(piece)
            current = []
    tail = "".join(current).strip()
    if tail:
        buf.append(tail)
    return buf or [text]


def _apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def build_asr(model_name: str, backend: str = "auto"):
    want_mlx = backend == "mlx" or (backend == "auto" and _apple_silicon())
    if want_mlx:
        try:
            return MlxAsr(model_name)
        except Exception:
            if backend == "mlx":
                raise
    return FasterWhisperAsr(model_name)


@dataclass
class Pipeline:
    asr: object
    mt: object | None
    target: str = "en"

    @property
    def label(self) -> str:
        mt = self.mt.label if self.mt else "whisper-translate"
        return f"ASR {self.asr.label} · MT {mt}"

    @property
    def target_tag(self) -> str:
        return TARGET_TAGS.get(self.target, "EN")

    def run(self, audio: np.ndarray) -> tuple[str, str]:
        japanese = self.asr.transcribe(audio)
        if not japanese or is_junk(japanese):
            return "", ""
        if self.mt is not None:
            translated = self.mt.translate(japanese)
        else:
            translated = self.asr.translate_audio(audio)
        if is_junk(translated):
            translated = ""
        return japanese, translated


def build_translator(mt: str, target: str):
    """Load text MT only (no Whisper). None means caller should use Whisper translate."""
    _quiet_hub()
    target = normalize_target(target)
    kind = (mt or "cascade").strip().lower()
    if kind == "hymt":
        try:
            return HyMtTranslator(target)
        except Exception as exc:
            raise RuntimeError(
                f"Hy-MT2 failed ({exc}). First run downloads ~1.8B weights."
            ) from exc
    if kind == "nanosakura":
        if target != "zh":
            raise RuntimeError("--mt nanosakura is Japanese→Chinese only. Use --to zh.")
        try:
            return NanoSakuraTranslator()
        except Exception as exc:
            raise RuntimeError(
                f"NanoSakura MT failed ({exc}). "
                "Install: pip install torch transformers"
            ) from exc
    if kind == "whisper" and target != "en":
        print(
            "warning: Whisper translate is English-only; using cascade for --to zh",
            file=sys.stderr,
        )
        kind = "cascade"
    if kind == "cascade":
        try:
            return OpusMtTranslator(target)
        except Exception as exc:
            if target != "en":
                raise RuntimeError(
                    f"Japanese→Chinese MT failed ({exc}). "
                    "Install: pip install torch transformers"
                ) from exc
            print(
                f"warning: cascade MT unavailable ({exc}); falling back to Whisper translate",
                file=sys.stderr,
            )
            return None
    if kind == "whisper":
        return None
    raise ValueError(f"Unknown --mt {mt!r}. Use cascade, nanosakura, hymt, or whisper.")


def load_pipeline(
    model_name: str,
    backend: str = "auto",
    mt: str = "cascade",
    target: str = "en",
) -> Pipeline:
    _quiet_hub()
    target = normalize_target(target)
    asr = build_asr(model_name, backend)
    translator = build_translator(mt, target)
    return Pipeline(asr=asr, mt=translator, target=target)


def load_model(
    model_name: str, backend: str = "auto", mt: str = "cascade", target: str = "en"
) -> Pipeline:
    """Back-compat alias used by overlay/console."""
    return load_pipeline(model_name, backend=backend, mt=mt, target=target)


def transcribe_and_translate(pipeline: Pipeline, audio: np.ndarray) -> tuple[str, str]:
    return pipeline.run(audio)


# Extra time after Whisper's last word before the next caption may appear.
# Whisper end times run early; switching at t1 cuts off the spoken line.
CAPTION_HOLD_PAD = 0.65
FILE_WINDOW_SECONDS = 30.0


def reading_hold_seconds(japanese: str, translated: str) -> float:
    """How long a caption should stay up after it appears, from text length."""
    text = (translated or "").strip() or (japanese or "").strip()
    if not text:
        return 1.5
    cjk = 0
    other = 0
    for char in text:
        if char.isspace():
            continue
        code = ord(char)
        if 0x3040 <= code <= 0x30FF or 0x3400 <= code <= 0x9FFF or 0xF900 <= code <= 0xFAFF:
            cjk += 1
        else:
            other += 1
    return min(5.5, max(1.5, cjk * 0.18 + other * 0.06))


def cue_visible_end(
    t0: float,
    t1: float,
    japanese: str,
    translated: str,
    next_t0: float | None = None,
) -> float:
    """When to hide a line. Does not keep it through a long pause until the next sentence."""
    hold = reading_hold_seconds(japanese, translated)
    stretched = next_t0 is not None and abs(t1 - next_t0) < 0.08
    if stretched:
        end = t0 + hold
    else:
        end = max(t1, t0 + hold)
    if next_t0 is not None and next_t0 > t0:
        end = min(end, next_t0)
    return max(end, t0 + 0.6)


def build_cues(
    pipeline: Pipeline,
    audio: np.ndarray,
    max_seconds: float = BUFFER_SECONDS,
) -> list[tuple[float, float, str, str]]:
    """One caption per spoken sentence; stay up until that sentence is done."""
    del max_seconds  # kept so callers do not change
    return list(_cues_from_timed(pipeline, pipeline.asr.transcribe_timed(audio)))


def iter_file_cues(pipeline: Pipeline, audio: np.ndarray, should_stop=None):
    """Yield (end_sec, total_sec, cues) for ~30s windows so playback can start now."""
    total = audio.size / SAMPLE_RATE
    start = 0.0
    while start < total:
        if should_stop is not None and should_stop():
            return
        end = min(total, start + FILE_WINDOW_SECONDS)
        i0 = int(start * SAMPLE_RATE)
        i1 = max(i0 + 1, int(end * SAMPLE_RATE))
        timed = [
            (t0 + start, t1 + start, text)
            for t0, t1, text in pipeline.asr.transcribe_timed(audio[i0:i1])
        ]
        yield end, total, _cues_from_timed(pipeline, timed)
        start = end


def _cues_from_timed(
    pipeline: Pipeline, timed: list[tuple[float, float, str]]
) -> list[tuple[float, float, str, str]]:
    japanese_lines: list[tuple[float, float, str]] = []
    for t0, t1, japanese in timed:
        if not japanese or is_junk(japanese):
            continue
        if t1 <= t0:
            t1 = t0 + 1.2
        japanese_lines.append((t0, t1, japanese))
    if not japanese_lines:
        return []
    texts = [ja for _t0, _t1, ja in japanese_lines]
    if pipeline.mt is not None and hasattr(pipeline.mt, "translate_many"):
        translated = pipeline.mt.translate_many(texts)
    elif pipeline.mt is not None:
        translated = [pipeline.mt.translate(text) for text in texts]
    else:
        translated = [""] * len(texts)
    cues: list[tuple[float, float, str, str]] = []
    for (t0, t1, japanese), english in zip(japanese_lines, translated, strict=True):
        if is_junk(english):
            english = ""
        cues.append((t0, t1, japanese, english))
    return _hold_captions_until_spoken(cues)


def _hold_captions_until_spoken(
    cues: list[tuple[float, float, str, str]],
) -> list[tuple[float, float, str, str]]:
    """Keep a line until it is spoken and readable, then hide during a pause."""
    held: list[tuple[float, float, str, str]] = []
    for i, (t0, t1, japanese, english) in enumerate(cues):
        next_t0 = cues[i + 1][0] if i + 1 < len(cues) else None
        speech_end = max(t1, t0 + 0.8) + CAPTION_HOLD_PAD
        end = cue_visible_end(t0, speech_end, japanese, english, next_t0)
        held.append((t0, end, japanese, english))
    return held


SENTENCE_END = "。！？!?"


def _join_ja_tokens(tokens: list[str]) -> str:
    return " ".join("".join(tokens).split()).strip()


def _sentences_from_words(
    words: list[tuple[float, float, str]],
) -> list[tuple[float, float, str]]:
    """Cut captions on 。！？ using word-level timestamps."""
    if not words:
        return []
    out: list[tuple[float, float, str]] = []
    buf: list[str] = []
    t0: float | None = None
    t1: float | None = None
    for start, end, token in words:
        if t0 is None:
            t0 = start
        t1 = end
        buf.append(token)
        if any(char in SENTENCE_END for char in token):
            text = _join_ja_tokens(buf)
            if text:
                out.append((t0, t1, text))
            buf, t0, t1 = [], None, None
    if buf and t0 is not None and t1 is not None:
        text = _join_ja_tokens(buf)
        if text:
            out.append((t0, t1, text))
    return out


def _sentences_from_segments(
    segments: list[tuple[float, float, str]],
) -> list[tuple[float, float, str]]:
    """If word times are missing, split a segment by 。 and share its duration."""
    out: list[tuple[float, float, str]] = []
    for t0, t1, text in segments:
        parts = [p for p in _split_ja(text) if p]
        if len(parts) <= 1:
            if text.strip():
                out.append((t0, t1, text.strip()))
            continue
        weights = [max(len(part), 1) for part in parts]
        total = float(sum(weights))
        cursor = t0
        span = max(t1 - t0, 0.8 * len(parts))
        for i, part in enumerate(parts):
            dur = span * (weights[i] / total)
            end = t1 if i == len(parts) - 1 else cursor + dur
            out.append((cursor, max(end, cursor + 0.6), part))
            cursor = end
    return out


def _merge_timed_segments(
    timed: list[tuple[float, float, str]], min_len: float = 1.0
) -> list[tuple[float, float, str]]:
    """Only glue fragments that are too short to stand as a caption."""
    if not timed:
        return []
    merged = [timed[0]]
    for start, end, text in timed[1:]:
        p0, p1, ptext = merged[-1]
        prev_short = (p1 - p0) < min_len
        this_short = (end - start) < min_len
        close = (start - p1) <= 0.35
        already_sentence = ptext[-1] in SENTENCE_END if ptext else False
        if (prev_short or this_short) and close and not already_sentence:
            merged[-1] = (p0, max(p1, end), f"{ptext} {text}".strip())
        else:
            merged.append((start, end, text))
    return merged


def split_speech(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    max_seconds: float = BUFFER_SECONDS,
):
    """Yield (chunk, start_sec, end_sec) for each Silero speech region."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    options = VadOptions(
        threshold=0.35,
        min_speech_duration_ms=400,
        min_silence_duration_ms=400,
        max_speech_duration_s=max_seconds,
        speech_pad_ms=200,
    )
    stamps = get_speech_timestamps(audio, options, sampling_rate=sample_rate)
    stamps = _merge_speech_stamps(stamps, sample_rate, max_len_s=max_seconds)
    if not stamps:
        if not is_silent(audio) and audio.size >= sample_rate // 2:
            yield audio, 0.0, audio.size / sample_rate
        return
    for stamp in stamps:
        start, end = int(stamp["start"]), int(stamp["end"])
        chunk = audio[start:end]
        if chunk.size >= sample_rate // 4:
            yield chunk, start / sample_rate, end / sample_rate


def _merge_speech_stamps(
    stamps: list[dict],
    sample_rate: int,
    max_gap_s: float = 0.45,
    max_len_s: float = BUFFER_SECONDS,
) -> list[dict]:
    """Glue short/close Silero cuts, but keep captions near max_len_s."""
    if not stamps:
        return []
    merged = [dict(stamps[0])]
    for stamp in stamps[1:]:
        prev = merged[-1]
        gap = (stamp["start"] - prev["end"]) / sample_rate
        prev_len = (prev["end"] - prev["start"]) / sample_rate
        combined = (stamp["end"] - prev["start"]) / sample_rate
        if combined <= max_len_s and (gap <= max_gap_s or prev_len < 1.0):
            prev["end"] = stamp["end"]
        else:
            merged.append(dict(stamp))
    return [
        stamp
        for stamp in merged
        if (stamp["end"] - stamp["start"]) / sample_rate >= 1.2
    ]


def iter_live_utterances(record_hop, hop_seconds: float, max_seconds: float, should_stop):
    """Cut live audio on pause (trailing silence) or max length."""
    hop_frames = max(1, int(hop_seconds * SAMPLE_RATE))
    max_frames = max(hop_frames, int(max_seconds * SAMPLE_RATE))
    silence_flush = max(int(0.4 * SAMPLE_RATE), hop_frames)
    pieces: list[np.ndarray] = []
    voiced = False
    trailing = 0
    while not should_stop():
        piece = to_mono_f32(record_hop(hop_frames))
        silent = is_silent(piece)
        if not silent:
            voiced = True
            trailing = 0
            pieces.append(piece)
        elif voiced:
            pieces.append(piece)
            trailing += piece.size
            if trailing >= silence_flush:
                yield np.concatenate(pieces)
                pieces = []
                voiced = False
                trailing = 0
                continue
        total = sum(p.size for p in pieces)
        if voiced and total >= max_frames:
            yield np.concatenate(pieces)
            pieces = []
            voiced = False
            trailing = 0
