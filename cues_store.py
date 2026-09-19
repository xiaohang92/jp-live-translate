"""Caption JSON lives in this project's cues/ folder, not next to the video."""

from __future__ import annotations

import json
from pathlib import Path

from engine import cue_visible_end, normalize_target

_PROJECT_ROOT = Path(__file__).resolve().parent
_LEGACY_HOME = Path.home() / ".jp-live-translate" / "cues"


def cues_dir() -> Path:
    return _PROJECT_ROOT / "cues"


def sidecar_path(media: str, target: str) -> Path:
    src = Path(media).resolve()
    lang = normalize_target(target)
    folder = cues_dir()
    named = folder / f"{src.name}.{lang}.cues.json"
    try:
        size = src.stat().st_size
    except OSError:
        return named
    if not named.is_file():
        return named
    existing = _payload_size(named)
    if existing in (None, size):
        return named
    return folder / f"{src.name}.{size}.{lang}.cues.json"


def file_fingerprint(media: str) -> dict:
    src = Path(media).resolve()
    st = src.stat()
    return {"source": str(src), "mtime_ns": st.st_mtime_ns, "size": st.st_size}


def cue_at(cues: list[tuple[float, float, str, str]], t: float):
    """First matching caption at media time t. Hides after a length-based read time."""
    for i, (t0, t1, japanese, translated) in enumerate(cues):
        next_t0 = cues[i + 1][0] if i + 1 < len(cues) else None
        end = cue_visible_end(t0, t1, japanese, translated, next_t0)
        if t0 <= t < end:
            return (t0, t1, japanese, translated)
    return None


def load_cues_auto(media: str, preferred: str = "auto"):
    """Load stored captions. preferred=auto tries zh then en."""
    if preferred != "auto":
        lang = normalize_target(preferred)
        cues = load_cues(media, lang)
        return (cues, lang) if cues else (None, lang)
    for lang in ("zh", "en"):
        cues = load_cues(media, lang)
        if cues:
            return cues, lang
    return None, "zh"


def load_cues(media: str, target: str) -> list[tuple[float, float, str, str]] | None:
    src = Path(media).resolve()
    try:
        size = src.stat().st_size
    except OSError:
        return None
    lang = normalize_target(target)
    folder = cues_dir()
    seen: set[Path] = set()
    for path in (
        sidecar_path(media, lang),
        folder / f"{src.name}.{lang}.cues.json",
        folder / f"{src.name}.{size}.{lang}.cues.json",
    ):
        key = _resolved(path)
        if key in seen:
            continue
        seen.add(key)
        cues = _read_cues(path, size)
        if cues:
            return cues
    if folder.is_dir():
        for path in folder.glob(f"*.{lang}.cues.json"):
            key = _resolved(path)
            if key in seen:
                continue
            seen.add(key)
            cues = _read_cues(path, size)
            if cues:
                return cues
    return _migrate_legacy(src, lang, size)


def save_cues(
    media: str,
    target: str,
    cues: list[tuple[float, float, str, str]],
    *,
    model: str,
    backend: str,
    mt: str,
) -> Path:
    folder = cues_dir()
    folder.mkdir(parents=True, exist_ok=True)
    path = sidecar_path(media, target)
    payload = {
        **file_fingerprint(media),
        "target": normalize_target(target),
        "model": model,
        "backend": backend,
        "mt": mt,
        "cues": [list(cue) for cue in cues],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _migrate_legacy(
    src: Path, lang: str, size: int
) -> list[tuple[float, float, str, str]] | None:
    name = f"{src.name}.{lang}.cues.json"
    dest_dir = cues_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_named = dest_dir / name
    candidates = [
        src.with_name(name),
        _PROJECT_ROOT / "file" / name,
        _LEGACY_HOME / name,
    ]
    for folder in (src.parent, _PROJECT_ROOT / "file", _LEGACY_HOME):
        if folder.is_dir():
            try:
                candidates.extend(folder.glob(f"*.{lang}.cues.json"))
            except OSError:
                pass
    seen: set[Path] = set()
    matches: list[Path] = []
    found: list[tuple[float, float, str, str]] | None = None
    for path in candidates:
        key = _resolved(path)
        if key in seen or key.parent == _resolved(dest_dir):
            continue
        seen.add(key)
        cues = _read_cues(path, size)
        if not cues:
            continue
        matches.append(path)
        if found is None:
            found = cues
    if found is None or not matches:
        return None
    dest = dest_named
    if dest.is_file() and _payload_size(dest) not in (None, size):
        dest = dest_dir / f"{src.name}.{size}.{lang}.cues.json"
    first = matches[0]
    if _resolved(dest) != _resolved(first):
        dest.write_bytes(first.read_bytes())
    for path in matches:
        try:
            path.unlink()
        except OSError:
            pass
    return found


def _payload_size(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    size = payload.get("size")
    return int(size) if isinstance(size, int) else None


def _resolved(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:
        return path


def _read_cues(path: Path, expected_size: int) -> list[tuple[float, float, str, str]] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("size") != expected_size:
        return None
    cues = payload.get("cues")
    if not isinstance(cues, list) or not cues:
        return None
    out: list[tuple[float, float, str, str]] = []
    for item in cues:
        if not isinstance(item, (list, tuple)) or len(item) != 4:
            return None
        t0, t1, japanese, translated = item
        out.append((float(t0), float(t1), str(japanese), str(translated)))
    return out
