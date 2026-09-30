"""Validated user configuration for the local transcriber."""

from pathlib import Path
import os
import shutil
import tomllib


class ConfigError(ValueError):
    pass


DATA_DIR = Path(os.environ.get("LOCAL_MEETING_TRANSCRIBER_HOME",
                               Path.home() / "Library/Application Support/local-meeting-transcriber")).expanduser()
CONFIG_FILE = DATA_DIR / "config.toml"
MODEL_DIR = DATA_DIR / "model"
DIARIZATION_MODEL_DIR = DATA_DIR / "diarization-model"
DIARIZATION_PYTHON = DATA_DIR / "venv/bin/python"
STATE_FILE = DATA_DIR / "state.json"
LOG_FILE = DATA_DIR / "worker.log"
SOURCE_DIRS = ()
SOURCE_LABELS = ()
DEST_DIR = None
FFMPEG = None
FFPROBE = None
LANGUAGE = "ru"
DIARIZATION_ENABLED = True
POLL_SECONDS = 30
STABLE_SECONDS = 90
RETRY_SECONDS = 3600
DIARIZATION_TIMEOUT_MIN_SECONDS = 600
DIARIZATION_TIMEOUT_MULTIPLIER = 3
MODEL_REPO = "mlx-community/whisper-large-v3-turbo"
DIARIZATION_REPO = "pyannote/speaker-diarization-community-1"
EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".m4a", ".mp3", ".wav"}


def _path(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name}: path must be a non-empty string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ConfigError(f"{name}: path must be absolute or start with ~")
    return path


def external_volume(path):
    """Return /Volumes/<name> for an external path, or None for local paths."""
    parts = path.parts
    if len(parts) >= 3 and parts[1] == "Volumes":
        return Path("/Volumes") / parts[2]
    return None


def path_available(path):
    volume = external_volume(path)
    return volume is None or volume.is_mount()


def _tool(raw, key):
    value = raw.get(key)
    if value is None:
        value = shutil.which(key)
    path = _path(value, f"tools.{key}") if value else None
    if path is None or not path.is_file():
        raise ConfigError(f"{key} not found; install ffmpeg or set [tools].{key} in config.toml")
    return path


def load(path):
    try:
        with Path(path).open("rb") as stream:
            data = tomllib.load(stream)
    except FileNotFoundError as error:
        raise ConfigError(f"config not found: {path}; copy config.example.toml and edit it") from error
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"invalid TOML in {path}: {error}") from error
    sources = data.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ConfigError("at least one [[sources]] entry is required")
    paths, labels = [], []
    for index, item in enumerate(sources, 1):
        if not isinstance(item, dict):
            raise ConfigError(f"sources #{index}: expected a table")
        source = _path(item.get("path"), f"sources #{index}")
        label = item.get("label")
        if not isinstance(label, str) or not label.strip():
            raise ConfigError(f"sources #{index}: label must be non-empty")
        if any(character in label for character in "/\\:"):
            raise ConfigError(f"sources #{index}: label cannot contain /, \\, or :")
        for existing in paths:
            if (source == existing or source.is_relative_to(existing)
                    or existing.is_relative_to(source)):
                raise ConfigError(f"source directories must not overlap: {existing} and {source}")
        paths.append(source)
        labels.append(label.strip())
    destination = data.get("destination")
    if not isinstance(destination, dict):
        raise ConfigError("[destination] with path is required")
    target = _path(destination.get("path"), "destination")
    if any(target == source or target.is_relative_to(source) for source in paths):
        raise ConfigError("destination must not be inside a monitored source")
    transcription = data.get("transcription", {})
    runtime = data.get("runtime", {})
    tools = data.get("tools", {})
    if not all(isinstance(item, dict) for item in (transcription, runtime, tools)):
        raise ConfigError("[transcription], [runtime], and [tools] must be tables")
    language = transcription.get("language", "ru")
    if not isinstance(language, str) or not language.strip():
        raise ConfigError("transcription.language must be non-empty")
    poll = runtime.get("poll_seconds", 30)
    stable = runtime.get("stable_seconds", 90)
    enabled = runtime.get("diarization_enabled", True)
    if type(poll) is not int or poll <= 0 or type(stable) is not int or stable <= 0:
        raise ConfigError("runtime poll_seconds and stable_seconds must be positive integers")
    if type(enabled) is not bool:
        raise ConfigError("runtime.diarization_enabled must be a boolean")
    return {"sources": tuple(paths), "labels": tuple(labels), "destination": target,
            "language": language.strip(), "poll": poll, "stable": stable,
            "diarization_enabled": enabled, "ffmpeg": _tool(tools, "ffmpeg"),
            "ffprobe": _tool(tools, "ffprobe")}


def configure(path=None):
    values = load(path or CONFIG_FILE)
    global SOURCE_DIRS, SOURCE_LABELS, DEST_DIR, LANGUAGE, POLL_SECONDS
    global STABLE_SECONDS, DIARIZATION_ENABLED, FFMPEG, FFPROBE
    SOURCE_DIRS = values["sources"]
    SOURCE_LABELS = values["labels"]
    DEST_DIR = values["destination"]
    LANGUAGE = values["language"]
    POLL_SECONDS = values["poll"]
    STABLE_SECONDS = values["stable"]
    DIARIZATION_ENABLED = values["diarization_enabled"]
    FFMPEG = values["ffmpeg"]
    FFPROBE = values["ffprobe"]
    return values
