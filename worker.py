"""Последовательный локальный транскриптор записей."""

import argparse
import fcntl
import hashlib
import json
import logging
from logging.handlers import RotatingFileHandler
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from datetime import datetime

import settings as config


def signature(path):
    stat = path.stat()
    return [stat.st_size, stat.st_mtime_ns]


def output_path(path, destination, state, sources):
    key = str(path)
    source_index = next((index for index, root in enumerate(sources)
                         if path.is_relative_to(root)), None)
    if source_index is None:
        raise ValueError(f"recording outside configured source roots: {path}")
    label = f" [{config.SOURCE_LABELS[source_index]}]"
    entry = state["files"].get(key, {})
    saved_name = entry.get("output") if entry.get("naming_version") == 2 else None
    if saved_name:
        return destination / saved_name
    stem = path.stem
    digest = hashlib.sha256(os.fsencode(key)).hexdigest()
    claimed = {entry.get("output") for other, entry in state["files"].items() if other != key}
    name_limit = os.pathconf(destination, "PC_NAME_MAX")

    def filename(suffix):
        budget = name_limit - len(os.fsencode(label + suffix + ".txt"))
        fitted = stem
        while fitted and len(os.fsencode(fitted)) > budget:
            fitted = fitted[:-1]
        if not fitted:
            raise OSError("filesystem filename limit is too small for a transcript")
        return fitted + label + suffix + ".txt"

    for name in [filename("")] + [filename(f"__{digest[:size]}") for size in (8, 12, 16, 64)]:
        if name not in claimed and not (destination / name).exists():
            return destination / name
    raise FileExistsError(f"no available transcript name for {path}")


def load_state(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data.get("files"), dict):
            raise ValueError("invalid state format")
        if data.get("version") == 1:
            completed = {}
            for key, value in data["files"].items():
                if value.get("status") != "done":
                    continue
                source = next((item for item in config.SOURCE_DIRS
                               if Path(key).is_relative_to(item)), None)
                if source is None:
                    continue
                digest = hashlib.sha256(os.fsencode(key)).hexdigest()[:12]
                # Preserve the old v1 filename solely to find an existing transcript.
                completed[key] = {**value, "output":
                                  f"{Path(key).stem[:50]}__{source.name}__{digest}.txt"}
            return {"version": 2, "files": completed, "baseline_sources": []}
        if data.get("version") != 2 or not isinstance(data.get("baseline_sources"), list):
            raise ValueError("invalid state format")
        return data
    except FileNotFoundError:
        return {"version": 2, "files": {}, "baseline_sources": []}


def save_state(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(state, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def format_time(seconds):
    value = max(0, int(round(float(seconds))))
    return f"{value // 3600:02}:{value // 60 % 60:02}:{value % 60:02}"


def format_transcript(path, result, now):
    lines = [f"Источник: {path}", f"Дата транскрибации: {now:%Y-%m-%d %H:%M}", ""]
    for segment in result["segments"]:
        speech = segment["text"].strip()
        if speech:
            lines.append(f"[{format_time(segment['start'])} - {format_time(segment['end'])}] {speech}")
    return "\n".join(lines) + "\n"


def format_speaker_transcript(path, result, turns, now):
    """Assign Whisper words to the turn with greatest overlap and group them."""
    clean_turns = sorted((float(t["start"]), float(t["end"]), str(t["speaker"]))
                         for t in turns if float(t["end"]) > float(t["start"]))
    if not clean_turns:
        raise ValueError("diarization returned no speaker turns")
    labels = {speaker: f"SPEAKER_{index:02d}" for index, speaker in
              enumerate(dict.fromkeys(item[2] for item in clean_turns))}
    words = [word for segment in result["segments"] for word in segment.get("words", [])
             if word.get("word", "").strip()]
    if not words:
        raise ValueError("Whisper returned no word timestamps")

    def speaker_for_word(start, end):
        overlaps = [(max(0.0, min(end, stop) - max(start, begin)),
                     -abs((start + end - begin - stop) / 2), speaker)
                    for begin, stop, speaker in clean_turns]
        overlap, _, speaker = max(overlaps, key=lambda item: (item[0], item[1], item[2]))
        if overlap > 0:
            return labels[speaker]
        # Keep text near a VAD boundary, but do not attribute words in long silence.
        nearest = min(clean_turns, key=lambda item: max(item[0] - end, start - item[1], 0))
        gap = max(nearest[0] - end, start - nearest[1], 0)
        return labels[nearest[2]] if gap <= 0.35 else None

    lines = [f"Источник: {path}", f"Дата транскрибации: {now:%Y-%m-%d %H:%M}",
             f"Модель: {config.MODEL_REPO}", f"Язык: {config.LANGUAGE}",
             f"Спикеров обнаружено: {len(labels)}", ""]
    groups = []
    for word in words:
        start, end = float(word["start"]), float(word["end"])
        if end < start:
            raise ValueError("invalid Whisper word timestamps")
        speaker = speaker_for_word(start, end)
        if groups and groups[-1][0] == speaker and start - groups[-1][2] <= 1.0:
            groups[-1][2] = max(groups[-1][2], end)
            groups[-1][3].append(word["word"])
        else:
            groups.append([speaker, start, end, [word["word"]]])
    for speaker, start, end, parts in groups:
        prefix = f"{speaker}: " if speaker else ""
        lines.append(f"[{format_time(start)} - {format_time(end)}] {prefix}{''.join(parts).strip()}")
    return "\n".join(lines) + "\n"


def ready(path, observation, now, stable_seconds):
    current = signature(path)
    if now - path.stat().st_mtime < stable_seconds:
        return False, current
    if observation is None or observation[0] != current:
        return False, current
    return now - observation[1] >= stable_seconds, current


def recordings(source):
    def onerror(error):
        raise error
    for root, dirs, files in os.walk(source, onerror=onerror):
        dirs[:] = sorted(d for d in dirs if not (Path(root) / d).is_symlink())
        for name in sorted(files):
            path = Path(root) / name
            if path.suffix.lower() in config.EXTENSIONS and not path.is_symlink():
                yield path


def has_audio(path, ffprobe):
    result = subprocess.run(
        [str(ffprobe), "-v", "error", "-select_streams", "a", "-show_entries",
         "stream=index", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, timeout=15, check=True,
    )
    if not result.stdout.strip():
        raise ValueError("audio stream not found")


def is_open(path):
    result = subprocess.run(["/usr/sbin/lsof", "-t", str(path)],
                            capture_output=True, text=True, timeout=30)
    if result.returncode not in (0, 1):
        raise OSError(f"lsof failed with exit code {result.returncode}")
    return result.returncode == 0


def transcribe(path, model_dir):
    import mlx_whisper

    os.environ["PATH"] = str(config.FFMPEG.parent) + os.pathsep + os.environ.get("PATH", "")
    return mlx_whisper.transcribe(
        str(path), path_or_hf_repo=str(model_dir), word_timestamps=True,
        verbose=False, language=config.LANGUAGE,
        hallucination_silence_threshold=2.0,
    )


def diarize(path):
    probe = subprocess.run(
        [str(config.FFPROBE), "-v", "error", "-select_streams", "a:0",
         "-show_entries", "format=duration:stream=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True, timeout=15,
    )
    durations = []
    for value in probe.stdout.splitlines():
        try:
            duration = float(value)
        except ValueError:
            continue
        if math.isfinite(duration) and duration > 0:
            durations.append(duration)
    if not durations:
        raise ValueError("ffprobe did not return audio duration for diarization")
    timeout = max(config.DIARIZATION_TIMEOUT_MIN_SECONDS,
                  max(durations) * config.DIARIZATION_TIMEOUT_MULTIPLIER)
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    env["PYANNOTE_METRICS_ENABLED"] = "0"
    ffmpeg_lib = str(config.FFMPEG.parent.parent / "lib")
    env["DYLD_LIBRARY_PATH"] = ffmpeg_lib + os.pathsep + env.get("DYLD_LIBRARY_PATH", "")
    result = subprocess.run(
        [str(config.DIARIZATION_PYTHON), str(Path(__file__).with_name("diarize.py")),
         str(config.DIARIZATION_MODEL_DIR), str(path)],
        capture_output=True, text=True, check=True, env=env, timeout=timeout,
    )
    return json.loads(result.stdout)


def baseline_source(source, state, state_file, now, stable_seconds):
    additions = {}
    for path in recordings(source):
        try:
            sig = signature(path)
            if now - path.stat().st_mtime < stable_seconds or is_open(path):
                continue
            if signature(path) == sig:
                key = str(path)
                entry = state["files"].get(key, {})
                if entry.get("status") != "done" or entry.get("signature") != sig:
                    additions[key] = {"signature": sig, "status": "baseline"}
        except FileNotFoundError:
            continue
    state["files"].update(additions)
    state["baseline_sources"].append(str(source))
    save_state(state_file, state)
    logging.info("Baseline источника готов: %s; пропущено %d завершённых записей", source, len(additions))


def scan_once(state, observations, *, sources=config.SOURCE_DIRS,
              destination=None, volume=None,
              state_file=None, model_dir=None,
              ffprobe=None, transcriber=transcribe, diarizer=diarize,
              clock=time.time, stable_seconds=None,
              retry_seconds=None, backfill=False):
    """Один проход; возвращает число успешно обработанных файлов."""
    sources = sources or config.SOURCE_DIRS
    destination = destination or config.DEST_DIR
    state_file = state_file or config.STATE_FILE
    model_dir = model_dir or config.MODEL_DIR
    ffprobe = ffprobe or config.FFPROBE
    stable_seconds = stable_seconds if stable_seconds is not None else config.STABLE_SECONDS
    retry_seconds = retry_seconds if retry_seconds is not None else config.RETRY_SECONDS

    def available(path):
        if volume is not None and (path == volume or volume in path.parents):
            return volume.is_mount()
        return config.path_available(path)

    destination_ready = available(destination) and model_dir.is_dir() and config.FFMPEG.is_file()
    if destination_ready:
        try:
            destination.mkdir(parents=True, exist_ok=True)
        except OSError:
            logging.exception("Папка транскрипций недоступна")
            destination_ready = False
    now = clock()
    found = set()
    processed = 0
    for source in sources:
        if not available(source):
            logging.warning("Источник недоступен: %s", source)
            continue
        try:
            if not source.is_dir():
                logging.warning("Источник недоступен: %s", source)
                continue
            if str(source) not in state["baseline_sources"]:
                if backfill:
                    state["baseline_sources"].append(str(source))
                    save_state(state_file, state)
                else:
                    baseline_source(source, state, state_file, now, stable_seconds)
                    continue
            for path in recordings(source):
                key = str(path)
                found.add(key)
                entry = state["files"].get(key, {})
                try:
                    if backfill:
                        sig = signature(path)
                        is_ready = now - path.stat().st_mtime >= stable_seconds
                    else:
                        is_ready, sig = ready(path, observations.get(key), now, stable_seconds)
                    if observations.get(key, (None,))[0] != sig:
                        observations[key] = (sig, now, 0, 0)
                    if not is_ready:
                        continue
                    if not backfill and entry.get("signature") == sig and entry.get("status") == "baseline":
                        continue
                    if (entry.get("signature") == sig and entry.get("status") == "done"
                            and (destination / entry.get("output", "")).is_file()):
                        continue
                    if (entry.get("signature") == sig and entry.get("status") == "failed"
                            and (entry.get("reason") == "ffprobe" or
                                 now - entry.get("attempted_at", 0) < retry_seconds)):
                        continue
                    if not destination_ready or not available(destination) or not available(path):
                        continue
                    if is_open(path):
                        continue
                    failures, last_probe = observations[key][2:]
                    if entry.get("status") == "probe_pending" and entry.get("signature") == sig:
                        failures = max(failures, entry.get("attempts", 0))
                        last_probe = max(last_probe, entry.get("last_probe", 0))
                    if failures and now - last_probe < 60:
                        continue
                    try:
                        has_audio(path, ffprobe)
                    except (OSError, ValueError, subprocess.CalledProcessError,
                            subprocess.TimeoutExpired) as error:
                        if not available(destination) or not available(path) or not destination.is_dir() or not path.exists():
                            logging.warning("ffprobe отложен: источник или destination недоступен: %s", path)
                            continue
                        if signature(path) != sig:
                            observations[key] = (signature(path), now, 0, 0)
                            continue
                        failures += 1
                        observations[key] = (sig, observations[key][1], failures, now)
                        if failures < 3:
                            state["files"][key] = {"signature": sig, "status": "probe_pending",
                                                   "attempts": failures, "last_probe": now}
                            save_state(state_file, state)
                            logging.warning("ffprobe не открыл запись (попытка %d/3), повтор через 60 с: %s: %s",
                                            failures, path, error)
                            continue
                        logging.error("ffprobe не открыл неизменную запись после 3 попыток: %s: %s",
                                      path, error)
                        state["files"][key] = {"signature": sig, "status": "failed",
                                               "reason": "ffprobe", "attempted_at": now}
                        save_state(state_file, state)
                        continue
                    if signature(path) != sig:
                        observations[key] = (signature(path), now, 0, 0)
                        continue
                    result = transcriber(path, model_dir)
                    timestamp = datetime.now()
                    if config.DIARIZATION_ENABLED:
                        try:
                            content = format_speaker_transcript(path, result,
                                                                diarizer(path), timestamp)
                        except Exception as error:
                            logging.warning("diarization failed; transcript saved without speaker labels: %s: %s",
                                            path, error)
                            content = format_transcript(path, result, timestamp)
                    else:
                        content = format_transcript(path, result, timestamp)
                except Exception:
                    if not available(destination) or not available(path) or not destination.is_dir() or not path.exists():
                        logging.warning("Источник или destination временно недоступен: %s", path)
                        continue
                    logging.exception("Ошибка записи или транскрибации: %s", path)
                    try:
                        state["files"][key] = {"signature": signature(path),
                                               "status": "failed", "attempted_at": now,
                                               **({"output": entry["output"]} if entry.get("output") else {})}
                        save_state(state_file, state)
                    except OSError:
                        logging.exception("Не удалось сохранить состояние: %s", path)
                    continue
                temporary = None
                try:
                    if signature(path) != sig or not available(destination) or not available(path):
                        observations.pop(key, None)
                        continue
                    target = output_path(path, destination, state, sources)
                    state["files"][key] = {"signature": sig, "status": "pending_output",
                                           "output": target.name, "naming_version": 2}
                    save_state(state_file, state)
                    with tempfile.NamedTemporaryFile(
                            mode="w", encoding="utf-8", prefix=".transcriber-",
                            suffix=".tmp", dir=destination, delete=False) as stream:
                        temporary = Path(stream.name)
                        stream.write(content)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(temporary, target)
                    state["files"][key]["status"] = "done"
                    save_state(state_file, state)
                    processed += 1
                    logging.info("Готово: %s -> %s", path, target)
                except OSError:
                    logging.exception("Публикация ожидает доступный destination: %s", path)
                    destination_ready = False
                finally:
                    if temporary is not None:
                        try:
                            temporary.unlink(missing_ok=True)
                        except OSError:
                            logging.exception("Не удалось удалить временный TXT: %s", temporary)
        except OSError:
            logging.exception("Не удалось прочитать источник: %s", source)
            continue
    for key in list(observations):
        if key not in found:
            del observations[key]
    logging.info("Проход завершён: найдено %d, готово %d", len(found), processed)
    return processed


def setup_logging():
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[RotatingFileHandler(config.LOG_FILE, maxBytes=5_000_000,
                                                      backupCount=3, encoding="utf-8")])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="один проход по папкам")
    parser.add_argument("--backfill", action="store_true", help="явно обработать старые baseline-записи")
    parser.add_argument("--config", type=Path, help="путь к config.toml")
    args = parser.parse_args()
    if args.backfill and not args.once:
        parser.error("--backfill requires --once")
    try:
        config.configure(args.config)
    except config.ConfigError as error:
        parser.error(str(error))
    setup_logging()
    lock = (config.DATA_DIR / "worker.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        logging.error("Другой worker уже работает")
        return 1
    try:
        state = load_state(config.STATE_FILE)
    except (ValueError, json.JSONDecodeError):
        logging.exception("Повреждён state.json; сохраните/исправьте файл и перезапустите")
        return 1
    observations = {}
    logging.info("Worker запущен")
    while True:
        try:
            scan_once(state, observations, backfill=args.backfill)
        except Exception:
            logging.exception("Ошибка прохода; worker продолжит работу")
        if args.once:
            return 0
        time.sleep(config.POLL_SECONDS)


if __name__ == "__main__":
    sys.exit(main())
