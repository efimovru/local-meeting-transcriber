# Local Meeting Transcriber

A local macOS service that watches recording folders, transcribes speech with MLX Whisper, identifies speaker turns with pyannote Community-1, and writes one timestamped TXT per recording. After the models are downloaded, audio processing stays on your Mac. No paid API is used.

## Features

- Apple Silicon and MLX; Russian by default, with configurable language.
- Any number of watched folders, each with its own label; internal or external volumes.
- Word timestamps, speech activity segmentation (VAD), automatic speaker count, and speaker-aware replies.
- Plain transcript fallback if diarization fails or times out. Speaker IDs apply only within one recording.
- Autostart through launchd; completed files detected by size, mtime, and ffprobe.

## Requirements

- macOS 14 or newer on Apple Silicon, with 16 GB RAM recommended.
- Native Python 3.12–3.14; Python 3.14 is recommended and used for the full local integration test. The installer finds a compatible interpreter or accepts `PYTHON=/path/to/python3`.
- `ffmpeg` and `ffprobe` on PATH (`brew install ffmpeg`), or absolute paths in `[tools]`.
- A free Hugging Face account for the one-time download of `pyannote/speaker-diarization-community-1`.

The software is MIT licensed. The MLX Whisper and pyannote models have their own licenses and access terms; their weights are not in this repository.

## Quick start

```sh
git clone https://github.com/efimovru/local-meeting-transcriber.git
cd local-meeting-transcriber
cp config.example.toml config.toml
# Edit config.toml to use your recording and output folders.
./install.sh
./service.sh status
```

The installer validates macOS, Apple Silicon, Python, FFmpeg, and the config; creates a local venv; installs dependencies and code; downloads missing models; and starts the LaunchAgent. Rerunning it preserves the installed config, state, and models. Nothing is transcribed from existing completed files on the first normal scan: they become the baseline. New recordings and baseline recordings changed afterward are eligible.

Before the first model download, [accept the Community-1 conditions](https://huggingface.co/pyannote/speaker-diarization-community-1) and create a Hugging Face read token. If the installer stops with a 401/403 download error, authorize locally with `~/Library/Application\ Support/local-meeting-transcriber/venv/bin/hf auth login`, then run `./install.sh` again. The token is never placed in `config.toml` or the LaunchAgent; the installed agent sets `HF_HUB_OFFLINE=1`.

## Configuration

Edit the untracked `config.toml`. For example:

```toml
[[sources]]
path = "~/Documents/Meetings"
label = "Meetings"

[[sources]]
path = "/Volumes/ExternalDrive/OBS"
label = "OBS"

[destination]
path = "~/Documents/Transcripts"

[transcription]
language = "ru"

[runtime]
poll_seconds = 30
stable_seconds = 90
diarization_enabled = true
```

Add or remove `[[sources]]` blocks freely. `~`, spaces, Unicode, and absolute macOS paths are supported. Results are named `<stem> [label].txt`; collisions get a deterministic suffix. The destination cannot be inside a watched source. Optional `[tools]` paths override PATH lookup for FFmpeg and ffprobe. For manual testing, use `worker.py --config /absolute/path/config.toml --once`.

After installation, edit the installed runtime `config.toml` and restart the service. If you keep an edited `config.toml` in the clone, apply it explicitly with `./install.sh --config config.toml`; an ordinary reinstall preserves the installed config.

The default runtime directory is `~/Library/Application Support/local-meeting-transcriber/`; it contains the installed app, venv, config, state, logs, and models. Set `LOCAL_MEETING_TRANSCRIBER_HOME` or use `./install.sh --data-dir /absolute/path` for another directory.

## Operation

```sh
./service.sh status
./service.sh logs
./service.sh stop
./service.sh start
./service.sh restart
```

Normal scans run every 30 seconds. A file must keep both size and mtime unchanged for at least 90 seconds and have a readable audio stream. Missing external volumes are never created as fake folders; unavailable sources are skipped, and an unavailable destination leaves work pending.

To explicitly transcribe old baseline recordings, stop the agent first, then run:

```sh
./service.sh stop
"$HOME/Library/Application Support/local-meeting-transcriber/venv/bin/python" \
  "$HOME/Library/Application Support/local-meeting-transcriber/app/worker.py" --backfill --once
./service.sh start
```

Backfill is never automatic, preserves state, and skips recordings that already have a completed TXT. Open or recently modified files still wait.

Example output:

```text
Источник: /path/to/meeting.mp4
Спикеров обнаружено: 2

[00:01:02 - 00:01:06] SPEAKER_00: Доброе утро.
[00:01:07 - 00:01:12] SPEAKER_01: Начнём обсуждение.
```

## Permissions and troubleshooting

macOS may ask the installed Python for **Files & Folders** access to Documents or removable volumes. Allow only the relevant categories; Full Disk Access is usually unnecessary. Names and prompts can vary by macOS version.

If the service is not running, check `./service.sh status` and `./service.sh logs`. If FFmpeg is missing, run `brew install ffmpeg` or set `[tools]`. For Hugging Face 401/403, accept the model conditions, log in locally, and retry installation. If an external volume is offline, reconnect it; the worker resumes on its next scan. Logs are in the runtime directory.

Uninstall with `./uninstall.sh`; this preserves config, state, models, and logs. `./uninstall.sh --purge` explicitly removes those data too.
