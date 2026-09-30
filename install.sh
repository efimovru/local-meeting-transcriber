#!/bin/bash
set -euo pipefail

project_dir="$(cd "$(dirname "$0")" && pwd)"
data_dir="${LOCAL_MEETING_TRANSCRIBER_HOME:-$HOME/Library/Application Support/local-meeting-transcriber}"
config_source=""
no_start=0
while (($#)); do
  case "$1" in
    --data-dir) data_dir="$2"; shift 2 ;;
    --config) config_source="$2"; shift 2 ;;
    --no-start) no_start=1; shift ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  echo "This installer requires macOS on Apple Silicon." >&2; exit 2
fi
macos_major="$(sw_vers -productVersion | cut -d. -f1)"
if (( macos_major < 14 )); then
  echo "MLX requires macOS 14 or newer." >&2; exit 2
fi
if [[ "$data_dir" != /* || "$data_dir" == / || "$data_dir" == "$HOME" ]]; then
  echo "--data-dir must be a safe absolute directory." >&2; exit 2
fi

python=""
for candidate in "${PYTHON:-}" python3.14 python3.13 python3.12 python3; do
  [[ -n "$candidate" ]] || continue
  command -v "$candidate" >/dev/null 2>&1 || continue
  if "$candidate" -c 'import sys,platform; assert (3, 12) <= sys.version_info[:2] < (3, 15) and platform.machine() == "arm64"' >/dev/null 2>&1; then
    python="$(command -v "$candidate")"; break
  fi
done
if [[ -z "$python" ]]; then
  echo "Install native Python 3.12–3.14 (3.14 recommended), then rerun. Set PYTHON=/path/to/python3 if needed." >&2
  exit 2
fi
if [[ -z "$config_source" ]]; then
  if [[ -f "$data_dir/config.toml" ]]; then
    config_source="$data_dir/config.toml"
  else
    config_source="$project_dir/config.toml"
  fi
fi
if [[ ! -f "$config_source" ]]; then
  echo "Create config.toml from config.example.toml and edit its paths before installation." >&2
  exit 2
fi
export LOCAL_MEETING_TRANSCRIBER_HOME="$data_dir"
"$python" - "$project_dir" "$config_source" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from settings import ConfigError, load
try:
    load(sys.argv[2])
except ConfigError as error:
    raise SystemExit(f"Invalid config: {error}")
PY

mkdir -p "$data_dir" "$data_dir/app" "$data_dir/model" "$data_dir/diarization-model"
if [[ "$config_source" != "$data_dir/config.toml" ]]; then
  cp "$config_source" "$data_dir/config.toml"
fi
if [[ ! -x "$data_dir/venv/bin/python" ]]; then
  "$python" -m venv "$data_dir/venv"
fi
"$data_dir/venv/bin/python" -m pip install -r "$project_dir/requirements.txt"

# Existing complete model directories are never downloaded again or replaced.
if [[ ! -f "$data_dir/model/weights.safetensors" || ! -f "$data_dir/model/config.json" || \
      ! -f "$data_dir/diarization-model/config.yaml" || \
      ! -f "$data_dir/diarization-model/segmentation/pytorch_model.bin" || \
      ! -f "$data_dir/diarization-model/embedding/pytorch_model.bin" || \
      ! -f "$data_dir/diarization-model/plda/plda.npz" || \
      ! -f "$data_dir/diarization-model/plda/xvec_transform.npz" ]]; then
  if ! "$data_dir/venv/bin/python" - "$data_dir" <<'PY'
from pathlib import Path
import sys
from huggingface_hub import snapshot_download
root = Path(sys.argv[1])
asr = root / "model"
diarization = root / "diarization-model"
try:
    if not (asr / "weights.safetensors").is_file() or not (asr / "config.json").is_file():
        snapshot_download(repo_id="mlx-community/whisper-large-v3-turbo", local_dir=asr)
    if not all((diarization / item).is_file() for item in
               ("config.yaml", "segmentation/pytorch_model.bin", "embedding/pytorch_model.bin",
                "plda/plda.npz", "plda/xvec_transform.npz")):
        snapshot_download(repo_id="pyannote/speaker-diarization-community-1",
                          local_dir=diarization)
except Exception:
    raise SystemExit("Model download failed. For Community-1, accept its Hugging Face terms and run 'hf auth login' with a read token, then rerun ./install.sh.")
PY
  then
    exit 1
  fi
fi

cp "$project_dir/worker.py" "$project_dir/diarize.py" "$project_dir/settings.py" "$data_dir/app/"
plist_target="$data_dir/local.transcriber.worker.plist"
if (( ! no_start )); then
  mkdir -p "$HOME/Library/LaunchAgents"
  plist_target="$HOME/Library/LaunchAgents/local.transcriber.worker.plist"
fi
"$python" - "$project_dir/local.transcriber.worker.plist" "$plist_target" "$data_dir" <<'PY'
import plistlib
from pathlib import Path
import sys
template, output, data_dir = map(Path, sys.argv[1:])
with template.open("rb") as stream:
    plist = plistlib.load(stream)
def expand(value):
    if isinstance(value, str):
        return value.replace("__APP_DIR__", str(data_dir))
    if isinstance(value, list):
        return [expand(item) for item in value]
    if isinstance(value, dict):
        return {key: expand(item) for key, item in value.items()}
    return value
with output.open("wb") as stream:
    plistlib.dump(expand(plist), stream)
PY
plutil -lint "$plist_target" >/dev/null
if (( ! no_start )); then
  label="gui/$(id -u)/local.transcriber.worker"
  launchctl bootout "$label" >/dev/null 2>&1 || true
  launchctl bootstrap "gui/$(id -u)" "$plist_target"
  launchctl kickstart -k "$label"
fi
echo "Installed in $data_dir"
if (( no_start )); then echo "LaunchAgent was not started (--no-start)."; fi
