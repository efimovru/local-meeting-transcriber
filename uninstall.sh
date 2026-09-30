#!/bin/bash
set -euo pipefail
data_dir="${LOCAL_MEETING_TRANSCRIBER_HOME:-$HOME/Library/Application Support/local-meeting-transcriber}"
purge=0
if [[ "${1:-}" == --purge ]]; then purge=1
elif (($#)); then echo "Usage: $0 [--purge]" >&2; exit 2
fi
if [[ "$data_dir" != /* || "$data_dir" == / || "$data_dir" == "$HOME" ]]; then
  echo "Refusing unsafe data directory." >&2; exit 2
fi
label="gui/$(id -u)/local.transcriber.worker"
launchctl bootout "$label" >/dev/null 2>&1 || true
rm -f "$HOME/Library/LaunchAgents/local.transcriber.worker.plist"
rm -rf "$data_dir/app" "$data_dir/venv"
if (( purge )); then
  echo "PURGE: removing config, state, models, and logs from $data_dir"
  rm -rf "$data_dir"
else
  echo "Application removed. Config, state, models, and logs preserved in $data_dir"
fi
