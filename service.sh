#!/bin/bash
set -euo pipefail
data_dir="${LOCAL_MEETING_TRANSCRIBER_HOME:-$HOME/Library/Application Support/local-meeting-transcriber}"
label="gui/$(id -u)/local.transcriber.worker"
plist="$HOME/Library/LaunchAgents/local.transcriber.worker.plist"
case "${1:-}" in
  status) launchctl print "$label" ;;
  start) launchctl bootstrap "gui/$(id -u)" "$plist" ;;
  stop) launchctl bootout "$label" ;;
  restart) launchctl kickstart -k "$label" ;;
  logs) tail -f "$data_dir/worker.log" ;;
  *) echo "Usage: $0 {status|start|stop|restart|logs}" >&2; exit 2 ;;
esac
