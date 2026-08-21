#!/usr/bin/env bash

set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
LAUNCHER="$ROOT_DIR/Launch Auto RHEED.command"
DESKTOP_DIR="$HOME/Desktop"
APP_PATH="$DESKTOP_DIR/Auto RHEED.app"

if [ "$(uname -s)" != "Darwin" ]; then
  printf '%s\n' "This shortcut creator is for macOS." >&2
  exit 1
fi
if [ ! -f "$LAUNCHER" ]; then
  printf 'Could not find:\n  %s\n' "$LAUNCHER" >&2
  exit 1
fi

chmod +x "$ROOT_DIR/Launch Auto RHEED.sh" "$LAUNCHER"
mkdir -p "$DESKTOP_DIR"

if [ -e "$APP_PATH" ]; then
  BACKUP_PATH="$DESKTOP_DIR/Auto RHEED backup $(date +%Y%m%d-%H%M%S).app"
  mv "$APP_PATH" "$BACKUP_PATH"
  printf 'Existing shortcut moved to:\n  %s\n' "$BACKUP_PATH"
fi

# Escape the repository path for the generated AppleScript string.
APPLE_LAUNCHER=${LAUNCHER//\\/\\\\}
APPLE_LAUNCHER=${APPLE_LAUNCHER//\"/\\\"}

osacompile -o "$APP_PATH" \
  -e 'on run' \
  -e 'tell application "Terminal"' \
  -e 'activate' \
  -e "do script (quoted form of \"$APPLE_LAUNCHER\")" \
  -e 'end tell' \
  -e 'end run'

printf '\nCreated:\n  %s\n' "$APP_PATH"
printf '%s\n' "Double-click Auto RHEED on the Desktop to launch the server."
if [ -t 0 ]; then
  printf '\nPress Enter to close...'
  read -r _
fi
