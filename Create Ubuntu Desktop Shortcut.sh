#!/usr/bin/env bash

set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
LAUNCHER="$ROOT_DIR/Launch Auto RHEED.sh"

if [ "$(uname -s)" != "Linux" ]; then
  printf '%s\n' "This shortcut creator is for Linux desktops such as Ubuntu." >&2
  exit 1
fi
if [ ! -f "$LAUNCHER" ]; then
  printf 'Could not find:\n  %s\n' "$LAUNCHER" >&2
  exit 1
fi

if command -v xdg-user-dir >/dev/null 2>&1 &&
   DESKTOP_DIR=$(xdg-user-dir DESKTOP 2>/dev/null); then
  :
else
  DESKTOP_DIR="$HOME/Desktop"
fi
if [ -z "$DESKTOP_DIR" ]; then
  DESKTOP_DIR="$HOME/Desktop"
fi

mkdir -p "$DESKTOP_DIR"
chmod +x "$LAUNCHER"

# Freedesktop Exec values use double quotes; escape the characters that retain
# special meaning inside them. Newlines are rejected because desktop entries
# are line-oriented files.
case "$LAUNCHER$ROOT_DIR" in
  *$'\n'*)
    printf '%s\n' "The repository path cannot contain a newline." >&2
    exit 1
    ;;
esac
escape_exec() {
  printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g; s/`/\\`/g; s/\$/\\$/g'
}
EXEC_LAUNCHER=$(escape_exec "$LAUNCHER")
ICON_PATH=$(escape_exec "$ROOT_DIR/auto_rheed.ico")
DESKTOP_FILE="$DESKTOP_DIR/Auto RHEED.desktop"

{
  printf '%s\n' '[Desktop Entry]'
  printf '%s\n' 'Version=1.0'
  printf '%s\n' 'Type=Application'
  printf '%s\n' 'Name=Auto RHEED'
  printf '%s\n' 'Comment=Launch the local Auto RHEED analysis app'
  printf 'Exec=/bin/bash "%s"\n' "$EXEC_LAUNCHER"
  printf 'Path=%s\n' "$ROOT_DIR"
  printf 'Icon=%s\n' "$ICON_PATH"
  printf '%s\n' 'Terminal=true'
  printf '%s\n' 'Categories=Science;Education;'
  printf '%s\n' 'StartupNotify=true'
} > "$DESKTOP_FILE"

chmod +x "$DESKTOP_FILE"
if command -v gio >/dev/null 2>&1; then
  gio set "$DESKTOP_FILE" metadata::trusted true >/dev/null 2>&1 || true
fi

printf '\nCreated:\n  %s\n' "$DESKTOP_FILE"
printf '%s\n' "If Ubuntu shows an untrusted-launcher warning, right-click it and choose Allow Launching."
