#!/usr/bin/env bash

# Portable macOS/Linux launcher. Keep this process attached to the terminal so
# closing the terminal also stops the local Auto RHEED server.
set -u

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
APP_PORT=${PORT:-5000}
APP_URL="http://127.0.0.1:${APP_PORT}/"

find_uv() {
  if command -v uv >/dev/null 2>&1; then
    command -v uv
    return 0
  fi
  for candidate in \
    "$HOME/.local/bin/uv" \
    "/opt/homebrew/bin/uv" \
    "/usr/local/bin/uv"
  do
    if [ -x "$candidate" ]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

UV_BIN=$(find_uv) || {
  printf '%s\n' "Could not find uv." >&2
  printf '%s\n' "Install it from https://docs.astral.sh/uv/ and run uv sync in:" >&2
  printf '  %s\n' "$ROOT_DIR" >&2
  if [ -t 0 ]; then
    printf '\nPress Enter to close...'
    read -r _
  fi
  exit 1
}

if [ ! -f "$ROOT_DIR/pyproject.toml" ] || [ ! -f "$ROOT_DIR/uv.lock" ]; then
  printf 'Auto RHEED project files were not found in:\n  %s\n' "$ROOT_DIR" >&2
  exit 1
fi

export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

printf '%s\n' "=============================================================="
printf '%s\n' "   Auto RHEED is starting..."
printf '\n'
printf '   A browser tab will open automatically in a few seconds.\n'
printf '   If it does not, open:  %s\n' "$APP_URL"
printf '\n'
printf '%s\n' "   KEEP THIS WINDOW OPEN while using the app."
printf '%s\n' "   Close this window or press Ctrl+C to stop the server."
printf '%s\n' "=============================================================="
printf '\n'

(
  sleep 4
  case "$(uname -s)" in
    Darwin)
      open "$APP_URL" >/dev/null 2>&1 || true
      ;;
    Linux)
      if command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$APP_URL" >/dev/null 2>&1 || true
      elif command -v gio >/dev/null 2>&1; then
        gio open "$APP_URL" >/dev/null 2>&1 || true
      fi
      ;;
  esac
) &

cd "$ROOT_DIR" || exit 1
"$UV_BIN" run --frozen --no-sync python -m rheed_webapp.app
SERVER_STATUS=$?

printf '\nAuto RHEED has stopped (exit status %s).\n' "$SERVER_STATUS"
if [ -t 0 ]; then
  printf 'Press Enter to close...'
  read -r _
fi
exit "$SERVER_STATUS"
