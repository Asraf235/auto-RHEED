#!/usr/bin/env bash

# Double-clickable macOS wrapper around the shared Unix launcher.
ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec /bin/bash "$ROOT_DIR/Launch Auto RHEED.sh"
