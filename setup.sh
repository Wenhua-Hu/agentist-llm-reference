#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# Linux + NVIDIA driver + Python 3.12 are platform prerequisites. No Docker/systemd.
exec "${BOOTSTRAP_PYTHON:-python3}" scripts/native_setup.py "$@"
