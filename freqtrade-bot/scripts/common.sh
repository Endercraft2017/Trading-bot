#!/bin/bash
# common.sh - Shared utilities for PhantomBot scripts
# Source this from other scripts: source "$(dirname "$0")/common.sh"

set -euo pipefail

# Project root (one level up from scripts/)
BOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Load environment variables from .env
if [ -f "$BOT_DIR/.env" ]; then
    set -a
    source "$BOT_DIR/.env"
    set +a
else
    echo "[WARNING] .env file not found at $BOT_DIR/.env"
fi
