#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

cd "$REPO_ROOT"

OS_NAME="$(uname -s || true)"
case "$OS_NAME" in
  Linux*|Darwin*)
    ./scripts/bootstrap.sh
    # Activate venv for the remainder of this script (won't persist after exit)
    if [[ -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
      source "${REPO_ROOT}/.venv/bin/activate"
      expected="${REPO_ROOT}/.venv"
      if [[ -n "${VIRTUAL_ENV:-}" ]]; then
        expected_resolved="$(cd "${expected}" && pwd -P)"
        actual_resolved="$(cd "${VIRTUAL_ENV}" && pwd -P)"
        if [[ "${actual_resolved}" == "${expected_resolved}" ]]; then
          echo "✅ Venv activated: ${VIRTUAL_ENV}"
        else
          echo "⚠️  Venv activation mismatch."
          echo "   Expected: ${expected}"
          echo "   Actual  : ${VIRTUAL_ENV}"
        fi
      else
        echo "⚠️  Venv activation not detected. Try: source ${REPO_ROOT}/.venv/bin/activate"
      fi
    fi
    ;;
  MINGW*|MSYS*|CYGWIN*)
    if command -v powershell >/dev/null 2>&1; then
      powershell -ExecutionPolicy Bypass -File "scripts/bootstrap.ps1"
    else
      echo "⚠️  PowerShell not found. Run scripts/bootstrap.ps1 manually in PowerShell."
      exit 1
    fi
    ;;
  *)
    echo "⚠️  Unknown OS '$OS_NAME'. Run scripts/bootstrap.sh (macOS/Linux) or scripts/bootstrap.ps1 (Windows) manually."
    exit 1
    ;;
esac

echo "✅ Initial setup complete."
