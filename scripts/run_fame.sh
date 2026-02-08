#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${REPO_ROOT}/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "WARN:  .venv not found or Python missing. Run scripts/initial_setup.sh first." >&2
  exit 1
fi

export PYTHONPATH="$REPO_ROOT"
export FAME_BASE_DIR="${FAME_BASE_DIR:-$REPO_ROOT}"
export CHROMA_MODE="${CHROMA_MODE:-persistent}"

echo "==================== FAME Runner ===================="
echo "Repo        : $REPO_ROOT"
echo "Venv Python : $PYTHON"
echo "FAME_BASE_DIR: $FAME_BASE_DIR"
echo "CHROMA_MODE : $CHROMA_MODE"
echo "====================================================="

# 1) Preprocessing (ingestion + vectorization)
read -r -p "Run preprocessing (ingestion + vectorization) now? [Y/n]: " run_pre
if [[ -z "${run_pre}" || "${run_pre}" =~ ^[Yy] ]]; then
  echo "\n🧩 Running preprocessing_for_rag.py ..."
  "$PYTHON" scripts/preprocessing_for_rag.py || { echo "ERROR: Preprocessing failed"; exit 1; }
  echo "SUCCESS: Preprocessing finished"
else
  echo "⏩ Skipping preprocessing"
fi

# 2) Choose RAG vs Non-RAG
echo "\nChoose pipeline family:" 
echo "  1) RAG (retrieval-guided)"
echo "  2) Non-RAG (context-only)"
read -r -p "Select option [1-2]: " family

case "$family" in
  1)
    echo "\nRAG variants:" 
    echo "  1) Single-Stage RAG (ss-rgfm)"
    echo "  2) Iterative RAG (is-rgfm)"
    read -r -p "Select option [1-2]: " rag_choice
    case "$rag_choice" in
      1)
        echo "\n🚀 Running SS-RGFM (interactive)..."
        "$PYTHON" scripts/run_ss_rag.py --interactive
        ;;
      2)
        echo "\n🚀 Running IS-RGFM (interactive)..."
        "$PYTHON" scripts/run_is_rag.py --interactive
        ;;
      *) echo "Invalid RAG choice"; exit 1 ;;
    esac
    ;;
  2)
    echo "\nNon-RAG variants:" 
    echo "  1) Single-Stage Non-RAG (ss-nonrag)"
    echo "  2) Iterative Non-RAG (is-nonrag)"
    read -r -p "Select option [1-2]: " non_choice
    case "$non_choice" in
      1)
        echo "\n🚀 Running SS-NonRAG (interactive)..."
        "$PYTHON" scripts/run_ss_nonrag.py --interactive
        ;;
      2)
        echo "\n🚀 Running IS-NonRAG (interactive)..."
        "$PYTHON" scripts/run_is_nonrag.py --interactive
        ;;
      *) echo "Invalid Non-RAG choice"; exit 1 ;;
    esac
    ;;
  *) echo "Invalid family choice"; exit 1 ;;
esac

echo "\nSUCCESS: FAME run completed"
