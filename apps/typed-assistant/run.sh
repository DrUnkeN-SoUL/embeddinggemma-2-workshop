#!/usr/bin/env bash
# Type-safe on-device assistant: EmbeddingGemma 2 on the laptop CPU via LiteRT-LM.
# Usage: ./run.sh            # serve http://127.0.0.1:8091
#        ./run.sh --eval     # tool selection / exact args / clarifying questions on tests.json
set -euo pipefail
cd "$(dirname "$0")"

PY=../../demo/.venv/bin/python
MODEL=../../demo/models/embeddinggemma-2-text-270m.litertlm
PORT=${PORT:-8091}

if [ ! -x "$PY" ]; then
  echo "No venv at demo/.venv. Run:  python3 -m venv demo/.venv && demo/.venv/bin/pip install litert-lm==0.18.0 numpy" >&2
  exit 1
fi
"$PY" -c "import numpy" 2>/dev/null || "$PY" -m pip install -q numpy
if [ ! -f "$MODEL" ]; then
  echo "Model missing: $MODEL  (run demo/laptop_demo.sh once to download it)" >&2
  exit 1
fi

# Hide LiteRT's native INFO/WARNING chatter (errors still show).
quiet() { grep --line-buffered -vE '^(INFO: |WARNING: |[IW][0-9]{4} )' >&2; }

# One-time: embed the tool examples + enum descriptions (~95 texts, ~9 s on CPU). Edits to the
# tool table only re-embed the texts that changed.
[ -f data/vectors.npz ] || "$PY" app.py --build 2> >(quiet)

if [ "${1:-}" = "--eval" ]; then
  exec "$PY" app.py "$@" 2> >(quiet)
fi
exec "$PY" app.py --port "$PORT" "$@" 2> >(quiet)
