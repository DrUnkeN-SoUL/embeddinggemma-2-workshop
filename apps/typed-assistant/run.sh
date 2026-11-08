#!/usr/bin/env bash
# Type-safe on-device assistant: EmbeddingGemma 2 on the laptop CPU via LiteRT-LM.
# Usage: ./run.sh            # serve http://127.0.0.1:8091
#        ./run.sh --eval     # tool selection / exact args / clarifying questions on tests.json
#        NO_VOICE=1 ./run.sh # skip the voice server (the app works the same, the voice toggle says "off")
# Voice (optional, see README): demo/.venv-audio with torch + sentence-transformers + faster-whisper.
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
# Voice sidecar on :8092 if its venv exists. It loads in ~15 s in the background; the toggle in the
# app turns on once it answers.
VPY=../../demo/.venv-audio/bin/python
if [ -z "${NO_VOICE:-}" ] && [ -x "$VPY" ]; then
  HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_VERBOSITY=error "$VPY" voice.py \
    2> >(grep --line-buffered -vE 'Warning|warn|Loading weights' >&2) &
  VOICE_PID=$!
  trap 'kill $VOICE_PID 2>/dev/null' EXIT
else
  echo "(voice server not started: no demo/.venv-audio; typing still works)" >&2
fi
"$PY" app.py --port "$PORT" "$@" 2> >(quiet)
