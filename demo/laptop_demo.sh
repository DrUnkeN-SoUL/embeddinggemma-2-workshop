#!/usr/bin/env bash
# Laptop CPU demo: start `litert-lm serve` in the background, wait until it answers, run laptop_demo.py.
# Usage: ./laptop_demo.sh                     # interactive
#        ./laptop_demo.sh --query "my container keeps crashing"
set -euo pipefail
cd "$(dirname "$0")"

HOST=127.0.0.1
PORT=${PORT:-9379}
MODEL_FILE=models/embeddinggemma-2-text-270m.litertlm
MODEL_ID=embeddinggemma-2-text-270m.litertlm
MODEL_URL=https://huggingface.co/litert-community/embeddinggemma-2-text-270m-litert-lm/resolve/main/embeddinggemma-2-text-270m.litertlm
BIN=.venv/bin

if [ ! -x "$BIN/litert-lm" ]; then
  echo "No venv. Run:  python3 -m venv demo/.venv && demo/.venv/bin/pip install litert-lm==0.18.0" >&2
  exit 1
fi

# One-time setup: download the 165 MB text-only model and register it with the CLI (hard link, no copy).
if [ ! -f "$MODEL_FILE" ]; then
  mkdir -p models
  echo "Downloading $MODEL_FILE ..."
  curl -fL --progress-bar -o "$MODEL_FILE.part" "$MODEL_URL" && mv "$MODEL_FILE.part" "$MODEL_FILE"
fi
if ! "$BIN/litert-lm" list 2>/dev/null | grep -q "^$MODEL_ID "; then
  "$BIN/litert-lm" import "$MODEL_FILE"
fi

# Start the server unless one is already listening.
if curl -sf "http://$HOST:$PORT/v1/models" >/dev/null 2>&1; then
  echo "Using the server already running on $HOST:$PORT"
else
  "$BIN/litert-lm" serve --host "$HOST" --port "$PORT" >serve.log 2>&1 &
  SERVER_PID=$!
  trap 'kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null || true' EXIT
  echo -n "Starting litert-lm serve on $HOST:$PORT (log: demo/serve.log) "
  for _ in $(seq 1 60); do
    if curl -sf "http://$HOST:$PORT/v1/models" >/dev/null 2>&1; then break; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo; echo "Server exited:"; tail -20 serve.log; exit 1; fi
    echo -n "."; sleep 0.5
  done
  echo " ready"
fi

LITERT_URL="http://$HOST:$PORT/v1/embeddings" "$BIN/python" laptop_demo.py "$@"
