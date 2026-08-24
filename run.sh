#!/usr/bin/env bash
cd "$(dirname "$0")" || exit 1
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi
API_URL="${API_URL:-http://${API_HOST:-localhost}:${API_PORT:-9090}}"
if [[ -f static/script.js ]]; then
  sed -i.bak "s|const baseUrl = \"http://localhost:9090\"|const baseUrl = \"$API_URL\"|g" static/script.js
fi
uvicorn main:app --reload --host "${API_HOST:-localhost}" --port "${API_PORT:-9090}"
