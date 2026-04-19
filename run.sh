#!/usr/bin/env bash
cd "$(dirname "$0")" || exit 1
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi
if [ -n "$API_URL" ]; then
  sed -i "s|const baseUrl = \"http://localhost:9090\"|const baseUrl = \"$API_URL\"|g" static/script.js
fi
uvicorn main:app --reload --host 0.0.0.0 --port 9090
