IMAGE ?= resource-analytics_backend

.PHONY: run run-docker
run:
	set -a && [ -f .env ] && . ./.env; set +a && .venv/bin/uvicorn main:app --host "$${API_HOST:-localhost}" --port "$${API_PORT:-9090}"

run-docker:
	docker build -t $(IMAGE) .
	docker run -it --rm -p 9090:9090 $(IMAGE)
