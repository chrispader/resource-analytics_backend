IMAGE ?= resource-analytics_backend

.PHONY: run
run:
	docker build -t $(IMAGE) .
	docker run -it --rm -p 9090:9090 $(IMAGE)
