# Replication package targets. Run `make help` for the list.

.PHONY: setup verify ingest gate test lint help
.DEFAULT_GOAL := help

setup:   ## install dependencies into .venv (uv sync)
	uv sync --extra dev

verify:  ## reproduction path A: re-derive headline numbers from the digests
	uv run pytest

ingest:  ## reproduction path B step 1: download AIDev into the DuckDB lake
	uv run python -m prfix.common.loaders.aidev

gate:    ## reproduction path B: assert every headline number (needs the lake)
	uv run python human-annotation/consolidation_gate.py

test:    ## alias for verify
	uv run pytest

lint:    ## ruff check
	uv run ruff check src tests human-annotation

help:    ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | sort \
	  | awk 'BEGIN{FS=":.*## "}; {printf "  %-8s %s\n", $$1, $$2}'
