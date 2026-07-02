PYTHON ?= python3
VENV ?= .venv
HOST ?= 127.0.0.1
PORT ?= 8000
VENV_PYTHON := $(VENV)/bin/python
VENV_PYTHON_ABS := $(abspath $(VENV_PYTHON))
VENV_RUFF := $(VENV)/bin/ruff
VENV_MYPY := $(VENV)/bin/mypy

.PHONY: install dev test lint format typecheck demo-personal sync-config check

install:
	$(PYTHON) -m venv --clear $(VENV)
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -e ".[dev]"
	printf '%s\n' '#!/usr/bin/env bash' 'set -euo pipefail' 'ROOT="$(CURDIR)"' 'cd "$$ROOT"' 'exec "$(VENV_PYTHON_ABS)" -m switchboard.cli "$$@"' > $(VENV)/bin/switchboard
	chmod +x $(VENV)/bin/switchboard
	cp $(VENV)/bin/switchboard $(VENV)/bin/ai-switchboard

dev:
	$(VENV_PYTHON) -m uvicorn switchboard.app.main:app --reload --reload-dir switchboard --reload-dir config --host $(HOST) --port $(PORT)

test:
	$(VENV_PYTHON) -m pytest

lint:
	$(VENV_RUFF) check .

format:
	$(VENV_RUFF) format .
	$(VENV_RUFF) check --fix .

typecheck:
	$(VENV_MYPY) switchboard

demo-personal:
	bash scripts/demo_personal.sh

sync-config:
	rm -rf config
	mkdir -p config
	cp switchboard/config/*.yaml switchboard/config/*.json config/

check: lint typecheck test
