.PHONY: up up-ollama down test lint inspect check ui

PDF ?= data/pdf/document.pdf

up:
	docker compose up -d

# Opt-in: only needed if you don't want a native Ollama install (see docs/setup.md).
up-ollama:
	docker compose --profile ollama up -d

down:
	docker compose --profile ollama down

test:
	pytest

lint:
	ruff check .

inspect:
	python scripts/inspect_pdf.py $(PDF)

check:
	python scripts/check_env.py

# Launched with "python -m streamlit" (not the streamlit.exe entry point):
# some .exe launchers are blocked by this machine's Windows application
# control policy, see docs/setup.md.
ui:
	python -m streamlit run ui/streamlit_app.py
