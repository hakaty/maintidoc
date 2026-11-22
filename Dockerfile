# M7.5: containerised Streamlit UI + RAG/agent code. PostgreSQL stays its
# own service (docker-compose.yml); Ollama runs NATIVELY on the host (never
# in this image) -- OLLAMA_BASE_URL points at host.docker.internal instead.
#
# CPU-only by default: torch is installed from the CPU wheel index, so the
# colleague without a GPU can run this image as-is. See docker-compose.yml's
# "gpu" profile for the CUDA variant.

FROM python:3.11-slim AS base

# System deps: PyMuPDF (fonts/rendering), Tesseract (reserved: no OCR
# fallback is used yet, but the spec asks for it to be ready if one is
# added), curl (HEALTHCHECK below), postgresql-client (`psql`/`pg_dump` --
# scripts/import_index.py, run BY the no-GPU colleague INSIDE this
# container, needs `psql` on a direct TCP connection to the `db` service;
# there is no `docker` CLI in here to shell out to).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        tesseract-ocr \
        tesseract-ocr-fra \
        curl \
        postgresql-client \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Python deps (cached as its own layer, rebuilt only when deps change) --
COPY requirements-docker.txt pyproject.toml ./

# CPU-only torch first (large download, pinned to the version the rest of
# the stack (open-clip-torch, sentence-transformers, timm) was tested
# against) -- never the project's venv version (torch==2.14.1+cu126, a CUDA
# build that does not apply here).
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch==2.14.1 torchvision==0.29.1 \
    && pip install --no-cache-dir -r requirements-docker.txt

# --- App code ---------------------------------------------------------------
COPY app ./app
COPY ui ./ui
COPY scripts ./scripts
COPY prompts ./prompts
COPY eval ./eval
COPY .streamlit ./.streamlit
COPY README.md ./
# data/ itself is NOT copied in: it's a bind mount at runtime (see
# docker-compose.yml) so an uploaded PDF, extracted images and the index
# survive a rebuild/restart -- only an empty directory skeleton is needed
# here, created below for the case the mount is a fresh empty volume.

# Installs the project itself (app/ as an importable package) without
# re-resolving/reinstalling any of the pinned dependencies above.
RUN pip install --no-cache-dir --no-deps -e .

# --- Non-root user -----------------------------------------------------------
# Owns /app (code + the data/models dirs it writes to at runtime: uploaded
# PDFs, extracted images, the HF model cache) so it can read/write without
# needing root.
RUN useradd --create-home --uid 1000 maintidoc \
    && mkdir -p data/pdf data/processed data/media models \
    && chown -R maintidoc:maintidoc /app
USER maintidoc

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/models \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
    CMD curl -f http://127.0.0.1:8501/_stcore/health || exit 1

CMD ["python", "-m", "streamlit", "run", "ui/streamlit_app.py", \
     "--server.address", "0.0.0.0", "--server.port", "8501", \
     "--server.headless", "true"]
