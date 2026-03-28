# ── Build stage: compile native extensions ───────────────────────
FROM nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04 AS build

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    g++ \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# ── Runtime stage: lean production image ─────────────────────────
FROM nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04

LABEL org.opencontainers.image.title="person-identification-service" \
      org.opencontainers.image.description="Face recognition and motion direction detection" \
      org.opencontainers.image.licenses="AGPL-3.0"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    libgl1 \
    libglib2.0-0 \
    && apt-get purge -y software-properties-common \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire virtualenv from the build stage
COPY --from=build /opt/venv /opt/venv

WORKDIR /app

COPY app/ app/
COPY config/ config/

RUN mkdir -p data/embeddings data/models

VOLUME ["/app/data"]

EXPOSE 8100

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8100/health')" || exit 1

ENTRYPOINT ["uvicorn"]
CMD ["app.main:app", "--host", "0.0.0.0", "--port", "8100"]
