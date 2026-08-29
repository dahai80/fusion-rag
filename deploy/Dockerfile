# Fusion-RAG container image.
#
# Deploy pattern: fusion-rag runs in this container; fusion-mlx (the MLX
# inference engine, Apple Silicon native) runs on the host or a separate metal
# node, reached via FUSION_MLX_URL. fusion-rag never imports MLX — all inference
# is HTTP — so a linux container is fine for the RAG/storage layer even though
# MLX itself cannot run in a standard container.
#
# fusion-core is the in-tree shared base (../fusion-core in the monorepo). It is
# NOT on PyPI; CI installs it from git, and so does this image.
#
# Build:  docker build -t fusion-rag:0.7.2 -f deploy/Dockerfile .
# Run:    docker run -p 11436:11436 -e FUSION_MLX_URL=http://host.docker.internal:11432/v1 \
#           -v fusion-rag-stores:/root/.fusion-rag/stores fusion-rag:0.7.2
#
# Single-process only (see CLAUDE.md H3): do NOT scale this with `--workers N`
# or run replicas against one shared stores volume. Scale behind a stateless
# load balancer only with one stores volume per replica.

FROM python:3.12-slim AS base

# PyMuPDF / lancedb need libglib + libxml2 runtime libs on slim.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libxml2 libgl1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install fusion-core from git (same source CI uses — it is in-tree, not PyPI),
# then fusion-rag with its runtime deps. Split COPY so dep install is cached
# across source changes.
RUN pip install --no-cache-dir "git+https://github.com/dahai80/fusion-core.git"
COPY pyproject.toml ./
COPY fusion_rag/ ./fusion_rag/
RUN pip install --no-cache-dir .

# Stores live on a volume so KB data survives container recreation.
ENV FUSION_RAG_HOST=0.0.0.0 \
    FUSION_RAG_PORT=11436 \
    FUSION_RAG_STORES_DIR=/root/.fusion-rag/stores \
    FUSION_RAG_LOG_DIR=/root/.fusion-rag/logs
VOLUME ["/root/.fusion-rag/stores"]

EXPOSE 11436

# /health = liveness (process up), /ready = readiness (deps reachable). A
# orchestrator should use /ready for routing and /health for restart.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fs http://127.0.0.1:11436/health || exit 1

CMD ["python", "-m", "fusion_rag.api.server"]
