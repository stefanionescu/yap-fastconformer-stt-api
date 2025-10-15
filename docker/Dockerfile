FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/cache/hf \
    TRANSFORMERS_OFFLINE=0 \
    HF_HUB_ENABLE_HF_TRANSFER=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3-pip \
    ffmpeg libsndfile1 git ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /cache/hf

WORKDIR /app

COPY requirements.txt /app/requirements.txt
COPY scripts /app/scripts
RUN chmod +x /app/scripts/*.sh /app/scripts/steps/*.sh

RUN python3.12 -m venv /app/.venv \
    && . /app/.venv/bin/activate \
    && python -m pip install -U pip wheel setuptools \
    && pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    && pip install -r /app/requirements.txt

COPY server /app/server
COPY tests /app/tests
COPY samples /app/samples
COPY docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENV STEP_MS=240 \
    RIGHT_CONTEXT_MS=160 \
    MAX_INFLIGHT_STEPS=1 \
    PATH="/app/.venv/bin:${PATH}"

EXPOSE 8000
ENTRYPOINT ["/app/entrypoint.sh"]
