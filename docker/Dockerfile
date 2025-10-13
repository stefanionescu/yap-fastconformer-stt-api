FROM nvidia/cuda:12.1.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST="8.9;8.0+PTX" \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip python3-setuptools python3-wheel \
    ffmpeg git curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN python -m pip install --upgrade pip

# PyTorch with CUDA 12.1
RUN pip install --no-cache-dir torch==2.3.1+cu121 torchaudio==2.3.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121

WORKDIR /app

# Install requirements from repo root (includes NeMo and runtime deps)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Optional: prefetch model to reduce cold start
RUN python - <<'PY'
from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
_ = EncDecHybridRNNTCTCBPEModel.from_pretrained(model_name="nvidia/stt_en_fastconformer_hybrid_large_streaming_multi")
print("Model prefetch done")
PY

# App code
COPY server /app/server
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Copy tests and sample audio for in-container warmup/bench
COPY tests /app/tests
COPY samples /app/samples

EXPOSE 8000
ENTRYPOINT ["/entrypoint.sh"]


