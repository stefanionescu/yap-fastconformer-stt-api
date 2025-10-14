FROM nvidia/cuda:12.1.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip python3-setuptools python3-wheel \
    ffmpeg git curl ca-certificates build-essential pkg-config \
    libopus0 libvpx7 libsrtp2-1 libavformat58 libavcodec58 libavdevice58 libffi-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN python -m pip install --upgrade pip wheel

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.txt

COPY server /app/server
COPY tests /app/tests
COPY samples /app/samples
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV ASR_HOST=0.0.0.0 \
    ASR_PORT=8000 \
    ASR_WEBRTC_PATH=/webrtc \
    MOONSHINE_MODEL=moonshine/base \
    MOONSHINE_ONNX_PROVIDERS=TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider

EXPOSE 8000
ENTRYPOINT ["/entrypoint.sh"]
