# Yap FastConformer Streaming ASR Server

Production-ready streaming ASR server using NVIDIA NeMo FastConformer Hybrid Large with cache-aware streaming.

**Key Features:**
- Real-time streaming transcription with interim results
- Global batching for high concurrency (64+ streams)
- WebSocket protocol, VAD-agnostic
- Docker + script deployment options

### Quickstart (Docker)

1) Build the image

```bash
bash docker/build.sh
```

2) Run with GPUs

```bash
bash docker/run.sh
```

The server listens on `ws://localhost:8000`.

3) Run Tests

```bash
python tests/warmup.py --server 127.0.0.1:8000 --file mid.wav --rtf 10 --full-text
python tests/bench.py   --server 127.0.0.1:8000 --file mid.wav --rtf 1.0 --n 20 --concurrency 5
```

Note: When running these inside the container, first activate the virtualenv or use the wrapper:

```bash
source .venv/bin/activate
# or
bash scripts/run/warmup.sh mid.wav 10
```

### WebSocket Protocol

- Client → Server
  - Init: `{"op":"init","sid":"<session_id>","sr":16000}`
  - Audio frames: binary PCM16LE mono at 16 kHz in 20 ms chunks (640 samples → 1280 bytes)
  - Close: `{"op":"close","sid":"<session_id>"}`

- Server → Client
  - Interim hyp each tick: `{"op":"interim","sid":"<session_id>","text":"...","final":false,"ts":<unix_ms>}`

Notes:
- No built-in VAD; external EoS/turn-taking is recommended (e.g., Pipecat Smart-Turn). Stop sending audio to end a turn.

### Configuration (env vars)

- `ASR_HOST` (default `0.0.0.0`)
- `ASR_PORT` (default `8080`)
- `ASR_MODEL` (default `nvidia/stt_en_fastconformer_hybrid_large_streaming_multi`)
- `ASR_ATT_CTX` (default `70,1`) — look-ahead; supported: `70,0` (0 ms), `70,1` (80 ms), `70,16` (480 ms), `70,33` (1040 ms)
- `ASR_DECODER` (default `rnnt`) — `rnnt` or `ctc`
- `ASR_DEVICE` (default `cuda:0`)
- `ASR_STEP_MS` (default `20`) — batcher tick period (ms)
- `ASR_MAX_BATCH` (default `64`) — max concurrent streams per tick

Example override:

```bash
docker run --rm -it --gpus all -p 8080:8080 \
  -e ASR_ATT_CTX=70,0 -e ASR_MAX_BATCH=96 \
  fastconf-streaming:latest
```

## Script Deployment

1) Set up Hugging Face token:
```bash
export HF_TOKEN=your_token_here
# Get token from: https://huggingface.co/settings/tokens
```

2) Deploy server:
```bash
# Deploy server (creates venv, installs deps, starts in background):
bash scripts/deploy.sh

# Deploy + auto warmup test:
bash scripts/deploy.sh --warmup

# Follow logs (if you exited deploy):
tail -f logs/asr_server.log

# Manual warmup/test:
bash scripts/run/warmup.sh mid.wav 10
python tests/bench.py --server 127.0.0.1:8000 --file samples/mid.wav --rtf 1.0 --n 10 --concurrency 3

# Stop server + cleanup:
bash scripts/stop.sh
```

If you run Python directly inside the repo, activate the venv first:

```bash
source .venv/bin/activate
```

## Configuration

Edit `scripts/deps/env.sh` or set environment variables:
- `ASR_HOST` (default: 0.0.0.0)
- `ASR_PORT` (default: 8000)  
- `ASR_MAX_BATCH` (default: 64)
- `ASR_ATT_CTX` (default: 70,1)
- `ASR_DEVICE` (default: cuda:0)

### Performance tuning

- Batching: Increase `ASR_MAX_BATCH` until GPU saturates (monitor step latency and queueing).
- Look-ahead: `ASR_ATT_CTX=70,1` is a good default (80 ms worst, ~40 ms avg). `70,0` can reduce latency further with some WER cost.
- Tick period: `ASR_STEP_MS=20` (50 Hz) balances server overhead and responsiveness. 10 ms is possible at higher CPU/GPU overhead.
- Scale out: Run multiple containers and sticky-route sessions; one GPU per process is simplest.

### Build details

- Base: `nvidia/cuda:12.1.1-cudnn-runtime-ubuntu22.04`
- PyTorch: `2.3.1+cu121`, Torchaudio matched
- NeMo: `nemo_toolkit[asr]==1.23.0`
- Optional model prefetch during build to avoid cold start

### Known limitations

- Cache-aware streaming path uses fp32 internally and forces fp32 in several layers, so fp16/bf16/INT8 are not supported here.
- If you need INT8/FP8 with TensorRT, you must export and implement a custom stateful streaming loop; parity with NeMo’s cache-aware step is not guaranteed.