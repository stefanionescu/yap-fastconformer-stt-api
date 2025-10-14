# Moonshine Streaming ASR Server

Moonshine-based real-time speech-to-text service with GPU batching, WebRTC data-channel streaming, and Docker deployment assets. The server targets the English Moonshine Base model by default and is designed for low-latency transcription without baked-in VAD.

## Highlights
- Real-time streaming transcription over WebRTC data channels (binary PCM16 @ 16 kHz)
- Global batching with configurable batch size and wait time (default max batch 32)
- Moonshine Transformer backend with fp16, bf16, fp32, or int8 execution modes
- Clean scripts for local setup, warmup, benchmarking, and Docker builds
- Tests exercising streaming, warmup, and benchmarking flows

## Prerequisites
- `pkg-config` (required to build PyAV for WebRTC transport)
- `ffmpeg` (used by the sample tooling for audio resampling)

Install them via your package manager, e.g. `brew install pkg-config ffmpeg` on macOS or `apt install pkg-config ffmpeg` on Debian/Ubuntu.

## Local Quickstart
```bash
# 1) Create a virtual environment + install deps (CPU torch by default)
bash scripts/install.sh

# 2) Launch the server (defaults to 0.0.0.0:8000)
bash scripts/start.sh

# 3) Run a warmup / health check
bash scripts/warmup.sh --file mid.wav --rtf 10 --full-text

# 4) Try streaming from the sample client
bash scripts/client.sh --file mid.wav --rtf 1 --full-text
```

### GPU wheel override
Set `TORCH_CUDA` before running `scripts/install.sh` to install a specific CUDA wheel, e.g.:
```bash
TORCH_CUDA=cu121 TORCH_VERSION=2.4.0 bash scripts/install.sh
```

## Docker Quickstart
```bash
# Build image (moonshine-asr:latest)
bash docker/build.sh

# Run with GPU passthrough on :8000
docker run --rm -it --gpus all -p 8000:8000 moonshine-asr:latest
```

The container exposes the `/webrtc` HTTP endpoint for SDP exchange. Use the tests from the host to validate:
```bash
python tests/warmup.py --server 127.0.0.1:8000 --file mid.wav --rtf 10 --full-text
python tests/bench.py   --server 127.0.0.1:8000 --file mid.wav --rtf 1.0 --n 20 --concurrency 5
```

## WebRTC Protocol
1. **Offer/Answer** — POST SDP offer to `http://<host>:<port>/webrtc` (JSON body: `{ "sdp": ..., "type": ... }`). Server replies with an SDP answer.
2. **Data channel** — Client creates a data channel (label is ignored). After channel opens:
   - Client sends `{"op":"init","sid":"<uuid>","sr":16000}`.
   - Server replies `{"op":"ready","sid":...,"max_batch":...}`.
   - Client streams binary PCM16 mono audio frames (20 ms chunks recommended).
   - Client signals end with `{"op":"close"}`.
   - Server emits `{"op":"interim",...}` updates and a final `{"op":"final","final":true,...}` payload.

If an error occurs the server sends `{"op":"error","reason":"..."}` and closes the data channel.

## Configuration (environment variables)
- `ASR_HOST` (default `0.0.0.0`)
- `ASR_PORT` (default `8000`)
- `ASR_WEBRTC_PATH` (default `/webrtc`)
- `MOONSHINE_MODEL_ID` (default `UsefulSensors/moonshine-base`)
- `MOONSHINE_PRECISION` (`fp16`, `bf16`, `fp32`, `int8`; default `fp16`)
- `MAX_BATCH_SIZE` (default `32`)
- `MAX_BATCH_WAIT_MS` (default `10`)
- `MAX_BUFFER_SECONDS` (default `120`)
- `MODEL_WARMUP_SECONDS` (default `1.5`)

## Scripts
- `scripts/install.sh` — bootstrap virtualenv + install dependencies
- `scripts/start.sh` — start the ASR server using the active venv
- `scripts/warmup.sh` — run warmup/latency probe against a server
- `scripts/client.sh` — simple CLI client for manual testing
- `scripts/bench.sh` — concurrency benchmark harness
- `scripts/stop.sh` — stop the server, drop caches, and remove the local venv

Each script honours `VENV_PATH` if you want to reuse a custom environment.

## Tests & Benchmarks
- `tests/warmup.py` — single streaming session health check
- `tests/client.py` — manual client for experimentation
- `tests/bench.py` — runs multiple concurrent sessions and summarizes latency stats

All scripts/tests stream 16 kHz PCM16 mono audio from files under `samples/` by default.

## Docker Assets
- `docker/Dockerfile` — CUDA 12.1 runtime base with torch 2.4.0+cu121
- `docker/build.sh` — helper to build `moonshine-asr` image
- `docker/run.sh` — helper to run the container with GPU access

Override `MOONSHINE_MODEL_ID`, `MOONSHINE_PRECISION`, or `MAX_BATCH_SIZE` via `docker run -e ...`.
