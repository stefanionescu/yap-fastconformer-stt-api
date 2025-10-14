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

Install them via your package manager, e.g. `brew install pkg-config ffmpeg` on macOS or `apt install pkg-config ffmpeg libavformat-dev libavdevice-dev libavcodec-dev libavutil-dev libswscale-dev libopus-dev libvpx-dev` on Debian/Ubuntu. The helper `scripts/core/install.sh` will attempt to install these automatically when run with sufficient privileges.

## One-Step Quickstart
```bash
bash scripts/run_all.sh
```

This single command installs dependencies (or reuses the cached venv), launches the server, and tails the unified log at `logs/moonshine.log`. You can `Ctrl+C` the tail and the background service keeps running. To resume log streaming later:
```bash
tail -f logs/moonshine.log
```
Check the last 200 log lines without following:
```bash
tail -n 200 logs/moonshine.log
```

### Manual control / GPU overrides
If you need to customise the install step (e.g., selecting a different CUDA wheel), run:
```bash
TORCH_CUDA=cu121 TORCH_VERSION=2.4.0 bash scripts/core/install.sh
bash scripts/core/start.sh
```

Utility runners for tests live under `scripts/test/` (see below), or you can use `scripts/stop.sh` to tear everything down.

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
- `scripts/run_all.sh` — fire-and-forget install→launch pipeline with live log tail
- `scripts/install_and_start.sh` — helper invoked by `run_all.sh` (install + launch) — typically not called directly
- `scripts/core/install.sh` — bootstrap virtualenv + install dependencies
- `scripts/core/start.sh` — start the ASR server using the active venv
- `scripts/stop.sh` — stop the server, drop caches, remove the local venv, and uninstall any system packages `scripts/core/install.sh` added
- `scripts/test/warmup.sh` — run warmup/latency probe against a server
- `scripts/test/client.sh` — simple CLI client for manual testing
- `scripts/test/bench.sh` — concurrency benchmark harness

Each script honours `VENV_PATH` if you want to reuse a custom environment.

## Tests & Benchmarks
- `scripts/test/warmup.sh` — single streaming session health check (local GPU)
- `scripts/test/client.sh` — simple CLI client for quick manual trials (local GPU)
- `scripts/test/bench.sh` — concurrency benchmark harness (local GPU)
- `tests/client.py` — advanced client for local or RunPod endpoints

Example usage (assumes the server is running):
```bash
# Local GPU smoke test
bash scripts/test/warmup.sh --file mid.wav --rtf 10 --full-text

# Local GPU benchmark
bash scripts/test/bench.sh --file mid.wav --rtf 1.0 --n 20 --concurrency 4

# Remote RunPod call using profile defaults
python tests/client.py --print-partials

# Override with explicit HTTPS endpoint
python tests/client.py --https-url https://your-runpod-endpoint/webrtc --api-key "$RUNPOD_API_KEY" --file mid.wav
```
> **Note:** The helper scripts under `scripts/test/` automatically activate `.venv`. If you invoke the Python modules directly (e.g. `python tests/client.py`) make sure the virtual environment created by `scripts/core/install.sh` is activated first (`source .venv/bin/activate`).

### RunPod environment file
Create a simple `.env` file alongside the repo (or point `--env-file` elsewhere):
```
# .env (default path: repo_root/.env)
RUNPOD_API_KEY=your_runpod_api_key
RUNPOD_HTTPS_URL=https://your-endpoint.example.com/webrtc
# Or specify raw TCP instead:
# RUNPOD_TCP_HOST=1.2.3.4
# RUNPOD_TCP_PORT=8000
# Optional custom path
# RUNPOD_WEBRTC_PATH=/custom/path
```
Values in `.env` are automatically loaded but can be overridden via CLI flags or environment variables (`RUNPOD_API_KEY`, `RUNPOD_HTTPS_URL`, etc.).

All scripts/tests stream 16 kHz PCM16 mono audio from files under `` by default.

## Docker Assets
- `docker/Dockerfile` — CUDA 12.1 runtime base with torch 2.4.0+cu121
- `docker/build.sh` — helper to build `moonshine-asr` image
- `docker/run.sh` — helper to run the container with GPU access

Override `MOONSHINE_MODEL_ID`, `MOONSHINE_PRECISION`, or `MAX_BATCH_SIZE` via `docker run -e ...`.
