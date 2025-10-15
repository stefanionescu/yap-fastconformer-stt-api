# Parakeet-TDT Streaming ASR Server

Self-hosted Parakeet-TDT-0.6B-V3 streaming speech-to-text server built on the NeMo 2.4 toolchain. Audio arrives over a FastAPI WebSocket endpoint, partial hypotheses stream out in real time, and the final transcript flushes as soon as you emit an EOS control frame.

## Highlights
- True RNNT streaming via NeMo’s `partial_hypothesis` interface — no fake chunking
- Single-command installer (`./scripts/main.sh`) that runs fully detached, with all logs coalesced into `logs/latest.log`
- Scripted orchestration for environment checks, apt packages, Python 3.12 virtualenv, CUDA 12.1 PyTorch wheels, requirements, model prefetch, and server launch
- FastAPI + uvicorn + uvloop WebSocket server, tuned for low-latency multilingual transcription
- CLI clients and benchmarks for smoke tests, warmup, and concurrency sizing
- Docker build that mirrors the script flow and launches the exact same server inside a GPU-enabled container

## Requirements
- Ubuntu 22.04/24.04 GPU host with NVIDIA driver + CUDA runtime already installed
- `nvidia-smi` must work before running the setup
- Ability to install apt packages and Python 3.12 (use `sudo` or run as root)

## One-Command Quickstart (non-Docker)
```bash
bash scripts/main.sh
```

What happens:
1. `scripts/00_env_check.sh` verifies `nvidia-smi` and notes the Python 3.12 status.
2. `scripts/01_system_deps.sh` installs Python 3.12, `ffmpeg`, `libsndfile1`, `build-essential`, `git`, and base certificates.
3. `scripts/02_python_env.sh` creates `.venv` and upgrades `pip`, `wheel`, `setuptools`.
4. `scripts/03_python_deps.sh` installs CUDA 12.1 PyTorch 2.4.0 wheels and Python dependencies from `requirements.txt`.
5. `scripts/04_prefetch_model.sh` caches `nvidia/parakeet-tdt-0.6b-v3` via `huggingface_hub`.
6. `scripts/05_run_server.sh` activates `.venv` and execs `uvicorn server:app` on `0.0.0.0:8000`.

All stdout/stderr is appended to `logs/run_<timestamp>.log` with a symlink at `logs/latest.log`. `scripts/main.sh` automatically tails the log after launching; hit `Ctrl+C` to drop the tail — the background process keeps running. Logs are still available via:
```bash
tail -f logs/latest.log
```

Stop the detached pipeline (and uvicorn) with:
```bash
bash scripts/stop.sh
```

`scripts/stop.sh` kills the PID stored in `.run.pid` (or falls back to `pkill -f "uvicorn server:app"`), then removes the `.venv` virtualenv, local Hugging Face caches, build artifacts, and all logs.

### Tuning knobs
Override before invoking `./scripts/main.sh` (or export globally):
- `STEP_MS` (default `240`)
- `RIGHT_CONTEXT_MS` (default `160`)
- `MAX_INFLIGHT_STEPS` (default `1`)
- `HF_HOME` (default `$(pwd)/.cache/hf`)

### Scripts overview
- `scripts/main.sh` — orchestrates the full chain in the background, tails logs for you
- `scripts/stop.sh` — stops the detached pipeline and purges the virtualenv, caches, and logs
- `scripts/steps/00_env_check.sh` — sanity checks for GPU + Python 3.12
- `scripts/steps/01_system_deps.sh` — apt dependencies (uses `sudo` if necessary)
- `scripts/steps/02_python_env.sh` — virtualenv bootstrap and base tooling upgrade
- `scripts/steps/03_python_deps.sh` — installs CUDA 12.1 PyTorch 2.4.0 stack + project requirements
- `scripts/steps/04_prefetch_model.py|.sh` — caches Parakeet-TDT-0.6B-V3 into `HF_HOME`
- `scripts/steps/05_run_server.sh` — activates `.venv` and launches `uvicorn`

## WebSocket Protocol
Endpoint: `ws://<host>:8000/ws`

- Stream raw PCM16 mono audio at 16 kHz in any chunk size (20 ms suggested).
- After the final audio frame, send the control frame `b"__CTRL__:EOS"` to flush the final transcript immediately.
- Optional reset during a session with `b"__CTRL__:RESET"`.
- Partial responses arrive as JSON text frames: `{ "type": "partial", "text": "…" }`.
- Final response after EOS: `{ "type": "final", "text": "…" }`.
- Health check endpoint: `GET /status` → `200 ok`.

## Clients, Warmup, and Benchmarks
Activate the virtualenv first:
```bash
source .venv/bin/activate
```

### Smoke test
```bash
python tests/client.py --file mid.wav --full-text
```

### Warmup / latency probe
```bash
python tests/warmup.py --file mid.wav --rtf 10 --print-partials
```

### Concurrency benchmark
```bash
python tests/bench.py --url ws://127.0.0.1:8000/ws --streams 64 --duration 30
```

### Standalone utilities
- `tests/client.py` — minimal CLI client (`python tests/client.py --file mid.wav --url ws://…`)
- `tests/bench.py` — synthetic tone generator that hammers the server with many simultaneous streams

## Docker Workflow
```bash
# Build the image (parakeet-tdt-streaming:latest)
bash docker/build.sh

# Run with GPU access, publishing port 8000
PORT=8000 bash docker/run.sh
```

The Dockerfile mirrors the host workflow: it installs Python 3.12, creates `/app/.venv`, installs CUDA 12.1 PyTorch wheels plus requirements, and launches `scripts/05_run_server.sh` as the container entrypoint. Default env vars inside the container:
- `STEP_MS=240`
- `RIGHT_CONTEXT_MS=160`
- `MAX_INFLIGHT_STEPS=1`
- `HF_HOME=/cache/hf` (persistent model cache; mount a volume if desired)

Need to preseed the Hugging Face cache or reuse GPUs across runs? Mount a host directory:
```bash
docker run --rm -it --gpus all \
  -p 8000:8000 \
  -v /path/to/hf-cache:/cache/hf \
  parakeet-tdt-streaming:latest
```

## Requirements File (Python)
```
nemo_toolkit[asr]==2.4.0
fastapi, uvicorn[standard], uvloop, websockets, soundfile, pydub
huggingface_hub[hf-xet], hf_transfer, numpy<2
```
PyTorch (torch/torchvision/torchaudio 2.4.0) is installed separately via `scripts/03_python_deps.sh` to ensure the CUDA 12.1 wheels are pulled from the official index.

## Notes
- The first `./scripts/main.sh` run downloads ~2.5 GB of model weights. Subsequent runs reuse `HF_HOME`.
- The server loads the NeMo checkpoint on import, so expect ~20 s of initialisation on cold GPU nodes.
- Set `HF_HUB_ENABLE_HF_TRANSFER=1` to accelerate model downloads via the Rust binary.
- Use `MAX_INFLIGHT_STEPS>1` when the GPU has headroom to parallelise RNNT calls.

## Troubleshooting
- `ERROR: nvidia-smi not found` → install the proprietary NVIDIA driver + CUDA toolkit runtime first.
- `python3.12: command not found` → rerun `./scripts/main.sh` as a user with sudo privileges so apt can install Python 3.12.
- Hugging Face download stalls → ensure outbound HTTPS is allowed or pre-populate the cache and export `HF_HOME`.
- To reset everything, delete `.venv`, `.cache/hf`, `logs/*`, and rerun `./scripts/main.sh`.
