## FastConformer Streaming ASR (NeMo)

This is a production-ready, self-hosted streaming ASR server for the NVIDIA NeMo FastConformer Hybrid Large (cache-aware streaming) model:

- Model: `nvidia/stt_en_fastconformer_hybrid_large_streaming_multi` ([Hugging Face](https://huggingface.co/nvidia/stt_en_fastconformer_hybrid_large_streaming_multi))
- True streaming with interim hypotheses (RNNT decoder)
- Global batching across many concurrent WS clients (20 ms ticks)
- Tuned for L40S / CUDA 12.x
- Simple WebSocket protocol; VAD-agnostic (external turn-taking recommended)

Important: NeMo cache-aware streaming currently runs the model in fp32. Mixed-precision/INT8 are not supported on this code path at the moment (see NeMo script notes). Scale via batching and horizontal replication.

### Contents

```
./docker/      # Dockerfile + scripts
./server/      # NeMo loader, batcher, WebSocket server
./client/      # Example WAV client
```

### Quickstart

1) Build the image

```bash
bash docker/build.sh
```

2) Run with GPUs

```bash
bash docker/run.sh
```

The server listens on `ws://localhost:8080`.

3) Try the example client

```bash
python client/ws_client_wav.py /path/to/16khz_mono.wav ws://localhost:8080 demo1
```

You will see JSON `interim` messages printed as the audio streams.

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
- `ASR_MAX_BATCH` (default `128`) — max concurrent streams per tick

Example override:

```bash
docker run --rm -it --gpus all -p 8080:8080 \
  -e ASR_ATT_CTX=70,0 -e ASR_MAX_BATCH=96 \
  fastconf-streaming:latest
```

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

### References

- NVIDIA model card: [Hugging Face](https://huggingface.co/nvidia/stt_en_fastconformer_hybrid_large_streaming_multi)
- NeMo toolkit: `https://github.com/NVIDIA/NeMo`


