from __future__ import annotations
import asyncio, json, os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
from omegaconf import OmegaConf

import nemo, nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.submodules.transducer_decoding.label_looping_base import (
    GreedyBatchedLabelLoopingComputerBase,
)
from nemo.collections.asr.parts.utils.streaming_utils import ContextSize, StreamingBatchedAudioBuffer
from nemo.collections.asr.parts.utils.rnnt_utils import batched_hyps_to_hypotheses

# ---------- Tuning ----------
SAMPLE_RATE = 16_000
BYTES_PER_SAMPLE = 2
# keep <300 ms commit: ~80 ms chunk + ~120 ms RC (tweak if needed)
CHUNK_SECS = float(os.getenv("CHUNK_SECS", "0.08"))
RIGHT_CONTEXT_SECS = float(os.getenv("RIGHT_CONTEXT_SECS", "0.12"))
LEFT_CONTEXT_SECS = float(os.getenv("LEFT_CONTEXT_SECS", "5.0"))

MIN_EMIT_CHARS = int(os.getenv("MIN_EMIT_CHARS", "1"))
MAX_INFLIGHT_STEPS = int(os.getenv("MAX_INFLIGHT_STEPS", "4"))
MAX_DEBUG_STEPS = int(os.getenv("DEBUG_STEPS", "4"))

CONTROL_PREFIX = b"__CTRL__:"
CTRL_EOS = b"EOS"
CTRL_RESET = b"RESET"

print("Loading NeMo Parakeet-TDT-0.6b-v3 …")
model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v3")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
print(f"[device] {device}")
model.eval(); model.freeze()
print(f"[nemo] {nemo.__version__}")

# Label-looping greedy
decoding_cfg = OmegaConf.create({
    "strategy": "greedy_batch",
    "preserve_alignments": False,
    "fused_batch_size": -1,
    "compute_timestamps": False,
    "greedy": {
        "loop_labels": True,
        "max_symbols_per_step": 10,
        "use_cuda_graph_decoder": True,  # install cuda-python>=12.3 to enable CG decoding
    },
})
model.change_decoding_strategy(decoding_cfg)
decoding_computer: GreedyBatchedLabelLoopingComputerBase = model.decoding.decoding.decoding_computer  # type: ignore

# Preproc quirks off
try:
    model.preprocessor.featurizer.dither = 0.0
    model.preprocessor.featurizer.pad_to = 0
except Exception:
    pass

gpu_semaphore = asyncio.Semaphore(MAX_INFLIGHT_STEPS)

# ----- derive encoder frame mapping -----
cfg = model.cfg
assert int(cfg.preprocessor.sample_rate) == SAMPLE_RATE, "Sample rate mismatch"
feature_stride = float(cfg.preprocessor.window_stride)  # ~0.01
enc_sub = int(getattr(model.encoder, "subsampling_factor", 4))
features_frame2samples = (int(SAMPLE_RATE * feature_stride) // enc_sub) * enc_sub
encoder_frame2audio_samples = features_frame2samples * enc_sub  # one encoder frame == ~40ms*16000=640 samples
enc_hop_ms = 1000.0 * encoder_frame2audio_samples / SAMPLE_RATE

def _secs_to_samples(secs: float) -> int:
    return int(round(secs * SAMPLE_RATE))

# Build context in *encoder frames* and *samples*
context_enc = ContextSize(
    left=int(LEFT_CONTEXT_SECS / (feature_stride * enc_sub)),
    chunk=max(1, int(CHUNK_SECS / (feature_stride * enc_sub))),
    right=max(1, int(RIGHT_CONTEXT_SECS / (feature_stride * enc_sub))),
)
context_samples = ContextSize(
    left=context_enc.left * encoder_frame2audio_samples,
    chunk=context_enc.chunk * encoder_frame2audio_samples,
    right=context_enc.right * encoder_frame2audio_samples,
)

print(
    f"[stream] enc_hop≈{enc_hop_ms:.1f} ms | "
    f"chunk≈{context_samples.chunk/SAMPLE_RATE*1000:.0f} ms | "
    f"right≈{context_samples.right/SAMPLE_RATE*1000:.0f} ms | "
    f"left≈{context_samples.left/SAMPLE_RATE:.1f} s"
)

# ---- helpers ----
ASCII_RANGES = tuple(range(9, 14)) + (32,)

def _looks_like_text(buf: bytes) -> bool:
    if not buf:
        return False
    printable = sum((32 <= b < 127) or (b in ASCII_RANGES) for b in buf)
    return (printable / len(buf)) > 0.9

def _frame_ok(n: int) -> bool:
    return (n % 2 == 0) or (n % 4 == 0)

def _bytes_to_f32(buf: bytes) -> torch.Tensor:
    # int16 little endian → float32 in [-1,1], [1,T]
    audio_i16 = np.frombuffer(buf, dtype="<i2")
    audio = (audio_i16.astype(np.float32) / 32768.0)
    return torch.from_numpy(audio.copy()).unsqueeze(0).to(device=device, dtype=torch.float32)

@dataclass
class StreamState:
    pcm: bytearray
    # streaming buffer and decoding state
    sab: Optional[StreamingBatchedAudioBuffer]
    dec_state: Optional[object]
    bhyps: Optional[object]
    # book-keeping
    last_text: str
    first_done: bool
    def __init__(self) -> None:
        self.pcm = bytearray()
        self.sab = StreamingBatchedAudioBuffer(
            batch_size=1, context_samples=context_samples, dtype=torch.float32, device=device
        )
        self.dec_state = None
        self.bhyps = None
        self.last_text = ""
        self.first_done = False

async def _process_ready_chunks(state: StreamState) -> Optional[str]:
    """
    Consume enough audio to form (first) [chunk+right] or subsequent [chunk] and decode.
    Returns any newly committed text delta.
    """
    emitted: Optional[str] = None

    # How many *new* samples we need to push into SAB to form next hop
    need = (context_samples.chunk + context_samples.right) if not state.first_done else context_samples.chunk

    # While we have enough PCM for at least one hop, process hops
    while len(state.pcm) >= need * BYTES_PER_SAMPLE:
        # 1) Move bytes → float32 tensor, clip exactly `need` samples
        to_take_bytes = need * BYTES_PER_SAMPLE
        window = state.pcm[:to_take_bytes]
        del state.pcm[:to_take_bytes]
        audio_t = _bytes_to_f32(window)
        length_t = torch.tensor([audio_t.shape[1]], device=device, dtype=torch.int64)

        if getattr(state, "_dbg_seen", 0) < MAX_DEBUG_STEPS:
            a = audio_t[0].detach().float().cpu().numpy()
            print(f"[hop] samples={audio_t.shape[1]} f32.min={a.min():.3f} f32.max={a.max():.3f}")

        async with gpu_semaphore:
            with torch.inference_mode(), torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                # 2) Add audio to streaming buffer
                is_last_chunk = False
                state.sab.add_audio_batch_(
                    audio_t,
                    audio_lengths=length_t,
                    is_last_chunk=is_last_chunk,
                    is_last_chunk_batch=torch.tensor([is_last_chunk], device=device),
                )

                # 3) Run encoder on full [left+chunk+right] window (inside SAB)
                enc, enc_len = model(
                    input_signal=state.sab.samples,
                    input_signal_length=state.sab.context_size_batch.total(),
                )
                # enc: [B, C, T] -> [B, T, C]
                enc = enc.transpose(1, 2).contiguous()

                # 4) Convert SAB’s context to encoder frames, drop left
                enc_ctx = state.sab.context_size.subsample(encoder_frame2audio_samples)
                enc_ctx_b = state.sab.context_size_batch.subsample(encoder_frame2audio_samples)
                enc = enc[:, enc_ctx.left :]

                # 5) Decode *chunk* frames only (exclude right-context)
                if enc_ctx.chunk > 0:
                    out_len = enc_ctx_b.chunk  # [B] number of frames to commit this hop
                    chunk_bhyps, _, state.dec_state = decoding_computer(
                        x=enc, out_len=out_len, prev_batched_state=state.dec_state
                    )
                    # merge into running hyps
                    if state.bhyps is None:
                        state.bhyps = chunk_bhyps
                    else:
                        state.bhyps.merge_(chunk_bhyps)

        # 6) Convert to text + emit delta
        if state.bhyps is not None:
            hyps = batched_hyps_to_hypotheses(state.bhyps, None, batch_size=1)
            text = model.tokenizer.ids_to_text(hyps[0].y_sequence.tolist())
            if len(text) >= len(state.last_text) + MIN_EMIT_CHARS:
                delta = text[len(state.last_text):]
                emitted = (emitted or "") + delta
                state.last_text = text
                if getattr(state, "_dbg_seen", 0) < MAX_DEBUG_STEPS:
                    print(f"[emit] +'{delta[:40]}'  total='{text[:60]}'")

        state._dbg_seen = getattr(state, "_dbg_seen", 0) + 1
        state.first_done = True
        need = context_samples.chunk  # next hops only need chunk

    return emitted

async def _finalize(state: StreamState) -> str:
    # pad right-context to flush
    if len(state.pcm) > 0:
        state.pcm.extend(b"\x00" * (context_samples.right * BYTES_PER_SAMPLE))
        # push remaining (may be < chunk)
        audio_t = _bytes_to_f32(bytes(state.pcm))
        length_t = torch.tensor([audio_t.shape[1]], device=device, dtype=torch.int64)
        async with gpu_semaphore:
            with torch.inference_mode(), torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                state.sab.add_audio_batch_(
                    audio_t,
                    audio_lengths=length_t,
                    is_last_chunk=True,
                    is_last_chunk_batch=torch.tensor([True], device=device),
                )
                enc, enc_len = model(
                    input_signal=state.sab.samples,
                    input_signal_length=state.sab.context_size_batch.total(),
                )
                enc = enc.transpose(1, 2).contiguous()
                enc_ctx = state.sab.context_size.subsample(encoder_frame2audio_samples)
                enc_ctx_b = state.sab.context_size_batch.subsample(encoder_frame2audio_samples)
                enc = enc[:, enc_ctx.left :]
                if enc_ctx.chunk + enc_ctx.right > 0:
                    out_len = enc_ctx_b.chunk + enc_ctx_b.right  # commit the rest
                    chunk_bhyps, _, state.dec_state = decoding_computer(
                        x=enc, out_len=out_len, prev_batched_state=state.dec_state
                    )
                    if state.bhyps is None:
                        state.bhyps = chunk_bhyps
                    else:
                        state.bhyps.merge_(chunk_bhyps)

    text = ""
    if state.bhyps is not None:
        hyps = batched_hyps_to_hypotheses(state.bhyps, None, batch_size=1)
        text = model.tokenizer.ids_to_text(hyps[0].y_sequence.tolist())

    # reset
    state.pcm.clear()
    state.sab = StreamingBatchedAudioBuffer(
        batch_size=1, context_samples=context_samples, dtype=torch.float32, device=device
    )
    state.dec_state = None
    state.bhyps = None
    state.last_text = ""
    state.first_done = False
    return text

# ----------------- FastAPI -----------------
app = FastAPI()

@app.on_event("startup")
def _warmup() -> None:
    try:
        with torch.inference_mode():
            # encoder warm
            x = torch.zeros(1, int(0.5 * SAMPLE_RATE), device=device, dtype=torch.float32)
            l = torch.tensor([x.shape[1]], device=device, dtype=torch.int64)
            proc, plen = model.preprocessor(input_signal=x, length=l)
            enc, elen = model.encoder(audio_signal=proc, length=plen)
            enc_bt = enc.transpose(1, 2).contiguous()
            # one label-loop step to build kernels/graphs
            _ = decoding_computer(x=enc_bt[:, :1, :], out_len=torch.tensor([1], device=device), prev_batched_state=None)
        print(f"[warmup] ok | enc_hop≈{enc_hop_ms:.1f}ms | chunk≈{context_samples.chunk/SAMPLE_RATE*1000:.0f}ms | rc≈{context_samples.right/SAMPLE_RATE*1000:.0f}ms")
    except Exception as e:
        print(f"[warmup] non-fatal: {e}")

@app.get("/status")
async def status() -> PlainTextResponse:
    return PlainTextResponse("ok", status_code=200)

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    state = StreamState()
    try:
        while True:
            msg = await ws.receive_bytes()

            if msg.startswith(CONTROL_PREFIX):
                cmd = msg[len(CONTROL_PREFIX):]
                if cmd == CTRL_EOS:
                    final = await _finalize(state)
                    await ws.send_text(json.dumps({"type": "final", "text": final}))
                elif cmd == CTRL_RESET:
                    _ = await _finalize(state)
                    await ws.send_text(json.dumps({"type": "reset"}))
                continue

            if _looks_like_text(msg) or not _frame_ok(len(msg)):
                try:
                    await ws.close(code=1003, reason="Non-PCM payload")
                finally:
                    return

            state.pcm.extend(msg)
            delta = await _process_ready_chunks(state)
            if delta:
                await ws.send_text(json.dumps({"type": "partial", "text": delta}))

    except WebSocketDisconnect:
        try:
            final = await _finalize(state)
            await ws.send_text(json.dumps({"type": "final", "text": final}))
        except Exception:
            pass
    except Exception as exc:
        try:
            await ws.close(code=1011, reason=str(exc))
        except Exception:
            pass

__all__ = ["app"]
