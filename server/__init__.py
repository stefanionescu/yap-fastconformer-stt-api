from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
from omegaconf import OmegaConf
from nemo.collections.asr.parts.submodules.transducer_decoding.label_looping_base import (
    GreedyBatchedLabelLoopingComputerBase,
)
from nemo.collections.asr.parts.utils.rnnt_utils import batched_hyps_to_hypotheses
import torch
import nemo.collections.asr as nemo_asr
import nemo

SAMPLE_RATE = 16_000
BYTES_PER_SAMPLE = 2
STEP_MS = int(os.getenv("STEP_MS", "40"))
RIGHT_CONTEXT_MS = int(os.getenv("RIGHT_CONTEXT_MS", "160"))
MIN_EMIT_CHARS = int(os.getenv("MIN_EMIT_CHARS", "1"))
MAX_INFLIGHT_STEPS = int(os.getenv("MAX_INFLIGHT_STEPS", "4"))

CONTROL_PREFIX = b"__CTRL__:"
CTRL_EOS = b"EOS"
CTRL_RESET = b"RESET"

print("Loading NeMo Parakeet-TDT-0.6b-v3 …")
model = nemo_asr.models.ASRModel.from_pretrained(
    model_name="nvidia/parakeet-tdt-0.6b-v3"
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
print(f"[device] {device}")
model.eval()
model.freeze()
print(f"[nemo] {nemo.__version__}")

# Configure RNNT decoding - label-looping stateful greedy
_decoding_cfg = OmegaConf.create({
    "strategy": "greedy_batch",
    "preserve_alignments": False,
    "compute_timestamps": False,
    "fused_batch_size": -1,
    "greedy": {
        "loop_labels": True,
        "max_symbols_per_step": 10,
        "use_cuda_graph_decoder": True,
    },
})
model.change_decoding_strategy(_decoding_cfg)

# one-time handle to the decoder computer
decoding_computer: GreedyBatchedLabelLoopingComputerBase = model.decoding.decoding.decoding_computer  # type: ignore

# Set featurizer inference flags to avoid any training-time quirks
try:
    if hasattr(model, "preprocessor") and hasattr(model.preprocessor, "featurizer"):
        model.preprocessor.featurizer.dither = 0.0
        model.preprocessor.featurizer.pad_to = 0
except Exception:
    pass

gpu_semaphore = asyncio.Semaphore(MAX_INFLIGHT_STEPS)


# --- Encoder hop and aligned sizes ---
SR = SAMPLE_RATE
assert int(model._cfg.preprocessor.sample_rate) == SAMPLE_RATE, "SR mismatch"
try:
    stride_sec = float(model._cfg.preprocessor.window_stride)
except Exception:
    stride_sec = 0.01  # sensible default
sub = int(getattr(model.encoder, "subsampling_factor", 0))
assert hasattr(model.encoder, "subsampling_factor") and sub > 0

features_frame2samples = int(SR * stride_sec)
# Make divisible by subsampling factor (NeMo does this internally)
if sub > 0:
    features_frame2samples = (features_frame2samples // sub) * sub
ENC_HOP_SAMPLES = features_frame2samples * max(sub, 1)
ENC_HOP_MS = 1000.0 * ENC_HOP_SAMPLES / SR

def _align_samples(req_ms: int) -> int:
    req_samples = int(SR * (req_ms / 1000.0))
    hop = max(ENC_HOP_SAMPLES, 1)
    k = max(1, (req_samples + hop - 1) // hop)  # ceil to >=1 hop
    return k * hop

# Force 1-hop chunk and 1-hop right-context to minimize commit latency
CHUNK_SAMPLES = ENC_HOP_SAMPLES
RC_SAMPLES = ENC_HOP_SAMPLES

CHUNK_BYTES = CHUNK_SAMPLES * BYTES_PER_SAMPLE
RC_BYTES = RC_SAMPLES * BYTES_PER_SAMPLE

# Derive frame counts for decoder
CHUNK_FRAMES = max(1, CHUNK_SAMPLES // ENC_HOP_SAMPLES)
RC_FRAMES = max(1, RC_SAMPLES // ENC_HOP_SAMPLES)

print(f"[stream] enc_hop≈{ENC_HOP_MS:.1f} ms  chunk={CHUNK_SAMPLES/SR*1000:.1f} ms  right={RC_SAMPLES/SR*1000:.1f} ms")


@dataclass
class StreamState:
    pcm: bytearray
    last_text: str
    dec_state: Optional[object]
    hyps: Optional[object]

    def __init__(self) -> None:
        self.pcm = bytearray()
        self.last_text = ""
        self.dec_state = None
        self.hyps = None


MAX_DEBUG_STEPS = int(os.getenv("DEBUG_STEPS", "8"))


ASCII_RANGES = tuple(range(9, 14)) + (32,)


def _looks_like_text(buf: bytes) -> bool:
    if not buf:
        return False
    printable = sum((32 <= b < 127) or (b in ASCII_RANGES) for b in buf)
    return (printable / len(buf)) > 0.9


def _frame_ok(n: int) -> bool:
    # accept either s16 (2-byte) or f32 (4-byte) frames
    return (n % 2 == 0) or (n % 4 == 0)


def _bytes_to_f32_mono_tensor(buf: bytes, device: torch.device) -> torch.Tensor:
    """
    Decode s16le PCM chunk into a 1xT float32 tensor in [-1, 1].
    Strict s16le → f32 in [-1, 1]
    """
    audio_i16 = np.frombuffer(buf, dtype='<i2')  # little-endian int16
    audio = (audio_i16.astype(np.float32) / 32768.0)
    return torch.from_numpy(audio.copy()).to(device=device, dtype=torch.float32).unsqueeze(0)


async def rnnt_step(state: StreamState) -> Optional[str]:
    emitted: Optional[str] = None
    while len(state.pcm) >= CHUNK_BYTES + RC_BYTES:
        window = state.pcm[: CHUNK_BYTES + RC_BYTES]
        audio_t = _bytes_to_f32_mono_tensor(window, device).contiguous()
        length_t = torch.tensor([audio_t.shape[1]], device=device, dtype=torch.int64)

        if getattr(state, "_dbg_seen", 0) < MAX_DEBUG_STEPS:
            a = audio_t[0].detach().float().cpu().numpy()
            print(f"[step] bytes={len(window)} f32.min={a.min():.3f} f32.max={a.max():.3f}")

        async with gpu_semaphore:
            with torch.inference_mode(), torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                proc, proc_len = model.preprocessor(input_signal=audio_t, length=length_t)
                enc, enc_len = model.encoder(audio_signal=proc, length=proc_len)
                enc_bt = enc.transpose(1, 2).contiguous()  # [B, T, C]

                if getattr(state, "_dbg_seen", 0) == 0:
                    print(f"[dbg] enc.shape={tuple(enc.shape)} enc_len={int(enc_len[0].item())}")

                T = int(enc_len[0].item())
                if T < (CHUNK_FRAMES + RC_FRAMES):
                    break

                # decode only the CHUNK frames, exclude right-context
                decode_bt = enc_bt[:, : T - RC_FRAMES, :]
                out_len = torch.tensor([CHUNK_FRAMES], device=device, dtype=torch.int64)

                chunk_bhyps, _, state.dec_state = decoding_computer(
                    x=decode_bt, out_len=out_len, prev_batched_state=state.dec_state
                )

                if state.hyps is None:
                    state.hyps = chunk_bhyps
                else:
                    state.hyps.merge_(chunk_bhyps)

        # Convert running BatchedHyps -> text and emit delta
        hyp_list = batched_hyps_to_hypotheses(state.hyps, None, batch_size=1) if state.hyps is not None else []
        text = model.tokenizer.ids_to_text(hyp_list[0].y_sequence.tolist()) if hyp_list else ""

        if len(text) >= len(state.last_text) + MIN_EMIT_CHARS:
            delta = text[len(state.last_text):]
            state.last_text = text
            emitted = (emitted or "") + delta
            if getattr(state, "_dbg_seen", 0) < MAX_DEBUG_STEPS:
                print(f"[step] delta='{delta[:40]}' total_len={len(text)}")

        state._dbg_seen = getattr(state, "_dbg_seen", 0) + 1
        # slide exactly one chunk; keep RC in buffer
        del state.pcm[:CHUNK_BYTES]

    return emitted


def flush_final(state: StreamState) -> str:
    text = ""
    if state.hyps is not None:
        hyp_list = batched_hyps_to_hypotheses(state.hyps, None, batch_size=1)
        text = model.tokenizer.ids_to_text(hyp_list[0].y_sequence.tolist())
    state.pcm.clear()
    state.dec_state = None
    state.hyps = None
    state.last_text = ""
    return text


async def finalize(state: StreamState) -> str:
    if len(state.pcm) > 0:
        # pad right-context
        state.pcm.extend(b"\x00" * RC_BYTES)
        audio_t = _bytes_to_f32_mono_tensor(state.pcm, device).contiguous()
        length_t = torch.tensor([audio_t.shape[1]], device=device, dtype=torch.int64)

        async with gpu_semaphore:
            with torch.inference_mode(), torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                proc, proc_len = model.preprocessor(input_signal=audio_t, length=length_t)
                enc, enc_len = model.encoder(audio_signal=proc, length=proc_len)
                enc_bt = enc.transpose(1, 2).contiguous()
                T = int(enc_len[0].item())
                if T > RC_FRAMES:
                    decode_bt = enc_bt[:, : T - RC_FRAMES, :]
                    out_len = torch.tensor([T - RC_FRAMES], device=device, dtype=torch.int64)
                    chunk_bhyps, _, state.dec_state = decoding_computer(
                        x=decode_bt, out_len=out_len, prev_batched_state=state.dec_state
                    )
                    if state.hyps is None:
                        state.hyps = chunk_bhyps
                    else:
                        state.hyps.merge_(chunk_bhyps)

    text = ""
    if state.hyps is not None:
        hyp_list = batched_hyps_to_hypotheses(state.hyps, None, batch_size=1)
        text = model.tokenizer.ids_to_text(hyp_list[0].y_sequence.tolist())

    # reset
    state.pcm.clear()
    state.dec_state = None
    state.hyps = None
    state.last_text = ""
    return text


app = FastAPI()


@app.on_event("startup")
def _warmup() -> None:
    try:
        with torch.inference_mode():
            x = torch.zeros(1, int(0.5 * SAMPLE_RATE), device=device, dtype=torch.float32)
            length_t = torch.tensor([x.shape[1]], device=device, dtype=torch.int64)
            
            proc, proc_len = model.preprocessor(input_signal=x, length=length_t)
            enc, enc_len = model.encoder(audio_signal=proc, length=proc_len)
            enc_bt = enc.transpose(1, 2).contiguous()

            print(f"[warmup] enc.shape={tuple(enc.shape)}")

            # one tiny LL step to build kernels
            _ = decoding_computer(
                x=enc_bt[:, :1, :], out_len=torch.tensor([1], device=device), prev_batched_state=None
            )
        print(f"[warmup] hop={ENC_HOP_MS:.1f}ms chunk={CHUNK_SAMPLES/SR*1000:.1f}ms rc={RC_SAMPLES/SR*1000:.1f}ms")
        print("[warmup] Complete")
    except Exception as e:
        print(f"[warmup] Error: {e}")
        import traceback
        traceback.print_exc()


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
                cmd = msg[len(CONTROL_PREFIX) :]
                if cmd == CTRL_EOS:
                    final = await finalize(state)
                    await ws.send_text(json.dumps({"type": "final", "text": final}))
                elif cmd == CTRL_RESET:
                    flush_final(state)
                    await ws.send_text(json.dumps({"type": "reset"}))
                continue

            if _looks_like_text(msg) or not _frame_ok(len(msg)):
                try:
                    await ws.close(code=1003, reason="Non-PCM payload")
                finally:
                    return

            state.pcm.extend(msg)
            delta = await rnnt_step(state)
            if delta:
                await ws.send_text(json.dumps({"type": "partial", "text": delta}))

    except WebSocketDisconnect:
        if state.hyps is not None or len(state.pcm) > 0:
            final = await finalize(state)
            try:
                await ws.send_text(json.dumps({"type": "final", "text": final}))
            except Exception:
                pass
    except Exception as exc:
        try:
            await ws.close(code=1011, reason=str(exc))
        except Exception:
            pass


__all__ = ["app"]
