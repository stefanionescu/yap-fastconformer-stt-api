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
print(f"[device] {device}")
model.eval()
model.freeze()
print(f"[nemo] {nemo.__version__}")

# Configure RNNT decoding - simplified for better reliability
_decoding_cfg = OmegaConf.create({
    "strategy": "greedy",
    "preserve_alignments": False,
    "preserve_frame_confidence": False,
    "compute_timestamps": False,
    "greedy": {
        "max_symbols_per_step": 10,
        "use_cuda_graph_decoder": True,
    },
})
model.change_decoding_strategy(_decoding_cfg)

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
try:
    stride_sec = float(model._cfg.preprocessor.window_stride)
except Exception:
    stride_sec = 0.01  # sensible default
try:
    sub = int(getattr(model.encoder, "subsampling_factor", 4))
except Exception:
    sub = 4

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

# Aim for ~120 ms chunk (≈3 hops) and ~120 ms right context (≈3 hops)
CHUNK_SAMPLES = _align_samples(max(STEP_MS, int(3 * ENC_HOP_MS)))
RC_SAMPLES = _align_samples(max(RIGHT_CONTEXT_MS, int(3 * ENC_HOP_MS)))

CHUNK_BYTES = CHUNK_SAMPLES * BYTES_PER_SAMPLE
RC_BYTES = RC_SAMPLES * BYTES_PER_SAMPLE

print(f"[stream] enc_hop≈{ENC_HOP_MS:.1f} ms  chunk={CHUNK_SAMPLES/SR*1000:.1f} ms  right={RC_SAMPLES/SR*1000:.1f} ms")


@dataclass
class StreamState:
    pcm: bytearray
    partial: Optional[object]
    last_text: str

    def __init__(self) -> None:
        self.pcm = bytearray()
        self.partial = None
        self.last_text = ""


MAX_DEBUG_STEPS = int(os.getenv("DEBUG_STEPS", "8"))


def _bytes_per_window(step_ms: int) -> int:
    return int(SAMPLE_RATE * (step_ms / 1000.0)) * BYTES_PER_SAMPLE


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
    if len(state.pcm) < CHUNK_BYTES + RC_BYTES:
        return None

    window = state.pcm[: CHUNK_BYTES + RC_BYTES]
    audio_t = _bytes_to_f32_mono_tensor(window, device).contiguous()
    length_t = torch.tensor([audio_t.shape[1]], device=device, dtype=torch.int64)
    
    if getattr(state, "_dbg_seen", 0) < MAX_DEBUG_STEPS:
        a = audio_t[0].detach().float().cpu().numpy()
        print(f"[step] bytes={len(window)} f32.min={a.min():.3f} f32.max={a.max():.3f}")

    async with gpu_semaphore:
        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            # Process audio through model
            proc, proc_len = model.preprocessor(input_signal=audio_t, length=length_t)
            enc, enc_len = model.encoder(audio_signal=proc, length=proc_len)
            
            # Debug original encoder output shape (decoder expects [B, D, T])
            if getattr(state, "_dbg_seen", 0) == 0:
                print(f"[dbg] enc.shape={tuple(enc.shape)} enc_len={int(enc_len[0])}")
            
            # Require ≥2 encoder frames before decoding; otherwise wait for more audio
            if int(enc_len[0]) < 2:
                return None

            hyps = model.decoding.rnnt_decoder_predictions_tensor(
                encoder_output=enc,
                encoded_lengths=enc_len,
                return_hypotheses=True,
                partial_hypotheses=[state.partial] if state.partial is not None else None,
            )
            
    # Extract hypothesis
    hyp = hyps[0] if isinstance(hyps, (list, tuple)) else hyps
    state.partial = hyp
    
    # Get text from hypothesis
    try:
        text = hyp.text if hasattr(hyp, 'text') else ""
        if not text and hasattr(hyp, 'y_sequence'):
            text = model.tokenizer.ids_to_text(hyp.y_sequence.tolist())
    except Exception as e:
        print(f"[error] extracting text: {e}")
        text = ""
    
    # Emit delta
    delta: Optional[str] = None
    if len(text) >= len(state.last_text) + MIN_EMIT_CHARS:
        delta = text[len(state.last_text):]
        state.last_text = text
        if getattr(state, "_dbg_seen", 0) < MAX_DEBUG_STEPS:
            print(f"[step] delta='{delta[:40]}' total_len={len(text)}")
    
    state._dbg_seen = getattr(state, "_dbg_seen", 0) + 1
    # Slide by exactly one aligned chunk
    del state.pcm[:CHUNK_BYTES]
    return delta


def flush_final(state: StreamState) -> str:
    if state.partial:
        try:
            final_text = state.partial.text or model.tokenizer.ids_to_text(state.partial.y_sequence.tolist())
        except Exception:
            final_text = state.partial.text or ""
    else:
        final_text = ""
    state.pcm.clear()
    state.partial = None
    state.last_text = ""
    return final_text


async def finalize(state: StreamState) -> str:
    if len(state.pcm) == 0 and state.partial:
        # No more audio, just return what we have
        try:
            return state.partial.text or ""
        except:
            return ""
    
    if len(state.pcm) > 0:
        # Pad with ≥ right context, aligned
        PAD_BYTES = RC_BYTES
        state.pcm.extend(b"\x00" * PAD_BYTES)
        
        audio_t = _bytes_to_f32_mono_tensor(state.pcm, device).contiguous()
        length_t = torch.tensor([audio_t.shape[1]], device=device, dtype=torch.int64)
        
        async with gpu_semaphore:
            with torch.inference_mode(), torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                proc, proc_len = model.preprocessor(input_signal=audio_t, length=length_t)
                enc, enc_len = model.encoder(audio_signal=proc, length=proc_len)
                # Guard tiny tail
                hyps = None
                if int(enc_len[0]) >= 2:
                    hyps = model.decoding.rnnt_decoder_predictions_tensor(
                        encoder_output=enc,
                        encoded_lengths=enc_len,
                        return_hypotheses=True,
                        partial_hypotheses=[state.partial] if state.partial is not None else None,
                    )

        if hyps is not None:
            state.partial = hyps[0] if isinstance(hyps, (list, tuple)) else hyps
    
    # Extract final text
    try:
        final = state.partial.text if state.partial else ""
        if not final and state.partial and hasattr(state.partial, 'y_sequence'):
            final = model.tokenizer.ids_to_text(state.partial.y_sequence.tolist())
    except Exception as e:
        print(f"[error] finalizing: {e}")
        final = ""
    
    # Reset state
    state.pcm.clear()
    state.partial = None
    state.last_text = ""
    return final


app = FastAPI()


@app.on_event("startup")
def _warmup() -> None:
    try:
        with torch.inference_mode():
            x = torch.zeros(1, int(0.5 * SAMPLE_RATE), device=device, dtype=torch.float32)
            length_t = torch.tensor([x.shape[1]], device=device, dtype=torch.int64)
            
            proc, proc_len = model.preprocessor(input_signal=x, length=length_t)
            enc, enc_len = model.encoder(audio_signal=proc, length=proc_len)
            
            print(f"[warmup] enc.shape={tuple(enc.shape)}")
            
            _ = model.decoding.rnnt_decoder_predictions_tensor(
                encoder_output=enc,
                encoded_lengths=enc_len,
                return_hypotheses=False,
            )
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
        if state.partial or len(state.pcm) > 0:
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
