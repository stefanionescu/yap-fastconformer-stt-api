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
    needed = _bytes_per_window(STEP_MS)
    rc_bytes = _bytes_per_window(RIGHT_CONTEXT_MS)
    if len(state.pcm) < needed + rc_bytes:
        return None

    window = state.pcm[: needed + rc_bytes]
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
            
            # Debug original encoder output shape
            if getattr(state, "_dbg_seen", 0) == 0:
                print(f"[dbg] enc.shape={tuple(enc.shape)} enc_len={int(enc_len[0])}")
            
            # CRITICAL FIX: Check if encoder output is [B, C, T] format
            # NeMo Conformer/FastConformer outputs [B, C, T], decoder needs [B, T, C]
            # We check if dim 1 looks like a feature dimension (typically 256-1024)
            # and dim 2 looks like time frames (varies with audio length)
            if enc.ndim == 3 and enc.shape[1] in range(128, 2048):
                # Likely [B, C, T] format - transpose to [B, T, C]
                enc = enc.transpose(1, 2).contiguous()
                if getattr(state, "_dbg_seen", 0) == 0:
                    print(f"[dbg] transposed to enc.shape={tuple(enc.shape)}")
            
            # Use transcribe instead of direct decoder call for streaming
            # This handles the decoder state properly
            if state.partial is None:
                # First chunk - no previous state
                hyps = model.decoding.rnnt_decoder_predictions_tensor(
                    encoder_output=enc,
                    encoded_lengths=enc_len,
                    return_hypotheses=True,
                )
            else:
                # Continuing chunks - pass previous hypothesis
                hyps = model.decoding.rnnt_decoder_predictions_tensor(
                    encoder_output=enc,
                    encoded_lengths=enc_len,
                    return_hypotheses=True,
                    partial_hypotheses=[state.partial],
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
    del state.pcm[:needed]
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
        # Pad with silence to flush remaining audio
        rc_bytes = _bytes_per_window(RIGHT_CONTEXT_MS)
        state.pcm.extend(b"\x00" * rc_bytes)
        
        audio_t = _bytes_to_f32_mono_tensor(state.pcm, device).contiguous()
        length_t = torch.tensor([audio_t.shape[1]], device=device, dtype=torch.int64)
        
        async with gpu_semaphore:
            with torch.inference_mode(), torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                proc, proc_len = model.preprocessor(input_signal=audio_t, length=length_t)
                enc, enc_len = model.encoder(audio_signal=proc, length=proc_len)
                
                # Same transpose fix
                if enc.ndim == 3 and enc.shape[1] in range(128, 2048):
                    enc = enc.transpose(1, 2).contiguous()
                
                if state.partial is None:
                    hyps = model.decoding.rnnt_decoder_predictions_tensor(
                        encoder_output=enc,
                        encoded_lengths=enc_len,
                        return_hypotheses=True,
                    )
                else:
                    hyps = model.decoding.rnnt_decoder_predictions_tensor(
                        encoder_output=enc,
                        encoded_lengths=enc_len,
                        return_hypotheses=True,
                        partial_hypotheses=[state.partial],
                    )
        
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
            
            # Same transpose fix
            if enc.ndim == 3 and enc.shape[1] in range(128, 2048):
                enc = enc.transpose(1, 2).contiguous()
                print(f"[warmup] transposed to {tuple(enc.shape)}")
            
            _ = model.decoding.rnnt_decoder_predictions_tensor(
                encoder_output=enc,
                encoded_lengths=enc_len,
                return_hypotheses=True,
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
