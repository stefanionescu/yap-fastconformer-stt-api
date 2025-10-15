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

SAMPLE_RATE = 16_000
BYTES_PER_SAMPLE = 2
STEP_MS = int(os.getenv("STEP_MS", "80"))
RIGHT_CONTEXT_MS = int(os.getenv("RIGHT_CONTEXT_MS", "120"))
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

_decoding_cfg = OmegaConf.create(
    {
        "strategy": "greedy",
        "compute_timestamps": False,
        "preserve_alignments": False,
        "greedy": {
            "loop_labels": True,
            # only enable CUDA graphs on GPU
            "use_cuda_graph_decoder": bool(torch.cuda.is_available()),
            "max_symbols_per_step": 10,
        },
    }
)
model.change_decoding_strategy(_decoding_cfg)

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


def _bytes_to_f32_mono_tensor(buf: bytes, device: torch.device) -> torch.Tensor:
    """
    Decode either s16le or f32le PCM chunk into a 1xT float32 tensor in [-1, 1].
    Heuristics:
      * Prefer s16 if buf length divisible by 2 and s16 RMS > tiny.
      * If buf length divisible by 4 and s16 looks like silence, treat as f32.
    """
    n = len(buf)
    as_f32 = False
    looks_f32 = (n % 4) == 0

    audio_i16 = np.frombuffer(buf, dtype=np.int16)
    if audio_i16.size > 0:
        rms_i16 = float(np.sqrt((audio_i16.astype(np.float64) ** 2).mean() + 1e-12))
        if looks_f32 and rms_i16 <= 2.0:
            as_f32 = True
    else:
        as_f32 = looks_f32

    if not as_f32:
        audio = (audio_i16.astype(np.float32) / 32768.0)
    else:
        audio = np.frombuffer(buf, dtype=np.float32)
        np.clip(audio, -1.0, 1.0, out=audio)

    return torch.from_numpy(audio.copy()).to(device=device, dtype=torch.float32).unsqueeze(0)


async def rnnt_step(state: StreamState) -> Optional[str]:
    needed = _bytes_per_window(STEP_MS)
    if len(state.pcm) < needed:
        return None

    rc_bytes = _bytes_per_window(RIGHT_CONTEXT_MS)
    window = state.pcm[: needed + rc_bytes]
    audio_t = _bytes_to_f32_mono_tensor(window, device).contiguous()
    length_t = torch.tensor([audio_t.shape[1]], device=device, dtype=torch.int64)
    if getattr(state, "_dbg_seen", 0) < MAX_DEBUG_STEPS:
        a = audio_t[0].detach().float().cpu().numpy()
        print(f"[step] bytes={len(window)} f32.min={a.min():.3f} f32.max={a.max():.3f} f32.rms≈{float(np.sqrt((a*a).mean()+1e-12)):.3f}")

    async with gpu_semaphore:
        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            proc, proc_len = model.preprocessor(input_signal=audio_t, length=length_t)
            # encoder expects audio_signal and length
            enc, enc_len = model.encoder(audio_signal=proc, length=proc_len)
            out = model.decoding.rnnt_decoder_predictions_tensor(
                enc,
                enc_len,
                return_hypotheses=True,
                partial_hypotheses=[state.partial] if state.partial is not None else None,
            )
            hyps = out[0] if isinstance(out, tuple) else out
    hyp = hyps[0]
    state.partial = hyp

    text = hyp.text or ""
    delta: Optional[str] = None
    if len(text) >= len(state.last_text) + MIN_EMIT_CHARS:
        delta = text[len(state.last_text) :]
        state.last_text = text
        if getattr(state, "_dbg_seen", 0) < MAX_DEBUG_STEPS:
            print(f"[step] partial+ delta='{delta[:40]}' total_len={len(text)}")
    state._dbg_seen = getattr(state, "_dbg_seen", 0) + 1

    del state.pcm[:needed]
    return delta


def flush_final(state: StreamState) -> str:
    final_text = state.partial.text if state.partial else ""
    state.pcm.clear()
    state.partial = None
    state.last_text = ""
    return final_text


async def finalize(state: StreamState) -> str:
    if len(state.pcm) > 0:
        # Pad a bit of future context so RNNT can commit final tokens
        rc_bytes = _bytes_per_window(RIGHT_CONTEXT_MS)
        if rc_bytes > 0:
            state.pcm.extend(b"\x00" * rc_bytes)
        audio_t = _bytes_to_f32_mono_tensor(state.pcm, device).contiguous()
        length_t = torch.tensor([audio_t.shape[1]], device=device, dtype=torch.int64)
        async with gpu_semaphore:
            with torch.inference_mode(), torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                proc, proc_len = model.preprocessor(input_signal=audio_t, length=length_t)
                enc, enc_len = model.encoder(audio_signal=proc, length=proc_len)
                out = model.decoding.rnnt_decoder_predictions_tensor(
                    enc,
                    enc_len,
                    return_hypotheses=True,
                    partial_hypotheses=[state.partial] if state.partial else None,
                )
                hyps = out[0] if isinstance(out, tuple) else out
        state.partial = hyps[0]

    final = state.partial.text if state.partial else ""
    state.pcm.clear()
    state.partial = None
    state.last_text = ""
    return final


app = FastAPI()


@app.on_event("startup")
def _warmup() -> None:
    # Build kernels + graph once with a tiny buffer
    try:
        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            x = torch.zeros(1, int(0.25 * SAMPLE_RATE), device=device, dtype=torch.float32).contiguous()
            length_t = torch.tensor([x.shape[1]], device=device, dtype=torch.int64)
            proc, proc_len = model.preprocessor(input_signal=x, length=length_t)
            enc, enc_len = model.encoder(audio_signal=proc, length=proc_len)
            _ = model.decoding.rnnt_decoder_predictions_tensor(enc, enc_len, return_hypotheses=False)
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
                cmd = msg[len(CONTROL_PREFIX) :]
                if cmd == CTRL_EOS:
                    final = await finalize(state)
                    await ws.send_text(json.dumps({"type": "final", "text": final}))
                elif cmd == CTRL_RESET:
                    flush_final(state)
                    await ws.send_text(json.dumps({"type": "reset"}))
                continue

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
