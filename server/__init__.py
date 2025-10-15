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
model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v3")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"[device] torch_device={device} is_cuda={next(model.parameters()).is_cuda if hasattr(model,'parameters') else 'n/a'}")
model.eval()
model.freeze()

_decoding_cfg = OmegaConf.create(
    {
        "strategy": "greedy",
        "compute_timestamps": False,
        "preserve_alignments": False,
        "greedy": {
            "loop_labels": True,
            "use_cuda_graph_decoder": True,
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


async def rnnt_step(state: StreamState) -> Optional[str]:
    needed = _bytes_per_window(STEP_MS)
    if len(state.pcm) < needed:
        return None

    rc_bytes = _bytes_per_window(RIGHT_CONTEXT_MS)
    window = state.pcm[: needed + rc_bytes]
    audio_i16 = np.frombuffer(window, dtype=np.int16)
    # Fast sanity: are we receiving non-zero audio?
    if getattr(state, "_dbg_seen", 0) < MAX_DEBUG_STEPS:
        rms = float(np.sqrt((audio_i16.astype(np.float64) ** 2).mean() + 1e-9))
        print(f"[step] bytes={len(window)} i16.min={audio_i16.min()} i16.max={audio_i16.max()} i16.rms≈{rms:.1f}")
    audio = (audio_i16.astype(np.float32) / 32768.0)

    async with gpu_semaphore:
        hyps = model.transcribe(
            audio=[audio],
            batch_size=1,
            return_hypotheses=True,
            partial_hypothesis=[state.partial] if state.partial is not None else None,
            num_workers=0,
        )
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
        audio = (np.frombuffer(state.pcm, dtype=np.int16).astype(np.float32) / 32768.0)
        async with gpu_semaphore:
            hyps = model.transcribe(
                audio=[audio],
                batch_size=1,
                return_hypotheses=True,
                partial_hypothesis=[state.partial] if state.partial else None,
                num_workers=0,
            )
        state.partial = hyps[0]

    final = state.partial.text if state.partial else ""
    state.pcm.clear()
    state.partial = None
    state.last_text = ""
    return final


app = FastAPI()


@app.on_event("startup")
def _warmup() -> None:
    # Build CUDA kernels and CUDA graph paths with a tiny zero buffer
    x = np.zeros(int(0.25 * SAMPLE_RATE), dtype=np.float32)
    try:
        model.transcribe(audio=[x], batch_size=1, return_hypotheses=False, num_workers=0)
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
