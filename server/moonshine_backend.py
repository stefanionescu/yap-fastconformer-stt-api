from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

try:  # Optional dependency for int8 loading
    from transformers import BitsAndBytesConfig  # type: ignore
except ImportError:  # pragma: no cover - optional
    BitsAndBytesConfig = None  # type: ignore[misc,assignment]

try:
    from transformers import MoonshineForConditionalGeneration as _MoonshineModel
except ImportError:  # pragma: no cover - older transformers
    _MoonshineModel = None

from .config import Config

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelInfo:
    model_id: str
    device: torch.device
    precision: str
    autocast_dtype: torch.dtype | None


class MoonshineBackend:
    def __init__(self, cfg: Config) -> None:
        device = self._resolve_device(cfg)
        precision = cfg.precision
        self._autocast_dtype = self._resolve_autocast_dtype(cfg)
        self._info = ModelInfo(
            model_id=cfg.model_id,
            device=device,
            precision=precision,
            autocast_dtype=self._autocast_dtype,
        )
        self._processor = AutoProcessor.from_pretrained(
            cfg.model_id,
            trust_remote_code=True,
        )
        self._model = self._load_model(cfg, device)
        self._model.eval()
        self._target_dtype = self._detect_parameter_dtype()
        _LOG.info("Loaded Moonshine model %s on %s (%s)", cfg.model_id, device, precision)
        if cfg.warmup_seconds > 0:
            self._warmup(int(cfg.warmup_seconds * cfg.sample_rate))

    @property
    def info(self) -> ModelInfo:
        return self._info

    def _resolve_device(self, cfg: Config) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():  # pragma: no cover - mac fallback
            return torch.device("mps")
        if not cfg.allow_cpu_fallback:
            raise RuntimeError("No GPU found and CPU fallback disabled")
        return torch.device("cpu")

    def _resolve_autocast_dtype(self, cfg: Config) -> torch.dtype | None:
        if cfg.precision == "fp16":
            return torch.float16
        if cfg.precision == "bf16":
            return torch.bfloat16
        return None

    def _detect_parameter_dtype(self) -> torch.dtype | None:
        try:
            return next(self._model.parameters()).dtype
        except StopIteration:  # pragma: no cover - unlikely
            return None

    def _load_model(self, cfg: Config, device: torch.device) -> torch.nn.Module:
        precision = cfg.precision
        kwargs = {"trust_remote_code": True}
        quant_config = None
        if precision == "int8":
            if BitsAndBytesConfig is None:
                raise RuntimeError(
                    "bitsandbytes is required for int8 loading but is not available"
                )
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
            kwargs.update(
                {
                    "device_map": "auto",
                    "quantization_config": quant_config,
                }
            )
        else:
            dtype = {
                "fp16": torch.float16,
                "bf16": torch.bfloat16,
                "fp32": torch.float32,
            }.get(precision, torch.float16)
            kwargs.update({"dtype": dtype})
        model_cls = _MoonshineModel or AutoModelForSpeechSeq2Seq
        model = model_cls.from_pretrained(cfg.model_id, **kwargs)
        if precision != "int8":
            model.to(device)
        return model

    def _warmup(self, samples: int) -> None:
        samples = max(samples, 1)
        audio = np.zeros((1, samples), dtype=np.float32)
        with torch.no_grad():
            inputs = self._processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
            inputs = self._to_device_dtype(inputs)
            with self._autocast_context():
                tokens = self._model.generate(**inputs, max_new_tokens=1)
            _ = self._processor.batch_decode(tokens, skip_special_tokens=True)
        _LOG.info("Warmup complete (%d samples)", samples)

    @staticmethod
    def _prepare_batch(audios: Sequence[np.ndarray]) -> list[np.ndarray]:
        min_samples = 512  # ensure at least 32 ms of audio at 16 kHz
        cleaned: list[np.ndarray] = []
        for idx, audio in enumerate(audios):
            if audio.size == 0:
                cleaned.append(np.zeros(min_samples, dtype=np.float32))
                continue
            arr = np.asarray(audio, dtype=np.float32)
            if arr.ndim != 1:
                arr = arr.reshape(-1)
            if arr.size < min_samples:
                arr = np.pad(arr, (0, min_samples - arr.size), mode="constant")
            cleaned.append(arr)
        return cleaned

    def transcribe(self, audios: Sequence[np.ndarray]) -> list[str]:
        if not audios:
            return []
        batch = self._prepare_batch(audios)
        with torch.no_grad():
            inputs = self._processor(batch, sampling_rate=16000, return_tensors="pt", padding=True)
            inputs = self._to_device_dtype(inputs)
            with self._autocast_context():
                tokens = self._model.generate(**inputs)
            texts = self._processor.batch_decode(tokens, skip_special_tokens=True)
        return [t.strip() for t in texts]

    def _to_device_dtype(self, tensors: dict[str, object]) -> dict[str, object]:
        for key, value in list(tensors.items()):
            if not isinstance(value, torch.Tensor):
                continue
            value = value.to(self._info.device)
            if self._info.precision != "int8" and self._target_dtype is not None:
                value = value.to(self._target_dtype)
            tensors[key] = value
        return tensors

    def _autocast_context(self):
        autocast_dtype = self._info.autocast_dtype
        enabled = autocast_dtype is not None and self._info.device.type in {"cuda", "mps"}
        if enabled:
            return torch.autocast(self._info.device.type, dtype=autocast_dtype)
        return contextlib.nullcontext()


__all__ = ["MoonshineBackend", "ModelInfo"]
