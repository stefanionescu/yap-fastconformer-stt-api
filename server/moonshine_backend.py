from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from inspect import signature
from typing import Sequence

import numpy as np

from .config import Config

_LOG = logging.getLogger(__name__)

try:  # Optional dependency installed via requirements
    from moonshine_onnx import MoonshineOnnxModel, load_tokenizer  # type: ignore
except ImportError:  # pragma: no cover - enforced during startup
    MoonshineOnnxModel = None  # type: ignore
    load_tokenizer = None  # type: ignore


@dataclass(frozen=True)
class ModelInfo:
    model_name: str
    providers: tuple[str, ...]
    engine_dir: str | None


class MoonshineBackend:
    def __init__(self, cfg: Config) -> None:
        if MoonshineOnnxModel is None or load_tokenizer is None:
            raise RuntimeError(
                "useful-moonshine-onnx (v1.0.0) must be installed to run the ONNX backend"
            )

        init_kwargs = self._build_constructor_kwargs(cfg)

        _LOG.info(
            "Initialising Moonshine ONNX model %s (providers=%s)",
            cfg.model_name,
            ",".join(cfg.onnx_providers) if cfg.onnx_providers else "default",
        )
        self._model = MoonshineOnnxModel(model_name=cfg.model_name, **init_kwargs)
        self._tokenizer = load_tokenizer()
        self._apply_provider_preferences(cfg)
        self._min_samples = self._resolve_min_samples(cfg.sample_rate)
        self._info = ModelInfo(
            model_name=cfg.model_name,
            providers=tuple(cfg.onnx_providers),
            engine_dir=cfg.onnx_engine_dir,
        )
        warmup_n = (
            max(int(cfg.warmup_seconds * cfg.sample_rate), self._min_samples)
            if cfg.warmup_seconds > 0
            else self._min_samples
        )
        self._warmup(warmup_n)
        _LOG.info(
            "Moonshine ONNX ready; min_samples=%d (%.1f ms @16kHz)",
            self._min_samples,
            1000.0 * self._min_samples / 16000.0,
        )

    @property
    def info(self) -> ModelInfo:
        return self._info

    @property
    def min_samples(self) -> int:
        return self._min_samples

    def transcribe(self, audios: Sequence[np.ndarray]) -> list[str]:
        if not audios:
            return []
        batch = self._prepare_batch(audios)
        stacked = np.stack(batch, axis=0).astype(np.float32)
        tokens = self._model.generate(stacked)
        texts = self._tokenizer.decode_batch(tokens)
        return [t.strip() for t in texts]

    def _warmup(self, samples: int) -> None:
        samples = max(samples, 1)
        audio = np.zeros((1, samples), dtype=np.float32)
        self._model.generate(audio)

    def _prepare_batch(self, audios: Sequence[np.ndarray]) -> list[np.ndarray]:
        cleaned: list[np.ndarray] = []
        for audio in audios:
            if audio.size == 0:
                cleaned.append(np.zeros(self._min_samples, dtype=np.float32))
                continue
            arr = np.asarray(audio, dtype=np.float32)
            if arr.ndim != 1:
                arr = arr.reshape(-1)
            if arr.size < self._min_samples:
                arr = np.pad(arr, (0, self._min_samples - arr.size), mode="constant")
            cleaned.append(arr)
        return cleaned

    def _resolve_min_samples(self, sample_rate: int) -> int:
        env_min = int(os.environ.get("MIN_SAMPLES", "0") or "0")
        if env_min > 0:
            return env_min
        return self._calibrate_min_samples(sample_rate)

    def _calibrate_min_samples(self, _: int) -> int:
        candidates = [512, 1024, 1536, 2048, 3072, 4096, 6144, 8192]
        for n in candidates:
            try:
                audio = np.zeros((1, n), dtype=np.float32)
                _ = self._model.generate(audio)
                return n
            except Exception as exc:  # pragma: no cover - runtime dependent
                if "Kernel size" not in str(exc):
                    _LOG.warning("Calibration stopped at %d samples: %s", n, exc)
                    return 4096
                continue
        return 4096

    def _build_constructor_kwargs(self, cfg: Config) -> dict[str, object]:
        supported = set(signature(MoonshineOnnxModel.__init__).parameters)
        kwargs: dict[str, object] = {}
        if "engine_dir" in supported and cfg.onnx_engine_dir:
            kwargs["engine_dir"] = cfg.onnx_engine_dir
        return kwargs

    def _build_provider_option_list(
        self,
        providers: Sequence[str] | None,
        options_map: dict[str, dict[str, str]] | None,
    ) -> list[dict[str, str]]:
        if not providers:
            return []
        if not options_map:
            return [{} for _ in providers]
        return [options_map.get(p, {}) for p in providers]

    def _apply_provider_preferences(self, cfg: Config) -> None:
        providers = list(cfg.onnx_providers)
        if not providers:
            return
        session = (
            getattr(self._model, "session", None)
            or getattr(self._model, "_session", None)
            or getattr(self._model, "_inference_session", None)
        )
        if session is None or not hasattr(session, "set_providers"):
            return
        provider_options = self._build_provider_option_list(providers, cfg.onnx_provider_options)
        try:
            if provider_options:
                session.set_providers(providers, provider_options)
            else:
                session.set_providers(providers)
        except Exception as exc:  # pragma: no cover - best effort
            _LOG.warning("Failed to set ONNX providers %s: %s", providers, exc)


__all__ = ["MoonshineBackend", "ModelInfo"]
