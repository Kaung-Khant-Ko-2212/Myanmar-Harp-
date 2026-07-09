from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PitchEstimateResult:
    f0_hz: float | None
    pitch_conf: float | None
    backend: str
    status: str
    debug: dict[str, Any]


class PitchEstimator(ABC):
    backend_name: str

    @abstractmethod
    def estimate(
        self,
        audio: np.ndarray,
        sr: int,
        *,
        t0_sec: float,
        t1_sec: float,
        min_f0_hz: float,
        max_f0_hz: float,
    ) -> PitchEstimateResult:
        raise NotImplementedError


def _slice_audio(audio: np.ndarray, sr: int, t0_sec: float, t1_sec: float) -> np.ndarray:
    n = int(audio.shape[0])
    i0 = max(0, min(n, int(round(float(t0_sec) * sr))))
    i1 = max(i0, min(n, int(round(float(t1_sec) * sr))))
    return np.asarray(audio[i0:i1], dtype=np.float32)


def _robust_pitch_summary(f0s: np.ndarray, confs: np.ndarray | None = None) -> tuple[float | None, float | None, dict[str, Any]]:
    f0s = np.asarray(f0s, dtype=np.float32)
    valid = np.isfinite(f0s) & (f0s > 0)
    if not np.any(valid):
        return None, None, {"valid_count": 0}
    f0_valid = f0s[valid]

    if confs is None:
        weights = np.ones_like(f0_valid, dtype=np.float32)
        conf_valid = None
    else:
        conf_arr = np.asarray(confs, dtype=np.float32)
        conf_valid = conf_arr[valid]
        conf_valid = np.clip(conf_valid, 0.0, 1.0)
        weights = np.maximum(conf_valid, 1e-3)

    logf = np.log(np.maximum(f0_valid, 1e-6))
    order = np.argsort(logf)
    logf = logf[order]
    weights = weights[order]
    cum = np.cumsum(weights)
    cutoff = 0.5 * float(cum[-1])
    idx = int(np.searchsorted(cum, cutoff))
    idx = max(0, min(idx, logf.shape[0] - 1))
    f0_med = float(np.exp(logf[idx]))

    pitch_conf = float(np.mean(conf_valid)) if conf_valid is not None else 1.0
    debug = {
        "valid_count": int(f0_valid.shape[0]),
        "f0_median_hz": f0_med,
        "f0_mean_hz": float(np.mean(f0_valid)),
        "f0_std_hz": float(np.std(f0_valid)),
        "pitch_conf_mean": pitch_conf,
    }
    return f0_med, pitch_conf, debug


class CrepePitchEstimator(PitchEstimator):
    backend_name = "crepe"

    def estimate(
        self,
        audio: np.ndarray,
        sr: int,
        *,
        t0_sec: float,
        t1_sec: float,
        min_f0_hz: float,
        max_f0_hz: float,
    ) -> PitchEstimateResult:
        segment = _slice_audio(audio, sr, t0_sec, t1_sec)
        if segment.size < max(32, int(0.02 * sr)):
            return PitchEstimateResult(None, None, self.backend_name, "no_pitch", {"reason": "short_window"})
        try:
            import crepe  # type: ignore
        except Exception as exc:  # noqa: BLE001
            return PitchEstimateResult(None, None, self.backend_name, "backend_unavailable", {"error": str(exc)})

        try:
            step_ms = 10
            pred = crepe.predict(
                segment.astype(np.float32, copy=False),
                int(sr),
                step_size=step_ms,
                viterbi=True,
                verbose=0,
            )
            if isinstance(pred, tuple) and len(pred) >= 3:
                _, freqs, confs = pred[:3]
            else:
                return PitchEstimateResult(None, None, self.backend_name, "backend_error", {"reason": "unexpected_crepe_output"})
            f0s = np.asarray(freqs, dtype=np.float32)
            conf_arr = np.asarray(confs, dtype=np.float32)
            mask = np.isfinite(f0s) & (f0s >= float(min_f0_hz)) & (f0s <= float(max_f0_hz))
            f0, conf, dbg = _robust_pitch_summary(f0s[mask], conf_arr[mask] if np.any(mask) else conf_arr[:0])
            return PitchEstimateResult(f0, conf, self.backend_name, "ok" if f0 is not None else "no_pitch", {"step_ms": step_ms, **dbg})
        except Exception as exc:  # noqa: BLE001
            return PitchEstimateResult(None, None, self.backend_name, "backend_error", {"error": str(exc)})


class TorchCrepePitchEstimator(PitchEstimator):
    backend_name = "torchcrepe"

    def estimate(
        self,
        audio: np.ndarray,
        sr: int,
        *,
        t0_sec: float,
        t1_sec: float,
        min_f0_hz: float,
        max_f0_hz: float,
    ) -> PitchEstimateResult:
        segment = _slice_audio(audio, sr, t0_sec, t1_sec)
        if segment.size < max(32, int(0.02 * sr)):
            return PitchEstimateResult(None, None, self.backend_name, "no_pitch", {"reason": "short_window"})
        try:
            import torch  # type: ignore
            import torchcrepe  # type: ignore
        except Exception as exc:  # noqa: BLE001
            return PitchEstimateResult(None, None, self.backend_name, "backend_unavailable", {"error": str(exc)})

        try:
            hop_length = max(64, int(round(sr * 0.01)))
            x = torch.from_numpy(segment.astype(np.float32, copy=False)).unsqueeze(0)
            out = torchcrepe.predict(
                x,
                int(sr),
                hop_length,
                float(min_f0_hz),
                float(max_f0_hz),
                model="tiny",
                batch_size=256,
                device="cpu",
                return_periodicity=True,
            )
            if isinstance(out, tuple) and len(out) >= 2:
                pitch_t, periodicity_t = out[:2]
            else:
                pitch_t = out
                periodicity_t = None
            f0s = pitch_t.detach().cpu().numpy().reshape(-1).astype(np.float32)
            confs = None
            if periodicity_t is not None:
                confs = periodicity_t.detach().cpu().numpy().reshape(-1).astype(np.float32)
            mask = np.isfinite(f0s) & (f0s >= float(min_f0_hz)) & (f0s <= float(max_f0_hz))
            f0, conf, dbg = _robust_pitch_summary(f0s[mask], (confs[mask] if confs is not None and np.any(mask) else None))
            return PitchEstimateResult(f0, conf, self.backend_name, "ok" if f0 is not None else "no_pitch", {"hop_length": int(hop_length), **dbg})
        except Exception as exc:  # noqa: BLE001
            return PitchEstimateResult(None, None, self.backend_name, "backend_error", {"error": str(exc)})


class YinPitchEstimator(PitchEstimator):
    backend_name = "yin"

    def estimate(
        self,
        audio: np.ndarray,
        sr: int,
        *,
        t0_sec: float,
        t1_sec: float,
        min_f0_hz: float,
        max_f0_hz: float,
    ) -> PitchEstimateResult:
        segment = _slice_audio(audio, sr, t0_sec, t1_sec)
        if segment.size < max(64, int(0.03 * sr)):
            return PitchEstimateResult(None, None, self.backend_name, "no_pitch", {"reason": "short_window"})
        try:
            import librosa  # type: ignore
        except Exception as exc:  # noqa: BLE001
            return PitchEstimateResult(None, None, self.backend_name, "backend_unavailable", {"error": str(exc)})

        try:
            frame_length = min(max(2048, int(0.04 * sr)), max(512, int(segment.size)))
            hop_length = max(64, int(frame_length // 4))
            f0s = librosa.yin(
                segment.astype(np.float32, copy=False),
                fmin=float(min_f0_hz),
                fmax=float(max_f0_hz),
                sr=int(sr),
                frame_length=int(frame_length),
                hop_length=int(hop_length),
            )
            f0s = np.asarray(f0s, dtype=np.float32)
            valid = np.isfinite(f0s) & (f0s >= float(min_f0_hz)) & (f0s <= float(max_f0_hz))
            if not np.any(valid):
                return PitchEstimateResult(None, None, self.backend_name, "no_pitch", {"frame_length": int(frame_length), "hop_length": int(hop_length), "valid_count": 0})

            f0_valid = f0s[valid]
            med = float(np.median(f0_valid))
            mad = float(np.median(np.abs(f0_valid - med)))
            rel_mad = mad / max(med, 1e-6)
            conf_proxy = float(np.clip(1.0 - (rel_mad * 10.0), 0.0, 1.0))
            f0, _, dbg = _robust_pitch_summary(f0_valid, None)
            dbg.update(
                {
                    "frame_length": int(frame_length),
                    "hop_length": int(hop_length),
                    "confidence_proxy": conf_proxy,
                    "confidence_proxy_type": "yin_stability",
                    "rel_mad": rel_mad,
                }
            )
            return PitchEstimateResult(f0, conf_proxy, self.backend_name, "ok" if f0 is not None else "no_pitch", dbg)
        except Exception as exc:  # noqa: BLE001
            return PitchEstimateResult(None, None, self.backend_name, "backend_error", {"error": str(exc)})


def build_pitch_estimators(preferred_backend: str | None) -> list[PitchEstimator]:
    preferred = (preferred_backend or "").strip().lower()
    factories = {
        "crepe": CrepePitchEstimator,
        "torchcrepe": TorchCrepePitchEstimator,
        "yin": YinPitchEstimator,
    }
    ordered: list[str] = []
    if preferred in factories:
        ordered.append(preferred)
    for name in ("torchcrepe", "crepe", "yin"):
        if name not in ordered:
            ordered.append(name)
    return [factories[name]() for name in ordered]


def estimate_pitch_with_fallbacks(
    audio: np.ndarray,
    sr: int,
    *,
    t0_sec: float,
    t1_sec: float,
    min_f0_hz: float,
    max_f0_hz: float,
    preferred_backend: str | None = None,
) -> PitchEstimateResult:
    last: PitchEstimateResult | None = None
    for estimator in build_pitch_estimators(preferred_backend):
        result = estimator.estimate(
            audio,
            sr,
            t0_sec=t0_sec,
            t1_sec=t1_sec,
            min_f0_hz=min_f0_hz,
            max_f0_hz=max_f0_hz,
        )
        last = result
        if result.status == "ok":
            return result
    return last or PitchEstimateResult(None, None, preferred_backend or "unknown", "no_pitch", {"reason": "no_backends"})

