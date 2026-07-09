from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AudioLoadResult:
    audio: np.ndarray | None
    sr: int | None
    backend: str | None
    error: str | None
    debug: dict[str, Any]

    @property
    def ok(self) -> bool:
        return self.audio is not None and self.sr is not None and self.error is None


def _to_float32_mono(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio)
    if arr.ndim == 1:
        mono = arr
    elif arr.ndim == 2:
        # Handle both (samples, channels) and (channels, samples).
        if arr.shape[0] <= 8 and arr.shape[1] > arr.shape[0]:
            mono = arr.mean(axis=0)
        else:
            mono = arr.mean(axis=1)
    else:
        mono = arr.reshape(-1)
    mono = np.asarray(mono, dtype=np.float32)
    if mono.size == 0:
        return mono
    max_abs = float(np.max(np.abs(mono)))
    if max_abs > 1.5:
        mono = mono / max_abs
    return mono


def load_audio_mono(
    path: str | Path,
    sample_rate: int | None = None,
) -> AudioLoadResult:
    wav_path = Path(path)
    if not wav_path.exists():
        return AudioLoadResult(audio=None, sr=None, backend=None, error=f"audio_missing:{wav_path}", debug={})

    # Try soundfile first (fast, stable for WAV).
    try:
        import soundfile as sf  # type: ignore

        data, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        audio = _to_float32_mono(np.asarray(data, dtype=np.float32))
        debug: dict[str, Any] = {"path": str(wav_path), "backend": "soundfile", "orig_sr": int(sr)}
        if sample_rate is not None and int(sample_rate) != int(sr):
            try:
                import librosa  # type: ignore

                audio = librosa.resample(audio, orig_sr=int(sr), target_sr=int(sample_rate))
                sr = int(sample_rate)
                debug["resampled"] = True
            except Exception as exc:  # noqa: BLE001
                return AudioLoadResult(
                    audio=None,
                    sr=None,
                    backend="soundfile",
                    error=f"resample_failed:{exc}",
                    debug=debug,
                )
        return AudioLoadResult(audio=audio.astype(np.float32, copy=False), sr=int(sr), backend="soundfile", error=None, debug=debug)
    except Exception:
        pass

    # Fallback to librosa loader.
    try:
        import librosa  # type: ignore

        y, sr = librosa.load(str(wav_path), sr=(int(sample_rate) if sample_rate else None), mono=True)
        audio = np.asarray(y, dtype=np.float32)
        return AudioLoadResult(
            audio=audio,
            sr=int(sr),
            backend="librosa",
            error=None,
            debug={"path": str(wav_path), "backend": "librosa", "resampled": sample_rate is not None},
        )
    except Exception as exc:  # noqa: BLE001
        return AudioLoadResult(audio=None, sr=None, backend="librosa", error=f"audio_load_failed:{exc}", debug={"path": str(wav_path)})

