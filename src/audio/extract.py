from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AudioExtractionResult:
    ok: bool
    wav_path: Path | None
    sample_rate: int
    backend: str | None
    error: str | None = None


def find_ffmpeg_executable() -> str | None:
    ffmpeg_exe = shutil.which("ffmpeg")
    if ffmpeg_exe:
        return ffmpeg_exe
    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def extract_audio_from_video(
    video_path: str | Path,
    out_wav_path: str | Path,
    sample_rate: int,
) -> AudioExtractionResult:
    video = Path(video_path)
    out_wav = Path(out_wav_path)
    ffmpeg_exe = find_ffmpeg_executable()
    if ffmpeg_exe is None:
        return AudioExtractionResult(
            ok=False,
            wav_path=None,
            sample_rate=int(sample_rate),
            backend=None,
            error="ffmpeg_not_found",
        )
    if not video.exists():
        return AudioExtractionResult(
            ok=False,
            wav_path=None,
            sample_rate=int(sample_rate),
            backend="ffmpeg",
            error=f"video_missing:{video}",
        )

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_exe,
        "-y",
        "-i",
        str(video),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(int(sample_rate)),
        "-c:a",
        "pcm_s16le",
        str(out_wav),
    ]
    try:
        proc = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or b"").decode("utf-8", errors="ignore").strip()
        if out_wav.exists():
            out_wav.unlink(missing_ok=True)
        msg = stderr or str(exc)
        return AudioExtractionResult(
            ok=False,
            wav_path=None,
            sample_rate=int(sample_rate),
            backend="ffmpeg",
            error=f"ffmpeg_extract_failed:{msg}",
        )
    except Exception as exc:  # noqa: BLE001
        if out_wav.exists():
            out_wav.unlink(missing_ok=True)
        return AudioExtractionResult(
            ok=False,
            wav_path=None,
            sample_rate=int(sample_rate),
            backend="ffmpeg",
            error=f"ffmpeg_extract_failed:{exc}",
        )

    if not out_wav.exists() or out_wav.stat().st_size <= 0:
        return AudioExtractionResult(
            ok=False,
            wav_path=None,
            sample_rate=int(sample_rate),
            backend="ffmpeg",
            error="audio_output_missing",
        )

    return AudioExtractionResult(
        ok=True,
        wav_path=out_wav,
        sample_rate=int(sample_rate),
        backend="ffmpeg",
        error=None,
    )

