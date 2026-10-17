"""Generic WAV validation; model-specific resampling/normalization belongs in adapters."""

from array import array
import math
import os
import wave

from .config import DetectorConfig
from .exceptions import AudioInputError
from .schemas import AudioMetadata


def load_audio(path: str | os.PathLike[str], config: DetectorConfig) -> tuple[array, AudioMetadata]:
    """Read PCM WAV, downmix channels by averaging, and return mono float samples."""
    file_path = os.fspath(path)
    if not os.path.isfile(file_path):
        raise AudioInputError("Audio file does not exist or is not a regular file")
    size = os.path.getsize(file_path)
    if size == 0:
        raise AudioInputError("Audio file is empty")
    if size > config.max_file_bytes:
        raise AudioInputError("Audio file exceeds the configured byte limit")
    try:
        with wave.open(file_path, "rb") as source:
            channels, width, rate, frames = source.getnchannels(), source.getsampwidth(), source.getframerate(), source.getnframes()
            if source.getcomptype() != "NONE" or width not in (1, 2, 3, 4):
                raise AudioInputError("Only uncompressed PCM WAV with 8, 16, 24, or 32-bit samples is supported")
            if channels < 1 or rate <= 0 or frames <= 0:
                raise AudioInputError("WAV has invalid channel, sample rate, or frame metadata")
            duration = frames / rate
            if duration < config.minimum_duration_seconds:
                raise AudioInputError("Audio is shorter than the configured minimum duration")
            if duration > config.max_duration_seconds:
                raise AudioInputError("Audio exceeds the configured maximum duration")
            raw = source.readframes(frames)
    except AudioInputError:
        raise
    except (wave.Error, EOFError, OSError) as exc:
        raise AudioInputError(f"Could not decode a valid WAV file: {exc}") from exc
    if len(raw) != frames * channels * width:
        raise AudioInputError("WAV data is truncated or malformed")
    samples = _decode_pcm(raw, width)
    mono = array("f", (sum(samples[i:i + channels]) / channels for i in range(0, len(samples), channels)))
    if not mono or not all(math.isfinite(x) for x in mono):
        raise AudioInputError("Audio contains no usable finite samples")
    rms = math.sqrt(sum(float(x) ** 2 for x in mono) / len(mono))
    peak = max(abs(float(x)) for x in mono)
    return mono, AudioMetadata(rate, channels, frames, duration, rms, peak)


def _decode_pcm(raw: bytes, width: int) -> array:
    """Decode little-endian PCM to [-1, 1]; handle unsigned 8-bit PCM explicitly."""
    result = array("f")
    if width == 1:
        result.extend((sample - 128) / 128.0 for sample in raw)
    elif width == 2:
        values = array("h"); values.frombytes(raw)
        if os.sys.byteorder != "little": values.byteswap()
        result.extend(sample / 32768.0 for sample in values)
    elif width == 4:
        values = array("i"); values.frombytes(raw)
        if os.sys.byteorder != "little": values.byteswap()
        result.extend(sample / 2147483648.0 for sample in values)
    else:
        for index in range(0, len(raw), 3):
            value = int.from_bytes(raw[index:index + 3], "little", signed=False)
            if value & 0x800000: value -= 1 << 24
            result.append(value / 8388608.0)
    return result
