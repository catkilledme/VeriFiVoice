"""AASIST-specific waveform preparation, separate from generic WAV decoding."""

from __future__ import annotations

import math

from .schemas import AudioMetadata

TARGET_SAMPLE_RATE = 16_000
TARGET_SAMPLES = 64_600


def prepare_aasist_input(audio, metadata: AudioMetadata):
    """Resample to 16 kHz and deterministically tile/crop to AASIST's 64,600 samples.

    AASIST's published inference wrapper expects float32 mono audio at 16 kHz and
    uses `data_utils.pad`: longer utterances take the first window; shorter ones
    repeat periodically. The model expects the original PCM scaling, not peak
    normalization. scipy's polyphase resampler is used only when rates differ.
    """
    import numpy as np

    waveform = np.asarray(audio, dtype=np.float32)
    if waveform.ndim != 1 or waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError("AASIST requires a non-empty finite mono waveform")
    if metadata.sample_rate != TARGET_SAMPLE_RATE:
        from scipy.signal import resample_poly

        divisor = math.gcd(metadata.sample_rate, TARGET_SAMPLE_RATE)
        waveform = resample_poly(waveform, TARGET_SAMPLE_RATE // divisor, metadata.sample_rate // divisor).astype(np.float32, copy=False)
    if waveform.size >= TARGET_SAMPLES:
        waveform = waveform[:TARGET_SAMPLES]
    else:
        repeats = (TARGET_SAMPLES + waveform.size - 1) // waveform.size
        waveform = np.tile(waveform, repeats)[:TARGET_SAMPLES]
    batch = np.ascontiguousarray(waveform.reshape(1, TARGET_SAMPLES), dtype=np.float32)
    if not np.isfinite(batch).all():
        raise ValueError("AASIST preprocessing produced non-finite samples")
    return batch
