"""Stable JSON-friendly shared contracts; scores are not calibrated probabilities."""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class AudioMetadata:
    sample_rate: int
    channels: int
    frames: int
    duration_seconds: float
    rms: float
    peak: float


@dataclass(frozen=True)
class ModelInfo:
    name: str
    version: str | None
    supported_sample_rates: tuple[int, ...] = ()
    input_requirements: str = "validated mono float waveform; adapter owns model-specific transforms"
    supports_spoof_score: bool = False
    supports_replay_detection: bool = False
    supports_probability_calibration: bool = False
    ready: bool = False


@dataclass(frozen=True)
class ModelPrediction:
    """Adapter output: spoof_score is model evidence, not a calibrated probability."""

    spoof_score: float | None = None
    calibrated_probability: float | None = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class AudioAnalysisResult:
    contract_version: str
    status: str
    authenticity_score: float | None
    score_kind: str | None
    calibrated_probability: float | None
    classification: str
    confidence: float | None
    quality_score: float | None
    model_name: str
    model_version: str | None
    sample_rate: int | None
    duration_seconds: float | None
    inference_ms: float | None
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable mapping (tuples become JSON arrays)."""
        return asdict(self)


def validate_prediction(prediction: ModelPrediction) -> None:
    """Reject non-finite scores or values outside their documented [0, 1] bounds."""
    import math

    for name, value in (("spoof_score", prediction.spoof_score), ("calibrated_probability", prediction.calibrated_probability)):
        if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError(f"{name} must be a finite number in [0, 1] or None")
