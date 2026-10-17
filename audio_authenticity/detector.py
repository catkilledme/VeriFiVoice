"""Model-independent orchestration and stable integration API."""

from __future__ import annotations

import atexit
from threading import Lock
import time

from .adapters.base import AudioDetector
from .adapters.registry import create_adapter
from .config import DetectorConfig
from .exceptions import AudioInputError, InferenceError, ModelUnavailableError
from .preprocessing import load_audio
from .schemas import AudioAnalysisResult, validate_prediction


class AudioAnalysisService:
    """Validate, preprocess, invoke an injected adapter, and format shared results."""

    def __init__(self, config: DetectorConfig | None = None, detector: AudioDetector | None = None) -> None:
        self.config = config or DetectorConfig.from_env()
        self.config.validate()
        self.detector = detector or create_adapter(self.config)
        self._load_error: str | None = None
        try:
            self.detector.load_model(self.config)
        except Exception as exc:
            self._load_error = str(exc)

    def analyze(self, audio_path: str) -> AudioAnalysisResult:
        """Analyze one WAV and return evidence/status without making a payment decision."""
        info = self.detector.get_model_info()
        try:
            audio, metadata = load_audio(audio_path, self.config)
        except AudioInputError as exc:
            return self._result("low_quality", None, "uncertain", info.name, info.version, None, None, ("invalid_or_low_quality_audio",), (str(exc),))
        quality = min(1.0, metadata.rms / max(self.config.minimum_rms * 10, 1e-12))
        if metadata.rms < self.config.minimum_rms:
            return self._result("low_quality", None, "uncertain", info.name, info.version, metadata.sample_rate, metadata.duration_seconds, ("silence_or_near_silence",), (), quality)
        if self._load_error or not self.detector.health_check():
            reason = "model_load_failed" if self._load_error else "model_unavailable"
            warning = self._load_error or "Configured adapter is not ready"
            audio_warnings = (warning, "possible_clipping") if metadata.peak >= 0.999 else (warning,)
            return self._result("model_unavailable", None, "uncertain", info.name, info.version, metadata.sample_rate, metadata.duration_seconds, (reason,), audio_warnings, quality)
        started = time.perf_counter()
        try:
            prediction = self.detector.predict(audio, metadata)
            validate_prediction(prediction)
        except Exception as exc:
            elapsed = (time.perf_counter() - started) * 1000
            return self._result("inference_error", None, "uncertain", info.name, info.version, metadata.sample_rate, metadata.duration_seconds, ("inference_failed",), (str(exc),), quality, elapsed)
        elapsed = (time.perf_counter() - started) * 1000
        score = prediction.spoof_score
        if score is None:
            status, label = "success", "uncertain"
        elif score >= self.config.classification_threshold:
            status, label = "success", "potential_spoof"
        else:
            status, label = "success", "likely_genuine"
        result_warnings = prediction.warnings + (("possible_clipping",) if metadata.peak >= 0.999 else ())
        return self._result(status, score, label, info.name, info.version, metadata.sample_rate, metadata.duration_seconds, (), result_warnings, quality, elapsed, "uncalibrated_spoof_score" if score is not None else None, prediction.calibrated_probability)

    def close(self) -> None:
        """Release adapter resources when an application service shuts down."""
        self.detector.close()

    def _result(self, status, score, label, name, version, rate, duration, reasons, warnings, quality=None, elapsed=None, kind=None, calibrated_probability=None):
        return AudioAnalysisResult("1.0", status, score, kind, calibrated_probability, label, None, quality, name, version, rate, duration, elapsed, tuple(reasons), tuple(warnings))


_default_service: AudioAnalysisService | None = None
_default_service_lock = Lock()


def analyze_audio(audio_path: str, detector: AudioDetector | None = None, config: DetectorConfig | None = None) -> AudioAnalysisResult:
    """Stable function for FastAPI and other clients; adapter injection is optional."""
    if detector is None and config is None:
        global _default_service
        if _default_service is None:
            with _default_service_lock:
                if _default_service is None:
                    _default_service = AudioAnalysisService()
        return _default_service.analyze(audio_path)
    return AudioAnalysisService(config, detector).analyze(audio_path)


def close_default_service() -> None:
    """Release the process-wide default detector, suitable for API shutdown hooks."""
    global _default_service
    with _default_service_lock:
        if _default_service is not None:
            _default_service.close()
            _default_service = None


atexit.register(close_default_service)

