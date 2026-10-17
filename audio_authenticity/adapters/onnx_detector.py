"""ONNX Runtime adapter for the published SpeechAntiSpoofingBenchmarks AASIST model."""

from __future__ import annotations

from array import array
from hashlib import sha256
import math
import os

from ..config import DetectorConfig
from ..exceptions import InferenceError, ModelUnavailableError
from ..model_preprocessing import TARGET_SAMPLE_RATE, prepare_aasist_input
from ..schemas import AudioMetadata, ModelInfo, ModelPrediction
from .base import AudioDetector


MODEL_NAME = "AASIST (ASVspoof 2019 LA pretrained)"
MODEL_REVISION = "16774d458d86d2a021ae31646c1bf66a5331b53e"
MODEL_SHA256 = "130e536266b7c537f9a13029e1612a9f392fd1cc827783683b6d1c062a3db5e1"


class AASISTOnnxDetector(AudioDetector):
    """CPU-first AASIST inference; score is softmax-normalized spoof evidence, not calibrated."""

    def __init__(self) -> None:
        self._session = None
        self._input_name: str | None = None
        self._output_name: str | None = None
        self._model_path: str | None = None

    def load_model(self, config: DetectorConfig) -> None:
        """Load and validate the configured ONNX graph and available providers."""
        path = os.path.abspath(config.model_path)
        if not os.path.isfile(path):
            raise ModelUnavailableError(f"AASIST ONNX model not found at {path}; run python scripts/download_model.py")
        digest = sha256()
        with open(path, "rb") as model_file:
            for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != MODEL_SHA256:
            raise ModelUnavailableError(f"AASIST model checksum mismatch at {path}; run python scripts/download_model.py")
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ModelUnavailableError("ONNX Runtime is missing; install requirements.txt") from exc
        available = set(ort.get_available_providers())
        missing = [provider for provider in config.execution_providers if provider not in available]
        if missing:
            raise ModelUnavailableError(f"Requested ONNX provider(s) are unavailable: {', '.join(missing)}; available: {', '.join(sorted(available))}")
        try:
            session = ort.InferenceSession(path, providers=list(config.execution_providers))
        except Exception as exc:
            raise ModelUnavailableError(f"Could not load AASIST ONNX model {path}: {exc}") from exc
        inputs, outputs = session.get_inputs(), session.get_outputs()
        if len(inputs) != 1 or inputs[0].name != "wav" or inputs[0].type != "tensor(float)" or not _has_fixed_dimension(inputs[0].shape, 1, 64_600):
            raise ModelUnavailableError("Unexpected AASIST ONNX input; expected float32 `wav` with shape [batch, 64600]")
        if len(outputs) != 1 or outputs[0].name != "logits" or outputs[0].type != "tensor(float)" or not _has_fixed_dimension(outputs[0].shape, 1, 2):
            raise ModelUnavailableError("Unexpected AASIST ONNX output; expected float32 `logits` with shape [batch, 2]")
        self._session = session
        self._input_name = inputs[0].name
        self._output_name = outputs[0].name
        self._model_path = path

    def predict(self, audio: array, metadata: AudioMetadata) -> ModelPrediction:
        """Run AASIST and map class 0 (spoof) logits to a normalized score."""
        if not self.health_check():
            raise InferenceError("AASIST session is not ready")
        try:
            waveform = prepare_aasist_input(audio, metadata)
            logits = self._session.run([self._output_name], {self._input_name: waveform})[0]
            if getattr(logits, "shape", None) != (1, 2):
                raise ValueError(f"Model returned an unexpected logits shape: {getattr(logits, 'shape', None)}")
            scores = logits[0].tolist()
            if not all(math.isfinite(float(value)) for value in scores):
                raise ValueError("Model returned non-finite logits")
            spoof_score = _spoof_score_from_logits(scores)
            return ModelPrediction(spoof_score=spoof_score)
        except Exception as exc:
            raise InferenceError(f"AASIST ONNX inference failed: {exc}") from exc

    def get_model_info(self) -> ModelInfo:
        """Return model provenance, audio requirements, and current readiness."""
        return ModelInfo(
            name=MODEL_NAME,
            version=MODEL_REVISION,
            supported_sample_rates=(TARGET_SAMPLE_RATE,),
            input_requirements="mono float32 PCM [-1,1], 16 kHz; deterministic 64,600-sample window/repeat padding",
            supports_spoof_score=True,
            supports_replay_detection=False,
            supports_probability_calibration=False,
            ready=self.health_check(),
        )

    def health_check(self) -> bool:
        """Report whether the ONNX session loaded and remains available."""
        return self._session is not None

    def close(self) -> None:
        """Release the ONNX session reference for process shutdown or reload."""
        self._session = None
        self._input_name = self._output_name = self._model_path = None


def _has_fixed_dimension(shape, rank: int, last_dimension: int) -> bool:
    """Check rank and final fixed model dimension while allowing dynamic batch."""
    return len(shape) == rank + 1 and shape[-1] == last_dimension and shape[0] in ("batch", "N", None, 1)


def _spoof_score_from_logits(logits: list[float]) -> float:
    """Softmax class logits; training maps spoof=0 and bona fide=1."""
    maximum = max(float(value) for value in logits)
    exponents = [math.exp(float(value) - maximum) for value in logits]
    return exponents[0] / sum(exponents)
