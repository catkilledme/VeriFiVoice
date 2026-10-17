"""Honest production default: no vetted pretrained artifact was available."""

from array import array

from ..config import DetectorConfig
from ..exceptions import ModelUnavailableError
from ..schemas import AudioMetadata, ModelInfo, ModelPrediction
from .base import AudioDetector


class UnavailableDetector(AudioDetector):
    """Adapter that clearly reports missing inference weights and never fabricates a score."""

    def load_model(self, config: DetectorConfig) -> None:
        raise ModelUnavailableError("No verified pretrained anti-spoofing model is configured; set VERIFVOICE_ADAPTER after adding a vetted adapter and weights")

    def predict(self, audio: array, metadata: AudioMetadata) -> ModelPrediction:
        raise ModelUnavailableError("No pretrained model is available for inference")

    def get_model_info(self) -> ModelInfo:
        return ModelInfo("unavailable", None, ready=False)

    def health_check(self) -> bool:
        return False
