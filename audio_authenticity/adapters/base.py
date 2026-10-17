"""Small stable protocol for interchangeable audio detector adapters."""

from abc import ABC, abstractmethod
from array import array

from ..config import DetectorConfig
from ..schemas import AudioMetadata, ModelInfo, ModelPrediction


class AudioDetector(ABC):
    """Adapter boundary. Implementations own model-specific transforms and inference."""

    @abstractmethod
    def load_model(self, config: DetectorConfig) -> None:
        """Initialize model resources or raise a clear model-loading error."""

    @abstractmethod
    def predict(self, audio: array, metadata: AudioMetadata) -> ModelPrediction:
        """Predict from validated mono audio and normalize native outputs."""

    @abstractmethod
    def get_model_info(self) -> ModelInfo:
        """Describe model version, inputs, and supported evidence capabilities."""

    @abstractmethod
    def health_check(self) -> bool:
        """Return whether inference can run now."""

    def close(self) -> None:
        """Release resources; override when the framework needs explicit cleanup."""
