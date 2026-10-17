"""Adapter registry and factory, independent from the detection service."""

from collections.abc import Callable

from ..config import DetectorConfig
from .base import AudioDetector
from .onnx_detector import AASISTOnnxDetector
from .unavailable import UnavailableDetector

_REGISTRY: dict[str, Callable[[], AudioDetector]] = {
    "unavailable": UnavailableDetector,
    "aasist_onnx": AASISTOnnxDetector,
}


def register_adapter(name: str, factory: Callable[[], AudioDetector]) -> None:
    """Register an adapter factory under a stable configuration name."""
    if not name or name in _REGISTRY:
        raise ValueError(f"Adapter name is empty or already registered: {name!r}")
    _REGISTRY[name] = factory


def create_adapter(config: DetectorConfig) -> AudioDetector:
    """Instantiate configured adapter, failing early for unknown names."""
    try:
        return _REGISTRY[config.adapter_name]()
    except KeyError as exc:
        raise ValueError(f"Unknown adapter {config.adapter_name!r}; registered: {', '.join(sorted(_REGISTRY))}") from exc

