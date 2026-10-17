"""Configuration for audio validation and adapter selection."""

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class DetectorConfig:
    """Runtime settings, overridable with VERIFVOICE_* environment variables."""

    adapter_name: str = "unavailable"
    max_file_bytes: int = 25 * 1024 * 1024
    max_duration_seconds: float = 60.0
    minimum_duration_seconds: float = 0.25
    minimum_rms: float = 1e-4
    classification_threshold: float = 0.5

    @classmethod
    def from_env(cls) -> "DetectorConfig":
        """Build settings from environment variables and validate them."""
        config = cls(
            adapter_name=os.getenv("VERIFVOICE_ADAPTER", "unavailable"),
            max_file_bytes=int(os.getenv("VERIFVOICE_MAX_FILE_BYTES", str(cls.max_file_bytes))),
            max_duration_seconds=float(os.getenv("VERIFVOICE_MAX_DURATION", str(cls.max_duration_seconds))),
            minimum_duration_seconds=float(os.getenv("VERIFVOICE_MIN_DURATION", str(cls.minimum_duration_seconds))),
            minimum_rms=float(os.getenv("VERIFVOICE_MIN_RMS", str(cls.minimum_rms))),
            classification_threshold=float(os.getenv("VERIFVOICE_THRESHOLD", str(cls.classification_threshold))),
        )
        config.validate()
        return config

    def validate(self) -> None:
        """Raise ValueError for inconsistent or unsafe limits."""
        if self.max_file_bytes <= 0 or self.max_duration_seconds <= 0:
            raise ValueError("File size and maximum duration limits must be positive")
        if not 0 <= self.minimum_duration_seconds <= self.max_duration_seconds:
            raise ValueError("Minimum duration must be between zero and maximum duration")
        if not 0 < self.classification_threshold < 1:
            raise ValueError("Classification threshold must be strictly between 0 and 1")
        if self.minimum_rms < 0:
            raise ValueError("Minimum RMS must be non-negative")
