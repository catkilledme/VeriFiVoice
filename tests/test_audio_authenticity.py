"""Preprocessing, service, adapter-contract, and modularity tests."""

from array import array
import math
import wave

import pytest

from audio_authenticity.adapters.base import AudioDetector
from audio_authenticity.adapters.registry import register_adapter
from audio_authenticity.config import DetectorConfig
from audio_authenticity.detector import AudioAnalysisService, analyze_audio
from audio_authenticity.exceptions import ModelUnavailableError
from audio_authenticity.schemas import AudioMetadata, ModelInfo, ModelPrediction


def write_wav(path, *, channels=1, seconds=1.0, silence=False, rate=16000):
    frames = int(seconds * rate)
    with wave.open(str(path), "wb") as out:
        out.setnchannels(channels); out.setsampwidth(2); out.setframerate(rate)
        values = []
        for frame in range(frames):
            value = 0 if silence else int(8000 * math.sin(2 * math.pi * 440 * frame / rate))
            values.extend([value] * channels)
        out.writeframes(array("h", values).tobytes())


class TestDoubleDetector(AudioDetector):
    """TEST DOUBLE ONLY: deterministic score used to verify interface wiring."""
    __test__ = False

    def __init__(self, score=0.2, failure=None):
        self.score, self.failure, self.ready = score, failure, False

    def load_model(self, config):
        self.ready = True

    def predict(self, audio, metadata):
        if self.failure: raise RuntimeError(self.failure)
        return ModelPrediction(spoof_score=self.score)

    def get_model_info(self):
        return ModelInfo("test-double", "1", supports_spoof_score=True, ready=self.ready)

    def health_check(self):
        return self.ready


@pytest.fixture
def config():
    return DetectorConfig(max_duration_seconds=2.0, minimum_duration_seconds=0.1)


def test_valid_mono_wav_and_public_schema(tmp_path, config):
    path = tmp_path / "mono.wav"; write_wav(path)
    result = analyze_audio(str(path), TestDoubleDetector(), config)
    assert result.status == "success" and result.classification == "likely_genuine"
    assert result.authenticity_score == 0.2 and result.confidence is None
    assert result.calibrated_probability is None
    assert result.to_dict()["contract_version"] == "1.0"


def test_valid_stereo_downmix(tmp_path, config):
    path = tmp_path / "stereo.wav"; write_wav(path, channels=2)
    result = AudioAnalysisService(config, TestDoubleDetector()).analyze(str(path))
    assert result.status == "success" and result.sample_rate == 16000


@pytest.mark.parametrize("kind", ["empty", "malformed", "missing", "long", "unsupported", "silent"])
def test_invalid_and_quality_audio_is_uncertain(tmp_path, config, kind):
    path = tmp_path / "input.wav"
    if kind == "empty": path.write_bytes(b"")
    elif kind == "malformed": path.write_bytes(b"RIFF malformed data")
    elif kind == "missing": pass
    elif kind == "long": write_wav(path, seconds=3)
    elif kind == "unsupported": path = tmp_path / "input.mp3"; path.write_bytes(b"not an mp3")
    elif kind == "silent": write_wav(path, silence=True)
    result = analyze_audio(str(path), TestDoubleDetector(), config)
    assert result.classification == "uncertain"
    assert result.status == ("low_quality" if kind == "silent" else "low_quality")
    assert result.authenticity_score is None


def test_model_loading_failure_is_distinct_from_low_score(tmp_path, config):
    path = tmp_path / "good.wav"; write_wav(path)
    result = AudioAnalysisService(config).analyze(str(path))
    assert result.status == "model_unavailable" and result.authenticity_score is None


def test_adapter_load_error_is_actionable(tmp_path, config):
    class BrokenLoad(TestDoubleDetector):
        __test__ = False
        def load_model(self, config): raise ModelUnavailableError("weights missing")
    path = tmp_path / "good.wav"; write_wav(path)
    result = analyze_audio(str(path), BrokenLoad(), config)
    assert result.status == "model_unavailable"
    assert "weights missing" in result.warnings[0]


def test_analysis_does_not_create_audio_artifacts(tmp_path, config):
    path = tmp_path / "consented.wav"; write_wav(path)
    before = set(tmp_path.iterdir())
    analyze_audio(str(path), TestDoubleDetector(), config)
    assert set(tmp_path.iterdir()) == before


def test_inference_failure_does_not_escape(tmp_path, config):
    path = tmp_path / "good.wav"; write_wav(path)
    result = analyze_audio(str(path), TestDoubleDetector(failure="simulated failure"), config)
    assert result.status == "inference_error" and result.classification == "uncertain"


def test_score_validation_rejects_invalid_values(tmp_path, config):
    path = tmp_path / "good.wav"; write_wav(path)
    result = analyze_audio(str(path), TestDoubleDetector(score=float("nan")), config)
    assert result.status == "inference_error" and result.authenticity_score is None


def test_unknown_adapter_configuration_fails_early():
    with pytest.raises(ValueError, match="Unknown adapter"):
        from audio_authenticity.adapters.registry import create_adapter
        create_adapter(DetectorConfig(adapter_name="not-registered"))


def test_configuration_limits_are_validated():
    with pytest.raises(ValueError, match="threshold"):
        DetectorConfig(classification_threshold=1.0).validate()


def test_calibrated_probability_has_its_own_field(tmp_path, config):
    class CalibratedDouble(TestDoubleDetector):
        __test__ = False
        def predict(self, audio, metadata):
            return ModelPrediction(spoof_score=0.4, calibrated_probability=0.35)
    path = tmp_path / "good.wav"; write_wav(path)
    result = analyze_audio(str(path), CalibratedDouble(), config)
    assert result.authenticity_score == 0.4
    assert result.calibrated_probability == 0.35
    assert result.confidence is None


def test_same_integration_contract_with_registered_alternate_adapter(tmp_path):
    """Registry selection swaps implementation while service/public API stay unchanged."""
    register_adapter("test-double", TestDoubleDetector)
    cfg = DetectorConfig(adapter_name="test-double")
    path = tmp_path / "good.wav"; write_wav(path)
    service_result = AudioAnalysisService(cfg).analyze(str(path))
    public_result = analyze_audio(str(path), TestDoubleDetector(0.8), cfg)
    assert service_result.classification == "likely_genuine"
    assert public_result.classification == "potential_spoof"
    assert set(service_result.to_dict()) == set(public_result.to_dict())


def test_default_public_api_reuses_one_loaded_adapter(tmp_path, monkeypatch):
    from audio_authenticity import detector as service_module

    load_count = 0

    class CountingDouble(TestDoubleDetector):
        __test__ = False
        def load_model(self, config):
            nonlocal load_count
            load_count += 1
            super().load_model(config)

    service_module.close_default_service()
    register_adapter("lifecycle-double", CountingDouble)
    monkeypatch.setenv("VERIFVOICE_ADAPTER", "lifecycle-double")
    path = tmp_path / "good.wav"; write_wav(path)
    try:
        assert service_module.analyze_audio(str(path)).status == "success"
        assert service_module.analyze_audio(str(path)).status == "success"
        assert load_count == 1
    finally:
        service_module.close_default_service()


def test_interface_is_abstract_and_test_double_satisfies_contract():
    assert isinstance(TestDoubleDetector(), AudioDetector)
    assert all(hasattr(TestDoubleDetector, name) for name in ("load_model", "predict", "get_model_info", "health_check", "close"))


