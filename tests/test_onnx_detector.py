"""AASIST adapter validation; synthetic waveforms test plumbing, not accuracy."""

from array import array
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import wave

import numpy as np
import onnx
import pytest

from audio_authenticity.adapters.onnx_detector import (
    AASISTOnnxDetector,
    _spoof_score_from_logits,
)
from audio_authenticity.adapters.registry import create_adapter
from audio_authenticity.config import DetectorConfig
from audio_authenticity.detector import AudioAnalysisService
from audio_authenticity.exceptions import ModelUnavailableError
from audio_authenticity.model_preprocessing import prepare_aasist_input
from audio_authenticity.schemas import AudioMetadata


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "aasist.onnx"


def _write_wave(path: Path, *, rate: int = 16_000, seconds: float = 1.0, channels: int = 1) -> None:
    """Write a deterministic sine wave to exercise file and model plumbing only."""
    with wave.open(str(path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(rate)
        samples = []
        for frame in range(int(rate * seconds)):
            value = int(8_000 * math.sin(2 * math.pi * 440 * frame / rate))
            samples.extend([value] * channels)
        output.writeframes(array("h", samples).tobytes())


def _config(path: Path = MODEL_PATH, **kwargs) -> DetectorConfig:
    return DetectorConfig(adapter_name="aasist_onnx", model_path=str(path), **kwargs)


def test_model_registry_selects_onnx_adapter():
    assert isinstance(create_adapter(_config()), AASISTOnnxDetector)


def test_missing_model_has_actionable_error(tmp_path):
    detector = AASISTOnnxDetector()
    with pytest.raises(ModelUnavailableError, match="download_model.py"):
        detector.load_model(_config(tmp_path / "missing.onnx"))


def test_malformed_onnx_graph_has_clear_error(tmp_path):
    invalid = tmp_path / "broken.onnx"
    invalid.write_bytes(b"not an ONNX protobuf")
    with pytest.raises(ModelUnavailableError, match="checksum mismatch"):
        AASISTOnnxDetector().load_model(_config(invalid))


def test_onnx_graph_passes_checker_and_declares_expected_signature():
    if not MODEL_PATH.exists():
        pytest.skip("Run python scripts/download_model.py to enable real-artifact checks")
    graph = onnx.load(str(MODEL_PATH))
    onnx.checker.check_model(graph)
    assert [(value.name, value.type.tensor_type.elem_type) for value in graph.graph.input] == [("wav", onnx.TensorProto.FLOAT)]
    assert [(value.name, value.type.tensor_type.elem_type) for value in graph.graph.output] == [("logits", onnx.TensorProto.FLOAT)]


def test_model_specific_resampling_and_window_shape():
    source_rate = 8_000
    audio = array("f", (0.1 * math.sin(2 * math.pi * 220 * i / source_rate) for i in range(source_rate)))
    metadata = AudioMetadata(source_rate, 1, source_rate, 1.0, 0.07, 0.1)
    waveform = prepare_aasist_input(audio, metadata)
    assert waveform.shape == (1, 64_600)
    assert waveform.dtype == np.float32 and waveform.flags.c_contiguous
    assert np.isfinite(waveform).all()


def test_logit_mapping_uses_documented_spoof_index():
    assert _spoof_score_from_logits([5.0, -5.0]) > 0.99
    assert _spoof_score_from_logits([-5.0, 5.0]) < 0.01


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Pinned AASIST ONNX artifact is not downloaded")
def test_real_onnx_inference_produces_finite_standard_result(tmp_path):
    path = tmp_path / "generated.wav"
    _write_wave(path)
    detector = AASISTOnnxDetector()
    service = AudioAnalysisService(_config(), detector)
    try:
        result = service.analyze(str(path))
        assert detector.health_check()
        assert result.status == "success"
        assert result.model_name.startswith("AASIST")
        assert 0 <= result.authenticity_score <= 1
        assert result.calibrated_probability is None and result.confidence is None
        assert result.inference_ms is not None and result.inference_ms >= 0
        json.dumps(result.to_dict())
    finally:
        service.close()
    assert not detector.health_check()


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Pinned AASIST ONNX artifact is not downloaded")
def test_main_cli_runs_the_real_adapter_without_api_changes(tmp_path):
    audio_path = tmp_path / "cli.wav"
    _write_wave(audio_path, rate=8_000)
    environment = os.environ.copy()
    environment["VERIFVOICE_ADAPTER"] = "aasist_onnx"
    environment["VERIFVOICE_MODEL_PATH"] = str(MODEL_PATH)
    completed = subprocess.run(
        [sys.executable, str(ROOT / "main.py"), str(audio_path)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    result = json.loads(completed.stdout)
    assert result["status"] == "success"
    assert result["model_name"].startswith("AASIST")
    assert result["sample_rate"] == 8_000
    assert 0 <= result["authenticity_score"] <= 1


def test_unavailable_provider_is_reported_without_provider_guessing():
    detector = AASISTOnnxDetector()
    with pytest.raises(ModelUnavailableError, match="provider.*unavailable"):
        detector.load_model(_config(execution_providers=("NoSuchProvider",)))

