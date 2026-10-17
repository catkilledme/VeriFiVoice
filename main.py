"""VerifVoice CLI and dependency/method inventory.

Direct runtime dependencies (exact pins in requirements.txt):
- onnxruntime 1.31.0, audio_authenticity/adapters/onnx_detector.py:
  InferenceSession, get_available_providers, get_inputs, get_outputs, run.
- numpy 2.5.4, audio_authenticity/model_preprocessing.py and onnx_detector.py:
  asarray, tile, ascontiguousarray, reshape, finite checks, result conversion.
- scipy 1.18.1, audio_authenticity/model_preprocessing.py:
  scipy.signal.resample_poly for model-required 16 kHz conversion.
- onnx 1.23.2, tests/test_onnx_detector.py only (requirements-test.txt):
  onnx.load and onnx.checker.check_model validate the downloaded graph.
- pytest 8.3.5, tests/: fixture, parametrized cases, skip conditions, assertions.

Pinned transitive dependencies: flatbuffers 25.12.19 and protobuf 7.36.2
(ONNX Runtime); packaging 26.3 (ONNX Runtime and pytest); ml-dtypes 0.6.0,
typing-extensions 4.16.0 (onnx); colorama 0.4.6, iniconfig 2.3.1,
pluggy 1.7.0 (pytest). None is imported directly by production modules.
The remaining implementation uses Python standard library modules including
dataclasses, array, wave, math, os, time, threading, urllib, and pathlib.

System dependency: Windows requires Microsoft Visual C++ 2019 runtime or newer
for the official ONNX Runtime wheel. LinuxONE requires compatible native builds.
WAV decode uses wave.open/getnchannels/getsampwidth/getframerate/getnframes/
readframes. Model acquisition is scripts/download_model.py (urllib, hashlib).

Files: config.py owns runtime settings; preprocessing.py validates WAV;
model_preprocessing.py prepares AASIST input; adapters/base.py is the model
contract; adapters/onnx_detector.py owns ONNX inference/output mapping;
adapters/unavailable.py is the honest fallback; adapters/registry.py selects the
adapter; schemas.py defines output; detector.py orchestrates and caches the
default service; tests/ validate the contract and real artifact.
"""

import argparse
from importlib.metadata import PackageNotFoundError, version
import json
import platform
import sys

from audio_authenticity import AudioAnalysisService, DetectorConfig


def _package_version(package: str) -> str | None:
    """Return an installed distribution version for diagnostics."""
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def main() -> int:
    """Run the diagnostic or analyze a WAV and print the JSON contract."""
    parser = argparse.ArgumentParser(description="VeriFiVoice audio authenticity module")
    parser.add_argument("audio", nargs="?", help="Path to PCM WAV audio")
    parser.add_argument("--diagnostics", action="store_true", help="Report environment and model readiness")
    args = parser.parse_args()
    config = DetectorConfig.from_env()
    service = AudioAnalysisService(config)
    if args.diagnostics:
        info = service.detector.get_model_info()
        output = {
            "os": platform.platform(),
            "python": sys.version,
            "machine": platform.machine(),
            "adapter": config.adapter_name,
            "model_path": config.model_path,
            "model_name": info.name,
            "model_version": info.version,
            "model_ready": service.detector.health_check(),
            "load_error": service._load_error,
            "dependencies": {name: _package_version(name) for name in ("onnxruntime", "numpy", "scipy")},
        }
    elif args.audio:
        output = service.analyze(args.audio).to_dict()
    else:
        parser.error("provide an audio path or --diagnostics")
    print(json.dumps(output, indent=2))
    service.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
