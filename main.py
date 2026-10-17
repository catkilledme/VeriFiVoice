"""VerifVoice audio authenticity CLI and dependency/method inventory.

Dependencies (direct, exact pins in requirements.txt):
- pytest 8.3.5 (direct): test runner imported in tests/test_audio_authenticity.py;
  uses pytest.fixture, pytest.mark.parametrize, and pytest.raises.
- colorama 0.4.6, iniconfig 2.3.1, packaging 26.3, pluggy 1.7.0
  (pytest transitive dependencies): used by pytest's terminal coloring, config,
  version/requirement handling, and plugin hooks; not imported by production code.
- Production package uses only Python standard library (dataclasses, array, wave,
  math, os, time, typing, abc, collections). No transitive runtime packages.
- No ONNX Runtime, NumPy, SciPy, soundfile, or model framework is used because
  no verified compatible artifact/runtime is present. WAV decoding uses wave.open,
  getparams/getnchannels/getsampwidth/getframerate/getnframes/readframes.
- Public module methods are documented in audio_authenticity and README.md.
System dependencies: none for the WAV-only baseline. A future adapter must
document its framework, native libraries, and architecture-specific support.

Files: config.py parses settings; preprocessing.py validates/decodes WAV;
adapters/base.py defines the contract; adapters/unavailable.py refuses fake
inference; adapters/registry.py selects implementations; schemas.py defines the
versioned output; detector.py orchestrates; tests/ holds compatibility tests.
"""

import argparse
import json
import platform
import sys

from audio_authenticity import AudioAnalysisService, DetectorConfig


def main() -> int:
    """Run the diagnostic or analyze a WAV and print the JSON contract."""
    parser = argparse.ArgumentParser(description="VeriFiVoice audio authenticity module")
    parser.add_argument("audio", nargs="?", help="Path to PCM WAV audio")
    parser.add_argument("--diagnostics", action="store_true", help="Report environment and model readiness")
    args = parser.parse_args()
    config = DetectorConfig.from_env()
    service = AudioAnalysisService(config)
    if args.diagnostics:
        output = {"os": platform.platform(), "python": sys.version, "machine": platform.machine(), "adapter": config.adapter_name, "model_ready": service.detector.health_check(), "load_error": service._load_error}
    elif args.audio:
        output = service.analyze(args.audio).to_dict()
    else:
        parser.error("provide an audio path or --diagnostics")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
