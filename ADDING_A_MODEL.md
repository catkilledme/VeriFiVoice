# Adding a model adapter

## Run the current pretrained ONNX detector

Install the pinned runtime and validation dependencies with `python -m pip install -r requirements.txt -r requirements-test.txt`, then run `python scripts/download_model.py`. The downloader pins the AASIST repository revision and checks SHA-256 before writing `models/aasist.onnx`. Choose it with `VERIFVOICE_ADAPTER=aasist_onnx`; optionally set `VERIFVOICE_MODEL_PATH` and comma-separated `VERIFVOICE_ONNX_PROVIDERS`. CPU is the default provider. `main.py` and `analyze_audio()` keep their existing call shapes.

The current adapter accepts only the checksum-pinned AASIST artifact. For a different ONNX model, create a separate adapter and register it instead of pointing `AASISTOnnxDetector` at unrelated weights.

1. Put model-specific code in `audio_authenticity/adapters/<name>.py`; keep optional framework imports in that file. Add the framework dependency to a model-specific requirements/lock file and document native library needs.
2. Implement `AudioDetector`: `load_model(config)`, `predict(audio, metadata)`, `get_model_info()`, `health_check()`, and optionally `close()`. Raise clear loading errors. `audio` is validated mono float PCM in [-1,1]; `metadata.sample_rate` is unchanged. The adapter must perform its own documented resampling, windowing, normalization, and tensor shaping.
3. Map documented native output semantics explicitly. Put an uncalibrated normalized spoof score in `spoof_score`; set `calibrated_probability` only when calibration has actually been evaluated. A bona-fide score must be inverted only when the model documentation supports that mapping. Do not infer class-index meanings.
4. Register a factory during application startup with `register_adapter("team_model", TeamModelAdapter)` and configure `DetectorConfig(adapter_name="team_model")` / `VERIFVOICE_ADAPTER=team_model`. Threshold remains service config, outside the adapter.
5. Run `python -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp`, then add the common adapter compatibility cases for the new implementation. Keep test doubles in tests and label them clearly.
6. Call `analyze_audio(path)` from FastAPI exactly as for the default adapter; return `result.to_dict()`. The process default is cached; injected services can be closed with `AudioAnalysisService.close()` on application shutdown. Treat unavailable and inference errors as operational states, not a low spoof score. Risk fusion and payment action remain in other services.
7. For IBM LinuxONE, test on real s390x: Python/framework wheel availability, native dependencies, operators, model input parity, service startup, latency, and inference on consented labeled data. The tested ONNX Runtime/SciPy versions had no PyPI s390x wheel; locate compatible IBM/distribution builds or validate a source build. Record the OS, Python, architecture, package versions, and results.

## Isolated test-double example (not production evidence)

`tests/test_audio_authenticity.py::TestDoubleDetector` implements the same interface and emits a fixed score solely to verify service substitution. Never register it in a production deployment or describe its output as model evidence.

## Troubleshooting

- `Unknown adapter`: import/register the module at startup before creating the service.
- Loading error: check configured model path, artifact integrity/license, optional dependencies, and device setting; preserve the actionable adapter error.
- Import/native failure on s390x: verify architecture-specific wheels/build instructions rather than assuming x86 compatibility.
- Uncertain with no score: inspect status/reasons; invalid/quiet audio and absent model do not mean genuine speech.
