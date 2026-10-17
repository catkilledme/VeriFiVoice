# Adding a model adapter

1. Put model-specific code in `audio_authenticity/adapters/<name>.py`; keep optional framework imports in that file. Add the framework dependency to a model-specific requirements/lock file and document native library needs.
2. Implement `AudioDetector`: `load_model(config)`, `predict(audio, metadata)`, `get_model_info()`, `health_check()`, and optionally `close()`. Raise clear loading errors. `audio` is validated mono float PCM in [-1,1]; `metadata.sample_rate` is unchanged. The adapter must perform its own documented resampling, windowing, normalization, and tensor shaping.
3. Map documented native output semantics explicitly. Put an uncalibrated normalized spoof score in `spoof_score`; set `calibrated_probability` only when calibration has actually been evaluated. A bona-fide score must be inverted only when the model documentation supports that mapping. Do not infer class-index meanings.
4. Register a factory during application startup with `register_adapter("team_model", TeamModelAdapter)` and configure `DetectorConfig(adapter_name="team_model")` / `VERIFVOICE_ADAPTER=team_model`. Threshold remains service config, outside the adapter.
5. Run `pytest -q`, then add the common adapter compatibility cases for the new implementation. Keep test doubles in tests and label them clearly.
6. Call `analyze_audio(path)` from FastAPI exactly as for the default adapter; return `result.to_dict()`. Treat unavailable and inference errors as operational states, not a low spoof score. Risk fusion and payment action remain in other services.
7. For IBM LinuxONE, test on real s390x: Python/framework wheel availability, native dependencies, operators, model input parity, service startup, latency, and inference on consented labeled data. Record the OS, Python, architecture, package versions, and results.

## Isolated test-double example (not production evidence)

`tests/test_audio_authenticity.py::TestDoubleDetector` implements the same interface and emits a fixed score solely to verify service substitution. Never register it in a production deployment or describe its output as model evidence.

## Troubleshooting

- `Unknown adapter`: import/register the module at startup before creating the service.
- Loading error: check configured model path, artifact integrity/license, optional dependencies, and device setting; preserve the actionable adapter error.
- Import/native failure on s390x: verify architecture-specific wheels/build instructions rather than assuming x86 compatibility.
- Uncertain with no score: inspect status/reasons; invalid/quiet audio and absent model do not mean genuine speech.
