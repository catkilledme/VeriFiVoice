# Implementation and verification notes

## Environment and starting point

The earlier project contained the modular WAV service, test double, CLI, and documentation, already committed to `catkilledme/VeriFiVoice` on `main`. Development host: Windows 11 x86-64. Python launcher was not initially on PATH, so the workspace-local uv-managed CPython 3.12.15 environment was reused. A first pytest run failed because the managed environment denied access to its default `%TEMP%` and `.pytest_cache` directories; using `--basetemp .pytest-tmp -p no:cacheprovider` resolved that without changing the tests.

## Selected model and provenance

Selected AASIST (the official architecture, ASVspoof 2019 LA pretrained checkpoint) from [SpeechAntiSpoofingBenchmarks/AASIST](https://huggingface.co/SpeechAntiSpoofingBenchmarks/AASIST), pinned at revision `16774d458d86d2a021ae31646c1bf66a5331b53e`. The model card labels the model MIT and describes a raw waveform model whose bona-fide score is higher for bona fide audio. The original [clovaai/aasist source](https://github.com/clovaai/aasist) maps `bonafide` to class 1 and spoof to class 0. The selected artifact was already a published ONNX export, so no PyTorch checkpoint conversion or training was needed.

`scripts/download_model.py` fetches that immutable revision and validates SHA-256 `130e536266b7c537f9a13029e1612a9f392fd1cc827783683b6d1c062a3db5e1`. Downloaded file: `models/aasist.onnx`, 1,615,195 bytes; ignored by Git. `ONNX_MODEL.md` documents graph details and interpretation.

## Runtime details

Tested package versions: CPython 3.12.15, ONNX Runtime 1.31.0, NumPy 2.5.4, SciPy 1.18.1, ONNX 1.23.2, pytest 8.3.5. ORT graph input is float32 `wav` `[batch, 64600]`; output is float32 `logits` `[batch, 2]`. The graph is opset 17 (see model file) and has 2,664 nodes. Input audio is mono float32 16 kHz. Preprocessing performs polyphase resampling only when necessary, then first-window truncation or periodic repeat padding to exactly 64,600 samples. No peak normalization is applied.

Output class 0 (spoof) and class 1 (bona fide) logits are transformed with stable softmax, yielding an uncalibrated normalized spoof score. It is not a calibrated probability, and confidence is null. The service threshold remains in `DetectorConfig` / `VERIFVOICE_THRESHOLD`.

## Commands and observed results

Commands run in the isolated environment:

```powershell
uv pip install --python .venv\Scripts\python.exe onnxruntime==1.31.0 scipy==1.18.1
uv pip install --python .venv\Scripts\python.exe onnx==1.23.2
uv pip install --python .venv\Scripts\python.exe -r requirements.txt -r requirements-test.txt
python scripts\download_model.py
python -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp
```

The model downloaded and matched the pinned SHA-256. `onnx.checker.check_model` accepted the graph. Direct CPU ONNX inference succeeded on generated synthetic input and returned finite logits. Final suite: 28 passed, including real model inference and CLI integration. The real-model tests exercise inference plumbing only; no labeled genuine/spoof dataset was evaluated. PyTorch/ONNX numerical parity was not compared because the published ONNX artifact was selected and the PyTorch stack was not installed.

The end-to-end CLI was run with `VERIFVOICE_ADAPTER=aasist_onnx` on a generated one-second 16 kHz tone. It returned `status=success`, `model_name=AASIST (ASVspoof 2019 LA pretrained)`, `authenticity_score=0.999776389388764`, and `classification=potential_spoof`. This synthetic tone is not a genuine/spoof evaluation sample and says nothing about detection accuracy.

The model-specific downloader initially failed on Windows because a temporary file remained open during atomic replacement. Closing it before replace fixed the issue; rerunning the downloader succeeded. A later baseline test run also exposed managed-temp permissions; the workspace-local `--basetemp` workaround made the suite pass.

## IBM LinuxONE compatibility

Not tested on s390x. The ONNX Runtime build documentation lists Linux S390X as a supported build architecture. PyPI had no s390x wheels for ONNX Runtime 1.31.0, SciPy 1.18.1, or ONNX 1.23.2 when inspected. Plan for source-built or IBM/distribution-provided packages and validate NumPy/SciPy native dependencies, ONNX operators, startup, and inference on the target LinuxONE host. Do not claim deployment readiness from this Windows run.

## Limitations

- The artifact is small and redistributable under the model card's MIT label, but is downloaded rather than committed to Git; download integrity is enforced by SHA-256.
- Model-card publisher-reported error rates worsen sharply on cross-domain sets (ASVspoof2019 LA 0.83% EER vs. InTheWild 43.01% and CD-ADD 51.05%). These are not measurements by this project and show why a generated-tone smoke result is not accuracy evidence.
- Only PCM WAV is decoded, no replay-specific capability is claimed, and threshold/classification is not a fraud or payment decision.
