# VerifVoice audio authenticity

This module runs a pretrained AASIST anti-spoofing model through ONNX Runtime. It returns model evidence only; it does not determine fraud, coercion, speaker identity, or payment action. Synthetic speech can be legitimate assistive communication.

## Windows setup

Use CPython 3.12â€“3.14 on Windows x64 with the pinned inference dependencies. The tested environment was CPython 3.12.15, ONNX Runtime 1.31.0, NumPy 2.5.4, and SciPy 1.18.1.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt -r requirements-test.txt
python scripts\download_model.py
```

The downloader retrieves a pinned Hugging Face revision and verifies SHA-256 before installing `models/aasist.onnx`. Model binaries are ignored by Git. To use a different path set `VERIFVOICE_MODEL_PATH`.

Run the real detector:

```powershell
$env:VERIFVOICE_ADAPTER = "aasist_onnx"
python main.py --diagnostics
python main.py .\sample.wav
```

The default adapter remains `unavailable`, so missing model configuration never produces fabricated evidence. Set `VERIFVOICE_ADAPTER=aasist_onnx` to select the pretrained model. `VERIFVOICE_ONNX_PROVIDERS` defaults to `CPUExecutionProvider`; only providers reported by the installed ONNX Runtime are accepted.

## Model and output semantics

The adapter uses `SpeechAntiSpoofingBenchmarks/AASIST`, pinned to revision `16774d458d86d2a021ae31646c1bf66a5331b53e`. The publisher identifies it as the official AASIST architecture with the upstream ASVspoof 2019 LA pretrained checkpoint and MIT license. The ONNX graph takes float32 `wav` shaped `[batch, 64600]` and returns two float32 `logits`. Training maps `bonafide` to class 1 and spoof to class 0. The adapter applies stable softmax to these logits and reports class 0 as an **uncalibrated spoof score**. It does not report a calibrated probability or confidence.

The model wrapper expects mono float32 PCM at 16 kHz. Adapter preprocessing averages channels in the generic loader, uses `scipy.signal.resample_poly` when the source rate differs, preserves PCM scaling, then follows the published deterministic windowing: keep the first 64,600 samples or repeat shorter audio to that length. It does not randomly crop or peak-normalize.

The stable API is `analyze_audio(audio_path, detector=None, config=None) -> AudioAnalysisResult`. Results keep score, calibrated probability, and confidence separate. A process-wide default service loads the model once; applications that inject their own service can call `AudioAnalysisService.close()` at shutdown. The status distinguishes `success`, `low_quality`, `model_unavailable`, and `inference_error`.

Threshold is configured separately with `VERIFVOICE_THRESHOLD` (default 0.5) and only maps model evidence to `likely_genuine`, `potential_spoof`, or `uncertain`; it is not a payment decision. Other settings: `VERIFVOICE_MAX_FILE_BYTES`, `VERIFVOICE_MAX_DURATION`, `VERIFVOICE_MIN_DURATION`, and `VERIFVOICE_MIN_RMS`.

## FastAPI integration

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from fastapi import FastAPI, UploadFile
from audio_authenticity import analyze_audio

app = FastAPI()

@app.post("/audio-authenticity")
async def inspect_audio(file: UploadFile):
    with TemporaryDirectory() as directory:
        path = Path(directory) / "upload.wav"
        path.write_bytes(await file.read())
        return analyze_audio(str(path)).to_dict()
```

The default service is reused across calls. Keep coercion analysis, transaction context, risk fusion, and payment actions in their own services. Treat unavailable and inference errors as operational failures, not as low spoof scores.

## Tests and verification

```powershell
python -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp
```

The suite checks WAV handling, adapter failures, ONNX graph validation, input shape and data type, sample-rate conversion, output semantics, real model inference, and the unchanged CLI integration. The generated-tone smoke sample tests plumbing only; it is not accuracy evidence. No labeled evaluation set was available.

`ONNX_MODEL.md` records model provenance, graph details, download hash, and validation. `IMPLEMENTATION_NOTES.md` records actual commands and compatibility findings. `ADDING_A_MODEL.md` explains adapter replacement.

## IBM LinuxONE

IBM Z was not available for execution. The official ONNX Runtime support page lists Linux s390x as a supported build architecture, but PyPI published no s390x wheels for the tested ONNX Runtime, SciPy, or ONNX versions. LinuxONE deployment therefore needs a compatible IBM/distribution build or source-built runtime, NumPy, and SciPy. Validate those exact builds and graph operators on an s390x host before claiming compatibility. Windows ONNX Runtime also requires the Microsoft Visual C++ 2019 runtime or newer.

## Known limitations

- ASVspoof 2019 training domain differs from real-world audio; model-card cross-domain results degrade substantially. Noise, codecs, replay attacks, new generators, and assistive speech require separate evaluation.
- Only uncompressed PCM WAV is decoded. The RMS quality indicator only catches silence and near-silence.
- The softmax score is uncalibrated model evidence, not an estimated real-world fraud probability.
- No ONNX conversion was done because a published pretrained ONNX artifact was available. PyTorch export parity was not tested.
- IBM LinuxONE runtime execution is not verified.

