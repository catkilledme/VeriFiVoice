# VerifVoice audio authenticity module

This package validates PCM WAV input and calls a replaceable model adapter. There is no vetted model artifact bundled, so the default adapter returns `model_unavailable` and never invents a spoof score. The module does not infer fraud, coercion, speaker identity, or payment action. Synthetic speech may be legitimate assistive communication.

## Setup and execution

Windows PowerShell (Python 3.10–3.13 is suitable for this standard-library runtime; Python itself must be installed):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py --diagnostics
python main.py .\sample.wav
python -m pytest -q
```

Input is uncompressed PCM WAV (8/16/24/32-bit), mono or multi-channel; channels are averaged to mono. Defaults limit inputs to 25 MiB and 60 seconds and flag near-silence as low quality. Configure with `VERIFVOICE_ADAPTER`, `VERIFVOICE_MAX_FILE_BYTES`, `VERIFVOICE_MAX_DURATION`, `VERIFVOICE_MIN_DURATION`, `VERIFVOICE_MIN_RMS`, and `VERIFVOICE_THRESHOLD`.

## Public contract

`analyze_audio(audio_path, detector=None, config=None) -> AudioAnalysisResult` is the integration API. `to_dict()` returns JSON-ready contract version `1.0`. `authenticity_score` currently carries an **uncalibrated spoof score** in `[0,1]` only if an adapter supplies one. `calibrated_probability` is a separate nullable field populated only if the adapter provides evaluated calibration. `confidence` remains null because the current detector does not define a separate confidence measure. `classification` uses a configurable threshold only as an evidence label. `quality_score` is a simple RMS-derived audio usability indicator, not detector confidence. `status` distinguishes success, low_quality, model_unavailable, and inference_error. Missing evidence is null.

## FastAPI integration

```python
from fastapi import FastAPI, UploadFile
from audio_authenticity import analyze_audio
from pathlib import Path
from tempfile import TemporaryDirectory

app = FastAPI()

@app.post("/audio-authenticity")
async def inspect_audio(file: UploadFile):
    with TemporaryDirectory() as directory:
        path = Path(directory) / "upload.wav"
        path.write_bytes(await file.read())
        return analyze_audio(str(path)).to_dict()
```

Map `inference_error` and `model_unavailable` to service-health/error handling separately from a successful low spoof score. Keep coercion analysis, transaction context, risk fusion, and bank actions outside this module.

## Testing and environment

`pytest -q` covers synthetic WAV preprocessing, stereo downmix, bad inputs, output bounds, failure states, and adapter interchangeability. Synthetic tones validate plumbing only; they are not spoof-detection evidence. No labeled evaluation or real-model inference has been performed.

The current implementation has no third-party runtime dependencies, which avoids claiming compatibility for audio wheels or ONNX Runtime on IBM Z. LinuxONE has not been tested. For a future adapter, validate the Python version, framework and native wheels on s390x, model operators, waveform requirements, and inference with representative labeled samples on an actual LinuxONE host/container. Do not infer IBM Z support from Windows results.

## Known limitations

- No pretrained model is configured, so production scores are unavailable.
- WAV only; compressed audio formats are intentionally rejected.
- RMS quality is only a basic silence check, not an acoustic quality estimate.
- No calibrated threshold or performance claims; domain shift, codecs, noise, replay, and unseen generators need evaluation.
- The registry is explicit and process-local; register an adapter at application startup.
