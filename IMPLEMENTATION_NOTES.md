# Implementation notes

## Environment inspected

The project directory was empty (no source, requirements file, or virtual environment); it was not a Git repository. Windows x86-64 is the observed host. `uv 0.13.0` is installed, but neither `python` nor `py` could initially launch an installed interpreter. `uv python list` showed downloadable CPython 3.10–3.15 runtimes. The first isolated Python 3.12 download and pytest install failed because outbound network sockets were blocked. Both succeeded after requesting escalation. No pre-existing project environment could be reused.

## Model choice

AASIST is a credible research direction. The [upstream `clovaai/aasist` repository](https://github.com/clovaai/aasist) documents a PyTorch training/evaluation implementation and ASVspoof 2019 workflow; this checkout had no bundled vetted weights or model runtime. A separate [Hugging Face AASIST ONNX model card](https://huggingface.co/SpeechAntiSpoofingBenchmarks/AASIST) describes raw waveform input and a bona-fide-oriented score and labels its model MIT, but its runtime, artifact provenance, dependencies, s390x operator support, and execution could not be established here. We therefore do not download or silently enable it. No training was performed.

The chosen safe fallback is a WAV validator plus explicit unavailable adapter and stable injection interface. Test-only adapters exercise the contract; they are not detector evidence. This keeps the current package free of model-specific assumptions and third-party runtime requirements.

## IBM LinuxONE next checks

On an actual s390x LinuxONE host/container: record OS/Python/architecture; install the chosen adapter dependencies from an architecture-compatible lock; verify binary wheel availability or documented source builds for each native dependency; inspect model input/output and required operators; run the shared adapter suite; run inference on consented labeled genuine/spoof examples; report metrics and dataset provenance; then run the FastAPI integration under load. No IBM Z compatibility is claimed today.

## Current limitations

No pretrained model inference or labeled real-sample evaluation was completed. The isolated CPython 3.12.15 environment was created and exact test dependencies installed; 18 automated tests passed. The CLI was executed on a generated 16 kHz, 1-second WAV and correctly returned `model_unavailable`, a null score, `uncertain` classification, and a model-not-configured warning. IBM Z validation remains unperformed. WAV-only decoding uses the standard library. No model license is applied to the module because no model artifact is included.
