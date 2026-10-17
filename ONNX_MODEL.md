# AASIST ONNX artifact

## Provenance and acquisition

- Repository: [SpeechAntiSpoofingBenchmarks/AASIST](https://huggingface.co/SpeechAntiSpoofingBenchmarks/AASIST), a publisher-maintained AASIST benchmark model repository.
- Pinned revision: `16774d458d86d2a021ae31646c1bf66a5331b53e`.
- Checkpoint basis: publisher identifies it as the official AASIST architecture using the upstream ASVspoof 2019 LA pretrained checkpoint.
- License: MIT, as declared by the model card (license inherited from the upstream AASIST project).
- Artifact: `aasist.onnx`, 1,615,195 bytes; SHA-256 `130e536266b7c537f9a13029e1612a9f392fd1cc827783683b6d1c062a3db5e1`.
- Reproduce with `python scripts/download_model.py`. The script uses the immutable revision and refuses to replace the destination if the checksum differs. The artifact is intentionally ignored by Git.

## ONNX graph and tested runtime

The published artifact was used directly; no ONNX conversion occurred. Verified locally with ONNX 1.23.2 (`onnx.checker.check_model`) and ONNX Runtime 1.31.0 CPU provider.

- Opset: 17.
- Input: name `wav`, type float32, shape `[batch, 64600]` (dynamic batch dimension).
- Output: name `logits`, type float32, shape `[batch, 2]` (dynamic batch dimension).
- Graph: 2,664 nodes.
- Tested host: Windows 11 x86-64, CPython 3.12.15.

The graph loaded through ONNX Runtime; adapter validates the names, types, ranks, and fixed sample/class dimensions before accepting it. CPU execution is default. Other execution providers must be explicitly configured and present in `onnxruntime.get_available_providers()`.

## Preprocessing and output mapping

The publisher model card specifies mono float32 PCM at 16 kHz, no resampling in its wrapper, and a deterministic 64,600-sample input: truncate to the first window or repeat a short waveform to fill it. This project preserves the same window policy and PCM scaling, but converts other input sample rates with `scipy.signal.resample_poly` first. Generic WAV handling converts channels to mono by averaging.

The model card describes a 2-class bona-fide/spoof output and calls its bona-fide value a logit. The upstream training input code assigns label 1 to `bonafide` and 0 otherwise. This ONNX graph returns two logits, so the adapter applies stable softmax and exposes class 0 as spoof evidence. This normalized class score is **not calibrated**. It does not provide replay detection or a confidence value. The threshold is owned by the service configuration, not the model adapter.

No PyTorch conversion parity check was performed; this exact pretrained ONNX artifact was selected instead. No labeled evaluation dataset was available in this project.

## IBM LinuxONE

No LinuxONE execution was performed. ONNX Runtime documentation lists Linux S390X as a supported build architecture, but the tested PyPI runtime and SciPy releases had no s390x wheel. Build or obtain compatible binaries, then validate graph loading, operators, input parity, and real inference on the target host.

