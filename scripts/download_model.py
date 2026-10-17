"""Download the pinned public AASIST ONNX artifact and verify its SHA-256."""

from hashlib import sha256
import os
from pathlib import Path
import tempfile
from urllib.request import urlopen


REPOSITORY = "SpeechAntiSpoofingBenchmarks/AASIST"
REVISION = "16774d458d86d2a021ae31646c1bf66a5331b53e"
FILENAME = "aasist.onnx"
EXPECTED_SHA256 = "130e536266b7c537f9a13029e1612a9f392fd1cc827783683b6d1c062a3db5e1"
URL = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{FILENAME}?download=true"


def main() -> int:
    """Fetch the immutable artifact revision and atomically install after hashing."""
    project_root = Path(__file__).resolve().parents[1]
    model_dir = project_root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    destination = model_dir / FILENAME
    descriptor, temporary_name = tempfile.mkstemp(dir=model_dir, prefix=".aasist-")
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        with temporary_path.open("wb") as temporary:
            with urlopen(URL, timeout=60) as response:
                while chunk := response.read(1024 * 1024):
                    temporary.write(chunk)
        digest = sha256(temporary_path.read_bytes()).hexdigest()
        if digest != EXPECTED_SHA256:
            raise RuntimeError(f"Downloaded model checksum mismatch: expected {EXPECTED_SHA256}, received {digest}")
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    print(f"Verified {REPOSITORY}@{REVISION}:{FILENAME} -> {destination}")
    print(f"sha256={EXPECTED_SHA256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

