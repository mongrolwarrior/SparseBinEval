"""Query GPU memory via nvidia-smi (no extra Python dependencies)."""

import subprocess


def free_total_bytes():
    """Return (free_bytes, total_bytes) for GPU 0."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free,memory.total",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout
    free_mib, total_mib = (int(x) for x in out.strip().splitlines()[0].split(","))
    return free_mib * 1024 * 1024, total_mib * 1024 * 1024
