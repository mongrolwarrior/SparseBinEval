"""Host capacity probes (RAM + disk), stdlib-only.

Companion to gpu.py (which probes VRAM). Used to warn — before downloading or
holding a large corpus — when a container is too small for the MEDLINE dataset.
"""

import os
import shutil


def available_ram_bytes():
    """Memory the OS can hand out right now (MemAvailable), or 0 if unknown."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024     # value is in kB
    except OSError:
        pass
    try:                                                   # fallback: free pages × page size
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError):
        return 0


def total_ram_bytes():
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError):
        return 0


def free_disk_bytes(path):
    """Free bytes on the filesystem that holds `path` (or its nearest parent)."""
    p = os.path.abspath(path)
    while p and not os.path.exists(p):
        p = os.path.dirname(p)
    try:
        return shutil.disk_usage(p or "/").free
    except OSError:
        return 0


def human(n):
    """Human-readable byte count, e.g. 1.5 GiB."""
    n = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}"
        n /= 1024
