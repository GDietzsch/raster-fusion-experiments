"""Device selection is explicit; requested CUDA never silently falls back."""
import importlib.metadata
import os
import platform
import subprocess
import sys
import psutil


def select_device(requested: str = "auto"):
    import torch
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError(f"Unknown device: {requested}")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Check the NVIDIA driver and PyTorch CUDA installation.")
    return torch.device("cuda" if requested != "cpu" and torch.cuda.is_available() else "cpu")


def environment() -> dict:
    info = {"python": sys.version, "executable": sys.executable, "os": platform.platform(),
            "cpu": platform.processor(), "cpu_cores": os.cpu_count(), "ram_bytes": psutil.virtual_memory().total,
            "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}}
    try:
        import torch
        info.update(pytorch=torch.__version__, cuda_runtime=torch.version.cuda, cuda_available=torch.cuda.is_available())
        if torch.cuda.is_available():
            info.update(gpu=torch.cuda.get_device_name(0), vram_bytes=torch.cuda.get_device_properties(0).total_memory,
                        allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved())
    except ImportError:
        info.update(pytorch=None, cuda_available=False)
    try:
        info["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        info["git_commit"] = None
    return info
