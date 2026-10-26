"""Small shared helpers for devices and reproducibility."""
import torch


def get_device(name="auto"):
    """Resolve a device name. "auto" prefers CUDA, then MPS, then CPU."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    try:
        device = torch.device(name)
    except (RuntimeError, ValueError):
        raise ValueError(f"unknown device {name!r}") from None

    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    mps = getattr(torch.backends, "mps", None)
    if device.type == "mps" and (mps is None or not mps.is_available()):
        raise ValueError("MPS was requested but is not available")
    if device.type not in ("cpu", "cuda", "mps"):
        raise ValueError(f"unsupported device {name!r}")
    return device


def set_seed(seed):
    """Seed PyTorch on CPU and, when present, CUDA/MPS."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available() and hasattr(torch, "mps"):
        torch.mps.manual_seed(seed)
