from __future__ import annotations

import types
from typing import Literal
from functools import lru_cache

#from zeus.utils.logging import get_logger

#logger = get_logger(name=__name__)
MODULE_CACHE: dict[str, types.ModuleType] = {}

@lru_cache(maxsize=1)
def torch_is_available() -> bool:
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        MODULE_CACHE["torch"] = torch
        if not cuda_available:
            raise RuntimeError("PyTorch is available but does not have CUDA support.")
        return True
    except ImportError as e:
        return False


def sync_execution(
    gpu_devices: list[int]
) -> None:
    """Block until all computations on the specified devices are finished.

    PyTorch only runs GPU computations asynchronously, so synchronizing computations
    for the given GPU devices is done by calling `torch.cuda.synchronize` on each
    device. 

    Args:
        gpu_devices: GPU device indices to synchronize.
    """
    if torch_is_available():
        torch = MODULE_CACHE["torch"]
        for device in gpu_devices:
            torch.cuda.synchronize(device)
        return

    raise RuntimeError("No framework is available.")
