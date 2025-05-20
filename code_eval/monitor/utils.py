from __future__ import annotations

import types
from typing import Literal
from functools import lru_cache

#from zeus.utils.logging import get_logger

logger = get_logger(name=__name__)
MODULE_CACHE: dict[str, types.ModuleType] = {}

def sync_execution(
    gpu_devices: list[int]
) -> None:
    """Block until all computations on the specified devices are finished.

    PyTorch only runs GPU computations asynchronously, so synchronizing computations
    for the given GPU devices is done by calling `torch.cuda.synchronize` on each
    device. On the other hand, JAX runs both CPU and GPU computations asynchronously,
    but by default it only has a single CPU device (id=0). Therefore for JAX, all GPU
    devices passed in and the CPU device (id=0) are synchronized.

    !!! Note
        `jax.device_put` with `block_until_ready` is used to synchronize computations
        on JAX devices. This is a workaround to the lack of a direct API for
        synchronizing computations on JAX devices. Tracking issue:
        https://github.com/google/jax/issues/4335

    !!! Note
        Across the Zeus library, an integer device index corresponds to a single whole
        physical device. This is usually what you want, except when using more advanced
        device partitioning (e.g., using `--xla_force_host_platform_device_count` in JAX
        to partition CPUs into more pieces). In such cases, you probably want to opt out
        from using this function and handle synchronization manually at the appropriate
        granularity.

    Args:
        gpu_devices: GPU device indices to synchronize.
        sync_with: Deep learning framework to use to synchronize computations.
            Defaults to `"torch"`, in which case `torch.cuda.synchronize` will be used.
    """
    
    torch = MODULE_CACHE["torch"]
    for device in gpu_devices:
        torch.cuda.synchronize(device)
    return

    raise RuntimeError("No framework is available.")
