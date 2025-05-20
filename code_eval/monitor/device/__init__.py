from code_eval.monitor.device.gpu import GPUs, nvml_is_available


def get_gpus(ensure_homogeneous: bool = False) -> GPUs:
    """Initialize and return a singleton object for GPU management.

    This function returns a GPU management object that aims to abstract
    the underlying GPU vendor and their specific monitoring library
    (pynvml for NVIDIA GPUs and amdsmi for AMD GPUs). Management APIs
    are mapped to methods on the returned [`GPUs`][zeus.device.gpu.GPUs] object.

    GPU availability is checked in the following order:

    1. NVIDIA GPUs using `pynvml`
    1. AMD GPUs using `amdsmi`
    1. If both are unavailable, a `ZeusGPUInitError` is raised.

    Args:
        ensure_homogeneous (bool): If True, ensures that all tracked GPUs have the same name.
    """
    global _gpus
    if _gpus is not None:
        return _gpus

    if nvml_is_available():
        _gpus = GPUs(ensure_homogeneous)
        return _gpus
    else:
        raise RuntimeError(
            "NVML and AMDSMI unavailable. Failed to initialize GPU management library."
        )
