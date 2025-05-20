from code_eval.monitor.device.gpu import GPUs, nvml_is_available
from code_eval.monitor.device.cpu import CPUs, EmptyCPUs, rapl_is_available

_gpus: GPUs | None = None
_cpus: CPUs | None = None

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


def get_cpus() -> CPUs:
    """Initialize and return a singleton CPU monitoring object for INTEL CPUs.

    The function returns a CPU management object that aims to abstract the underlying CPU monitoring libraries
    (RAPL for Intel CPUs).

    This function attempts to initialize CPU mointoring using RAPL. If this attempt fails, it raises
    a ZeusErrorInit exception.
    """
    global _cpus
    if _cpus is not None:
        return _cpus
    if rapl_is_available():
        _cpus = EmptyCPUs() # TODO: Change to RAPL when finish CPU module
        # _cpus = CPUs()
        return _cpus
    else:
        raise (
            "RAPL unvailable Failed to initialize CPU management library."
        )
