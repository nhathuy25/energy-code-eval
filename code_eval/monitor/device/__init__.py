from code_eval.monitor.device.gpu import GPUs, nvml_is_available
from code_eval.monitor.device.cpu import CPUs, EmptyCPUs, rapl_is_available

_gpus: GPUs | None = None
_cpus: CPUs | None = None

def get_gpus(ensure_homogeneous: bool = False) -> GPUs:
    """Initialize and return a singleton object for GPU management.

    This function returns a GPU management object that aims to abstract
    the underlying NVIDIA GPU and the specific monitoring library
    pynvml for NVIDIA GPUs

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
            "NVML is unavailable. Failed to initialize GPU management library."
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
