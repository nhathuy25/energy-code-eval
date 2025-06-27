"""NVIDIA GPUs. Inspired from zeus.device.gpu.nvidia
Source: https://github.com/ml-energy/zeus/tree/master/zeus/device/gpu
"""

from __future__ import annotations

import os
import warnings
import functools
import contextlib
from pathlib import Path
from typing import Sequence
from functools import lru_cache

import pynvml

@lru_cache(maxsize=1)
def nvml_is_available() -> bool:
    """Check if NVML is available."""
    try:
        import pynvml
    except ImportError:
        """
        logger.info(
            "Failed to import `pynvml`. Make sure you have `nvidia-ml-py` installed."
        )
        """
        return False
    
    # Detect unofficial pynvml packages.
    # If detected, this should be a critical error.
    if not hasattr(pynvml, "_nvmlGetFunctionPointer"):
        #logger.error("Unoffical pynvml package detected!")
        raise ImportError(
            "Unofficial pynvml package detected! "
            "This causes conflicts with the official NVIDIA bindings. "
            "Please remove with `pip uninstall pynvml` and instead use the official "
            "bindings from NVIDIA: `nvidia-ml-py`. "
        )

    try:
        pynvml.nvmlInit()
        #logger.info("pynvml is available and initialized.")
        return True
    except pynvml.NVMLError as e:
        #logger.info("pynvml is available but could not initialize NVML: %s.", e)
        return False


class GPUError(Exception):
    """Base exception for GPU-related errors."""
    pass

def _handle_nvml_errors(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except pynvml.NVMLError as e:
            error_message = GPU._exception_map.get(
                e.value,  
                "Unknown NVIDIA GPU error occurred"
            )
            raise GPUError(f"{error_message}: {str(e)}") from e

    return wrapper


class GPU:
    _exception_map = {
        pynvml.NVML_ERROR_UNINITIALIZED: "NVIDIA Management Library not initialized",
        pynvml.NVML_ERROR_INVALID_ARGUMENT: "Invalid argument provided to NVML function",
        pynvml.NVML_ERROR_NOT_SUPPORTED: "This operation is not supported by the device",
        pynvml.NVML_ERROR_NO_PERMISSION: "No permission to perform this operation",
        pynvml.NVML_ERROR_ALREADY_INITIALIZED: "NVIDIA Management Library already initialized",
        pynvml.NVML_ERROR_NOT_FOUND: "Requested GPU device not found",
        pynvml.NVML_ERROR_INSUFFICIENT_SIZE: "Provided buffer is too small",
        pynvml.NVML_ERROR_INSUFFICIENT_POWER: "Insufficient power available for operation",
        pynvml.NVML_ERROR_DRIVER_NOT_LOADED: "NVIDIA driver not loaded",
        pynvml.NVML_ERROR_TIMEOUT: "Operation timed out",
        pynvml.NVML_ERROR_IRQ_ISSUE: "Problem with interrupt request (IRQ)",
        pynvml.NVML_ERROR_LIBRARY_NOT_FOUND: "NVML library not found",
        pynvml.NVML_ERROR_FUNCTION_NOT_FOUND: "Function not found in NVML library",
        pynvml.NVML_ERROR_CORRUPTED_INFOROM: "GPU InfoROM is corrupted",
        pynvml.NVML_ERROR_GPU_IS_LOST: "GPU has fallen off the bus or is otherwise inaccessible",
        pynvml.NVML_ERROR_RESET_REQUIRED: "GPU requires a reset before it can be used again",
        pynvml.NVML_ERROR_OPERATING_SYSTEM: "Operating system error occurred",
        pynvml.NVML_ERROR_LIB_RM_VERSION_MISMATCH: "NVIDIA Resource Manager library version mismatch",
        pynvml.NVML_ERROR_MEMORY: "Memory error occurred during operation",
        pynvml.NVML_ERROR_UNKNOWN: "An unknown NVIDIA Management Library error occurred",
        }
    
    def __init__(self, gpu_index: int) -> None:
        self.gpu_index = gpu_index
        self.get_handle()
        self._supportsGetTotalEnergyConsumption = None        
            
    @_handle_nvml_errors
    def getName(self) -> str:
        """Return the name of the GPU model."""
        return pynvml.nvmlDeviceGetName(self.handle)

    @_handle_nvml_errors    
    def get_handle(self):
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)

    @_handle_nvml_errors
    def getInstantPowerUsage(self) -> int:
        """Return the current power draw of the GPU. Units: mW."""
        metric = pynvml.nvmlDeviceGetFieldValues(
            self.handle, [pynvml.NVML_FI_DEV_POWER_INSTANT]
        )[0]
        if (ret := metric.nvmlReturn) != pynvml.NVML_SUCCESS:
            raise pynvml.NVMLError(ret)
        return metric.value.uiVal

    @_handle_nvml_errors
    def supportsGetTotalEnergyConsumption(self) -> bool:
        """Check if the GPU supports retrieving total energy consumption."""
        # Supported on Volta or newer microarchitectures
        if self._supportsGetTotalEnergyConsumption is None:
            self._supportsGetTotalEnergyConsumption = (
                pynvml.nvmlDeviceGetArchitecture(self.handle)
                >= pynvml.NVML_DEVICE_ARCH_VOLTA
            )

        return self._supportsGetTotalEnergyConsumption
    
    @_handle_nvml_errors
    def getTotalEnergyConsumption(self) -> int:
        """Return the total energy consumption of the specified GPU. Units: mJ."""
        return pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
    
    @_handle_nvml_errors
    def getInstantTemperature(self) -> int:
        """Return the current temperature of GPU. Units: °C"""
        return pynvml.nvmlGetDeviceTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)


class GPUs:
    def __init__(self, ensure_homogeneous: bool = False) -> None:
        """Initialize NVML and sets up the GPUs.

        Args:
            ensure_homogeneous (bool): If True, ensures that all tracked GPUs have the same name.
        """
        try:
            pynvml.nvmlInit()
            self._init_gpus()
            if ensure_homogeneous:
                self._ensure_homogeneous()
        except pynvml.NVMLError as e:
            raise e
        
    @property
    def gpus(self) -> Sequence[GPU]:
        """Return a list of NVIDIAGPU objects being tracked."""
        return self._gpus

    def _init_gpus(self) -> None:
        # Must respect `CUDA_VISIBLE_DEVICES` if set
        if (visible_device := os.environ.get("CUDA_VISIBLE_DEVICES")) is not None:
            if not visible_device:
                raise GPUError(
                    "CUDA_VISIBLE_DEVICES is set to an empty string. "
                    "It should either be unset or a comma-separated list of GPU indices."
                )
            if visible_device.startswith("MIG"):
                raise GPUError(
                    "CUDA_VISIBLE_DEVICES contains MIG devices. NVML (the library used by Zeus) "
                    "currently does not support measuring the power or energy consumption of MIG "
                    "slices. You can still measure the whole GPU by temporarily setting "
                    "CUDA_VISIBLE_DEVICES to integer GPU indices and restoring it afterwards."
                )
            visible_indices = [int(idx) for idx in visible_device.split(",")]
        else:
            visible_indices = list(range(pynvml.nvmlDeviceGetCount()))
        
        self._gpus = [GPU(gpu_num) for gpu_num in visible_indices]

    def _ensure_homogeneous(self) -> None:
        """Ensures that all tracked GPUs are homogeneous in terms of name."""
        gpu_names = [gpu.getName() for gpu in self.gpus]
        # Both zero (no GPUs found) and one are fine.
        if len(set(gpu_names)) > 1:
            raise ValueError(f"Heterogeneous GPUs found: {gpu_names}")
        
    def __del__(self) -> None:
        """Shut down NVML."""
        with contextlib.suppress(pynvml.NVMLError):
            pynvml.nvmlShutdown()

    def __len__(self) -> int:
        """Return the number of GPUs being tracked."""
        return len(self.gpus)

    def getName(self, gpu_index: int) -> str:
        """Return the name of the specified GPU."""
        return self.gpus[gpu_index].getName()

    def getInstantPowerUsage(self, gpu_index: int) -> int:
        """Return the current power draw of the GPU. Units: mW."""
        return self.gpus[gpu_index].getInstantPowerUsage()
    
    def supportsGetTotalEnergyConsumption(self, gpu_index: int) -> bool:
        """Check if the GPU supports retrieving total energy consumption."""
        return self.gpus[gpu_index].supportsGetTotalEnergyConsumption()

    def getTotalEnergyConsumption(self, gpu_index: int) -> int:
        """Return the total energy consumption of the GPU since driver load. Units: mJ."""
        return self.gpus[gpu_index].getTotalEnergyConsumption()
    
    def getInstantTemperature(self, gpu_index: int) -> int:
        """Return the current temperature of the specified GPU. Units: °C."""
        return self.gpus[gpu_index].getInstantTemperature()