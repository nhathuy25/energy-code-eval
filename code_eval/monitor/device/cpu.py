""" CPU initialization and measurement module.
Inspired from zeus.device.cpu
"""

from __future__ import annotations

import os
from typing import Sequence, Literal
from dataclasses import dataclass

from functools import lru_cache

RAPL_DIR = "/sys/class/powercap/intel-rapl"

# Location of RAPL files when in a docker container. See
# https://ml.energy/zeus/getting_started/#system-privileges for more details
CONTAINER_RAPL_DIR = "/zeus_sys/class/powercap/intel-rapl"


# Assuming a maximum power draw of 1000 Watts when we are polling every 0.1 seconds, the maximum
# amount the RAPL counter would increase
RAPL_COUNTER_MAX_INCREASE = 1000 * 1e6 * 0.1

_cpus: CPUs | None = None


class CPUError(Exception):
    """Base exception for GPU-related errors."""
    pass


def get_current_cpu_index(pid: int | Literal["current"] = "current") -> int:
    """Retrieves the specific CPU index (socket) where the given PID is running.

    If no PID is given or pid is "current", the CPU index returned is of the CPU running the current process.

    !!! Note
        Linux schedulers can preempt and reschedule processes to different CPUs. To prevent this from happening
        during monitoring, use `taskset` to pin processes to specific CPUs.
    """
    if pid == "current":
        pid = os.getpid()

    with open(f"/proc/{pid}/stat") as stat_file:
        cpu_core = int(stat_file.read().split()[38])

    with open(
        f"/sys/devices/system/cpu/cpu{cpu_core}/topology/physical_package_id"
    ) as phys_package_file:
        return int(phys_package_file.read().strip())
    

class CPU:
    def __init__(self, cpu_index: int) -> None:
        """Initialize the CPU with a specified index."""
        self.cpu_index = cpu_index

    def getTotalEnergyConsumption(self) -> CpuDramMeasurement:
        """Returns the total energy consumption of the specified powerzone. Units: mJ."""
        pass
        
    def getTotalEnergyConsumption(self) -> CpuDramMeasurement:
        """Returns the total energy consumption of the specified powerzone. Units: mJ."""
        pass


class CPUs:
    pass


class EmptyCPUs(CPUs):
    """Empty CPUs management object to be used when CPUs management object is unavailable.

    Calls to any methods will return a value error and the length of this object will be 0
    """

    def __init__(self) -> None:
        """Instantiates empty CPUs object."""
        pass

    def __del__(self) -> None:
        """Shuts down the Intel CPU monitoring."""
        pass

    @property
    def cpus(self) -> Sequence[CPU]:
        """Returns a list of CPU objects being tracked."""
        return []

    def getTotalEnergyConsumption(self, index: int) -> CpuDramMeasurement:
        """Returns the total energy consumption of the specified powerzone. Units: mJ."""
        raise ValueError("No CPUs available.")

    def supportsGetDramEnergyConsumption(self, index: int) -> bool:
        """Returns True if the specified CPU powerzone supports retrieving the subpackage energy consumption."""
        raise ValueError("No CPUs available.")

    def __len__(self) -> int:
        """Returns 0 since the object is empty."""
        return 0


class CpuDramMeasurement:
    """Represents a measurement of CPU and DRAM energy consumption.

    Attributes:
        cpu_mj (int): The CPU energy consumption in millijoules.
        dram_mj (Optional[int]): The DRAM energy consumption in millijoules. Defaults to None.
    """

    cpu_mj: float
    dram_mj: float | None = None

    def __sub__(self, other: CpuDramMeasurement) -> CpuDramMeasurement:
        """Subtracts the values of another CpuDramMeasurement from this one.

        Args:
            other (CpuDramMeasurement): The other CpuDramMeasurement to subtract.

        Returns:
            CpuDramMeasurement: A new CpuDramMeasurement with the result of the subtraction.
        """
        dram_mj = None
        if self.dram_mj is not None and other.dram_mj is not None:
            dram_mj = self.dram_mj - other.dram_mj
        elif self.dram_mj is not None:
            dram_mj = self.dram_mj
        elif other.dram_mj is not None:
            dram_mj = -other.dram_mj
        return CpuDramMeasurement(self.cpu_mj - other.cpu_mj, dram_mj)

    def __truediv__(self, other: int | float) -> CpuDramMeasurement:
        """Divides the values of this CpuDramMeasurement by a float.

        Args:
            other: The float to divide by.

        Returns:
            CpuDramMeasurement: A new CpuDramMeasurement with the result of the division.

        Raises:
            ZeroDivisionError: If division by zero is attempted.
        """
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Division by zero is not allowed")
            dram_mj = None
            if self.dram_mj is not None:
                dram_mj = self.dram_mj / other
            return CpuDramMeasurement(self.cpu_mj / other, dram_mj)
        else:
            return NotImplemented
        

@lru_cache(maxsize=1)
def rapl_is_available() -> bool:
    """Check if RAPL is available."""
    if not os.path.exists(RAPL_DIR) and not os.path.exists(CONTAINER_RAPL_DIR):
        raise CPUError(
            "RAPL is not supported on this CPU."
        )
        return False
    return True