""" CPU initialization and measurement module.
Inspired from zeus.device.cpu
"""

from __future__ import annotations

import os
from typing import Sequence, Literal
from dataclasses import dataclass

_cpus: CPUs | None = None


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