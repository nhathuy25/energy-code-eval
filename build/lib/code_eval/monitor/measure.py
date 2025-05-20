import pynvml
from time import time, sleep
import threading
from dataclasses import dataclass
from typing import Iterable, Dict, List
import multiprocessing as mp

@dataclass
class BeginState:
    time: float
    gpu_energy: Dict[int, float]
    memory_usage: Dict[List] |None
    gpu_power: Dict[List] | None


    def total_gpus_energy(self) -> float:
        return sum(self.gpu_energy.values())
    
@dataclass
class EndState:
    time: float
    gpu_energy: Dict[int, float]
    memory_usage: Dict[List] |None
    gpu_power: Dict[List] | None


    def total_gpus_energy(self) -> float:
        return sum(self.gpu_energy.values())


class EnergyMonitor:
    def __init__(
            self,
            label: str,
            gpu_indices: List[int],
            update_period: float,
    ):
        pass

    def begin(self, label: str):
        pass

    def end(self, label: str):
        pass

    def extract_power(self, label: str):
        pass


