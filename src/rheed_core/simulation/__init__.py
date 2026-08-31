"""Framework-neutral local RHEED simulation support.

The core package owns stable requests and result contracts. Optional adapters
own simulator-specific imports and translate their outputs into NumPy arrays.
"""

from .base import (
    SimulationAdapter,
    SimulationAdapterDescriptor,
    create_simulation_adapter,
    list_simulation_adapters,
)
from .geometry import measure_detector_distance
from .runner import SimulationCancelled, run_simulation
from .store import SimulationStore, StoredSimulationRun
from .types import SimulationContext, SimulationRunResult, SimulationSpec

__all__ = [
    "SimulationAdapter",
    "SimulationAdapterDescriptor",
    "SimulationCancelled",
    "SimulationContext",
    "SimulationRunResult",
    "SimulationSpec",
    "SimulationStore",
    "StoredSimulationRun",
    "create_simulation_adapter",
    "list_simulation_adapters",
    "measure_detector_distance",
    "run_simulation",
]
