"""
Atom Sim: Time-Bin MPS Simulation for Neutral Atom Quantum Interface

This package provides a pure TeNPy implementation of time-bin MPS simulation
for neutral atom quantum entanglement protocols.

Main exports:
- MPSState: Tensor network state container
- run_simulation: Multi-trial statistics
- Config classes: TimeGrid, EmitParams, QFCParams, FiberParams, DetParams
"""

from .core.mps import MPSState, create_timebin_mps
from .config import TimeGrid, EmitParams, QFCParams, FiberParams, DetParams, SimParams
from .simulation import run_simulation, TrajectoryResult, SimulationResult

__version__ = '0.1.0'

__all__ = [
    # core
    'MPSState',
    'create_timebin_mps',
    # config
    'TimeGrid',
    'EmitParams',
    'QFCParams',
    'FiberParams',
    'DetParams',
    'SimParams',
    # simulation
    'run_simulation',
    'TrajectoryResult',
    'SimulationResult',
]
