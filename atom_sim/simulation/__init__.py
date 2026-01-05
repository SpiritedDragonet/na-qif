"""
Simulation Module

Provides trajectory execution and multi-trial statistics.
"""

from .trajectory import (
    TrajectoryRunner,
    TrajectoryResult,
    run_single_trajectory,
)
from .simulator import (
    SimulationResult,
    run_simulation,
)

__all__ = [
    # trajectory
    'TrajectoryRunner',
    'TrajectoryResult',
    'run_single_trajectory',
    # simulator
    'SimulationResult',
    'run_simulation',
]
