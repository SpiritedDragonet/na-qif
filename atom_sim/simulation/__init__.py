"""
Simulation Module

Provides trajectory execution and multi-trial statistics.
"""

from .trajectory import (
    TrajectoryRunner,
    TrajectoryResult,
    EmissionResult,
    run_single_trajectory,
    run_emission_only,
)
from .simulator import (
    SimulationResult,
    run_simulation,
)

__all__ = [
    # trajectory
    'TrajectoryRunner',
    'TrajectoryResult',
    'EmissionResult',
    'run_single_trajectory',
    'run_emission_only',
    # simulator
    'SimulationResult',
    'run_simulation',
]
