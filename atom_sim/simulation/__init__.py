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
    apply_qfc,
    apply_jones,
    apply_loss,
    apply_loss_combined,
    apply_bs,
    apply_fiber_channel,
    apply_detection,
    find_bsm_success,
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
    # apply_* functions
    'apply_qfc',
    'apply_jones',
    'apply_loss',
    'apply_loss_combined',
    'apply_bs',
    'apply_fiber_channel',
    'apply_detection',
    'find_bsm_success',
    # simulator
    'SimulationResult',
    'run_simulation',
]
