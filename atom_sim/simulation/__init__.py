# -*- coding: utf-8 -*-
"""
仿真模块

提供轨迹执行和多试验统计。
"""

from .trajectory import (
    TrajectoryRunner,
    TrajectoryResult,
    EmissionResult,
    run_single_trajectory,
    run_emission_only,
    apply_qfc,
    apply_780_filter,
    apply_jones,
    apply_loss,
    apply_loss_combined,
    apply_bs,
    apply_fiber_channel,
    project_to_1517,
)
from .simulator import (
    SimulationResult,
    run_simulation,
)
from .detection import (
    DetectionEvent,
    TwoPhotonDetectionResult,
    run_two_photon_detection,
    extract_spin_state,
    check_bsm_success,
    compute_fidelity_with_bell,
    compute_photon_statistics,
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
    'apply_780_filter',
    'apply_jones',
    'apply_loss',
    'apply_loss_combined',
    'apply_bs',
    'apply_fiber_channel',
    'project_to_1517',
    # simulator
    'SimulationResult',
    'run_simulation',
    # detection (Kraus measurement)
    'DetectionEvent',
    'TwoPhotonDetectionResult',
    'run_two_photon_detection',
    'extract_spin_state',
    'check_bsm_success',
    'compute_fidelity_with_bell',
    'compute_photon_statistics',
]
