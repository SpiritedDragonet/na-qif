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
    apply_detection,
    find_bsm_success,
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
    check_bsm_success as check_bsm_success_qjump,
    compute_fidelity_with_bell,
    compute_photon_statistics,
    build_jump_operators_18d,
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
    'apply_detection',
    'find_bsm_success',
    # simulator
    'SimulationResult',
    'run_simulation',
    # detection (quantum jump method)
    'DetectionEvent',
    'TwoPhotonDetectionResult',
    'run_two_photon_detection',
    'extract_spin_state',
    'check_bsm_success_qjump',
    'compute_fidelity_with_bell',
    'compute_photon_statistics',
    'build_jump_operators_18d',
]
