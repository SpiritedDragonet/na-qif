# -*- coding: utf-8 -*-
"""
仿真模块

提供轨迹执行和探测统计。
"""

from .trajectory import (
    EmissionResult,
    run_dual_atom_emission,
    apply_qfc,
    apply_780_filter,
    apply_bs,
    apply_fiber_channel,
    project_to_1517,
)
from .detection import (
    DetectionEvent,
    TwoPhotonDetectionResult,
    SuccessEnumerationResult,
    run_two_photon_detection,
    build_detection_kraus_6d,
    enumerate_success_events,
    compute_two_photon_arrival_prob,
    extract_spin_state,
    check_bsm_success,
    compute_fidelity_with_bell,
    compute_photon_statistics,
)

__all__ = [
    # trajectory
    'EmissionResult',
    'run_dual_atom_emission',
    # apply_* functions
    'apply_qfc',
    'apply_780_filter',
    'apply_bs',
    'apply_fiber_channel',
    'project_to_1517',
    # detection (Kraus measurement)
    'DetectionEvent',
    'TwoPhotonDetectionResult',
    'SuccessEnumerationResult',
    'run_two_photon_detection',
    'build_detection_kraus_6d',
    'enumerate_success_events',
    'compute_two_photon_arrival_prob',
    'extract_spin_state',
    'check_bsm_success',
    'compute_fidelity_with_bell',
    'compute_photon_statistics',
]
