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
    run_detection_pipeline,
    build_detection_effects_6d,
    extract_spin_state,
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
    # detection (POVM sampling)
    'DetectionEvent',
    'TwoPhotonDetectionResult',
    'SuccessEnumerationResult',
    'run_detection_pipeline',
    'build_detection_effects_6d',
    'extract_spin_state',
    'compute_fidelity_with_bell',
    'compute_photon_statistics',
]
