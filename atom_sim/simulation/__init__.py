# -*- coding: utf-8 -*-
"""
仿真模块

提供轨迹执行和探测统计。
"""

from .trajectory import (
    EmissionResult,
    run_dual_atom_emission,
    apply_fiber_channel,
)
from .detection import (
    DetectionEvent,
    TwoPhotonDetectionResult,
    SuccessEnumerationResult,
    run_detection_pipeline,
    build_detection_effects_6d,
    extract_spin_state,
    compute_fidelity_with_bell,
)

__all__ = [
    # trajectory
    'EmissionResult',
    'run_dual_atom_emission',
    # apply_* functions
    'apply_fiber_channel',
    # detection (POVM sampling)
    'DetectionEvent',
    'TwoPhotonDetectionResult',
    'SuccessEnumerationResult',
    'run_detection_pipeline',
    'build_detection_effects_6d',
    'extract_spin_state',
    'compute_fidelity_with_bell',
]
