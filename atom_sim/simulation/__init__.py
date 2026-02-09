# -*- coding: utf-8 -*-
"""
仿真模块

提供轨迹执行和探测统计。
"""

from .trajectory import (
    EmissionResult,
    run_dual_atom_emission,
    apply_qfc_filter_memory_chain,
    sample_fiber_realization,
)
from .detection import (
    DetectionEvent,
    TwoPhotonDetectionResult,
    SuccessEnumerationResult,
    run_detection_pipeline,
    run_detection_self_checks,
    extract_qubit_state,
    compute_fidelity_with_bell,
    compute_pauli_correlators_and_chsh,
)
from ..physics.gates import build_detection_effects_6d

__all__ = [
    # trajectory
    'EmissionResult',
    'run_dual_atom_emission',
    'apply_qfc_filter_memory_chain',
    # apply_* functions
    'sample_fiber_realization',
    # detection (POVM sampling)
    'DetectionEvent',
    'TwoPhotonDetectionResult',
    'SuccessEnumerationResult',
    'run_detection_pipeline',
    'run_detection_self_checks',
    'build_detection_effects_6d',
    'extract_qubit_state',
    'compute_fidelity_with_bell',
    'compute_pauli_correlators_and_chsh',
]
