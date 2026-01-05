"""
Parameter Configuration Classes

This module provides dataclasses for all physical parameters
used in the time-bin MPS simulation.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, Tuple, List, Optional
import numpy as np


@dataclass
class TimeGrid:
    """
    Time discretization parameters.

    Attributes
    ----------
    dt : float
        Time bin width (seconds)
    N : int
        Number of time bins
    """
    dt: float
    N: int

    @property
    def t(self) -> np.ndarray:
        """Array of time bin centers: t[n] = n * dt."""
        return np.arange(self.N) * self.dt

    @property
    def total_time(self) -> float:
        """Total time duration: N * dt."""
        return self.N * self.dt


@dataclass
class EmitParams:
    """
    Emission gate parameters.

    Attributes
    ----------
    gamma_A : float or Callable
        Emission rate for atom A (constant or function of time)
    gamma_B : float or Callable
        Emission rate for atom B (constant or function of time)
    Alpha_A : np.ndarray
        2x2 polarization mapping matrix for atom A
        [[alpha_H+, alpha_H-], [alpha_V+, alpha_V-]]
    Alpha_B : np.ndarray
        2x2 polarization mapping matrix for atom B
    phi_A : float
        Overall phase for atom A emission
    phi_B : float
        Overall phase for atom B emission
    """
    gamma_A: float = 0.1
    gamma_B: float = 0.1
    Alpha_A: np.ndarray = field(default_factory=lambda: np.eye(2))
    Alpha_B: np.ndarray = field(default_factory=lambda: np.eye(2))
    phi_A: float = 0.0
    phi_B: float = 0.0

    def get_gamma_A(self, t: float) -> float:
        """Get emission rate for atom A at time t."""
        if callable(self.gamma_A):
            return self.gamma_A(t)
        return float(self.gamma_A)

    def get_gamma_B(self, t: float) -> float:
        """Get emission rate for atom B at time t."""
        if callable(self.gamma_B):
            return self.gamma_B(t)
        return float(self.gamma_B)


@dataclass
class QFCParams:
    """
    Quantum Frequency Conversion parameters.

    Attributes
    ----------
    theta_H : float
        Conversion angle for H polarization (sin²(theta) = conversion prob)
    theta_V : float
        Conversion angle for V polarization
    eta_ins_H : float
        Insertion loss for H polarization
    eta_ins_V : float
        Insertion loss for V polarization
    """
    theta_H: float = 0.0
    theta_V: float = 0.0
    eta_ins_H: float = 1.0
    eta_ins_V: float = 1.0


@dataclass
class FiberParams:
    """
    Fiber/optical channel parameters.

    Attributes
    ----------
    eta_fiber_A : float
        Fiber transmissivity for arm A
    eta_fiber_B : float
        Fiber transmissivity for arm B
    Jones_A : np.ndarray
        2x2 Jones matrix for arm A
    Jones_B : np.ndarray
        2x2 Jones matrix for arm B
    PMD_A : float
        PMD delay for arm A (seconds)
    PMD_B : float
        PMD delay for arm B (seconds)
    Rin_A : np.ndarray
        PSP input rotation for arm A
    Rout_A : np.ndarray
        PSP output rotation for arm A
    Rin_B : np.ndarray
        PSP input rotation for arm B
    Rout_B : np.ndarray
        PSP output rotation for arm B
    delta_bins : int
        Relative bin delay between arms (B relative to A)
    """
    eta_fiber_A: float = 1.0
    eta_fiber_B: float = 1.0
    Jones_A: np.ndarray = field(default_factory=lambda: np.eye(2))
    Jones_B: np.ndarray = field(default_factory=lambda: np.eye(2))
    PMD_A: float = 0.0
    PMD_B: float = 0.0
    Rin_A: np.ndarray = field(default_factory=lambda: np.eye(2))
    Rout_A: np.ndarray = field(default_factory=lambda: np.eye(2))
    Rin_B: np.ndarray = field(default_factory=lambda: np.eye(2))
    Rout_B: np.ndarray = field(default_factory=lambda: np.eye(2))
    delta_bins: int = 0


@dataclass
class DetParams:
    """
    Detection parameters.

    Attributes
    ----------
    eta_det : float
        Detection efficiency
    p_dark : float
        Dark count probability per detector per bin
    success_patterns : List[Tuple[int, int, int, int]]
        List of detector click patterns that count as success.
        Each tuple is (d1_H, d1_V, d2_H, d2_V).
    pattern_to_bell : Dict[Tuple[int, int, int, int], str]
        Maps each success pattern to the Bell state it projects onto.
        Values: 'phi_plus', 'phi_minus', 'psi_plus', 'psi_minus'
    pattern_to_correction : Dict[Tuple[int, int, int, int], str]
        Maps each success pattern to the required Pauli correction.
        Values: 'I', 'X', 'Y', 'Z'

    Examples
    --------
    >>> # Partial BSM: success on (1H,2V) or (1V,2H) clicks
    >>> params = DetParams(
    ...     success_patterns=[(1,0,0,1), (0,1,1,0)],
    ...     pattern_to_bell={(1,0,0,1): 'psi_minus', (0,1,1,0): 'psi_plus'},
    ...     pattern_to_correction={(1,0,0,1): 'I', (0,1,1,0): 'X'}
    ... )
    """
    eta_det: float = 1.0
    p_dark: float = 0.0
    success_patterns: List[Tuple[int, int, int, int]] = field(default_factory=list)
    pattern_to_bell: Dict[Tuple[int, int, int, int], str] = field(default_factory=dict)
    pattern_to_correction: Dict[Tuple[int, int, int, int], str] = field(default_factory=dict)

    def is_success(self, pattern: Tuple[int, int, int, int]) -> bool:
        """Check if a detector pattern is a success."""
        return pattern in self.success_patterns

    def get_bell_state(self, pattern: Tuple[int, int, int, int]) -> Optional[str]:
        """Get the Bell state for a success pattern."""
        return self.pattern_to_bell.get(pattern)

    def get_correction(self, pattern: Tuple[int, int, int, int]) -> Optional[str]:
        """Get the Pauli correction for a success pattern."""
        return self.pattern_to_correction.get(pattern)


@dataclass
class SimParams:
    """
    Overall simulation parameters.

    Attributes
    ----------
    n_traj : int
        Number of trajectories to run
    chi_max : int
        Maximum bond dimension for MPS
    svd_min : float
        SVD cutoff threshold
    seed : Optional[int]
        Random seed for reproducibility
    """
    n_traj: int = 1000
    chi_max: int = 100
    svd_min: float = 1e-13
    seed: Optional[int] = None
