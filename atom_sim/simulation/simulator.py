"""
Multi-Trajectory Statistics

This module provides functions to run multiple trajectories
and compute statistical estimates of success probability, fidelity, etc.
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from scipy.stats import sem

from .trajectory import TrajectoryRunner, TrajectoryResult, run_single_trajectory
from ..config import TimeGrid, EmitParams, QFCParams, FiberParams, DetParams


@dataclass
class SimulationResult:
    """
    Result of running multiple trajectories.

    Attributes
    ----------
    p_succ : float
        Estimated success probability
    p_succ_stderr : float
        Standard error of success probability
    rho_cond : np.ndarray
        Conditional atomic density matrix (averaged over successful trajectories)
    F_cond : float
        Conditional fidelity with target Bell state
    F_cond_stderr : float
        Standard error of fidelity estimate
    n_succ : int
        Number of successful trajectories
    n_traj : int
        Total number of trajectories run
    """
    p_succ: float
    p_succ_stderr: float
    rho_cond: np.ndarray
    F_cond: float
    F_cond_stderr: float
    n_succ: int
    n_traj: int


def run_simulation(
    time_grid: TimeGrid,
    emit_params: EmitParams,
    qfc_params: QFCParams,
    fiber_params: FiberParams,
    det_params: DetParams,
    n_traj: int = 1000,
    chi_max: int = 100,
    seed: Optional[int] = None,
    target_bell: Optional[np.ndarray] = None,
) -> SimulationResult:
    """
    Run multiple trajectories and compute statistics.

    Parameters
    ----------
    time_grid : TimeGrid
        Time discretization
    emit_params : EmitParams
        Emission parameters
    qfc_params : QFCParams
        QFC parameters
    fiber_params : FiberParams
        Fiber/Jones/PMD parameters
    det_params : DetParams
        Detection parameters
    n_traj : int
        Number of trajectories to run
    chi_max : int
        Maximum bond dimension for MPS
    seed : int, optional
        Random seed for reproducibility
    target_bell : np.ndarray, optional
        Target Bell state for fidelity calculation (9x9 matrix).
        If None, uses |Phi+> = (|00> + |11>) / sqrt(2)

    Returns
    -------
    SimulationResult
        Statistical results including p_succ ± stderr, F_cond ± stderr
    """
    # Set default target Bell state: |Phi+> = (|00> + |11>) / sqrt(2)
    # In the |0>, |1>, |e> basis for each atom:
    # |00> means atom A in |0>, atom B in |0>
    # |11> means atom A in |1>, atom B in |1>
    if target_bell is None:
        target_bell = np.zeros((9, 9), dtype=complex)
        # Basis order: |0_A0_B>, |0_A1_B>, |0_Ae_B>, |1_A0_B>, ...
        # |00> is at index 0*3 + 0 = 0
        # |11> is at index 1*3 + 1 = 4
        target_bell[0, 0] = 0.5
        target_bell[4, 4] = 0.5
        target_bell[0, 4] = 0.5
        target_bell[4, 0] = 0.5

    # Track results
    success_count = 0
    rho_success_list = []
    fidelity_list = []

    # Run trajectories
    for i in range(n_traj):
        # Use different seed for each trajectory
        traj_seed = None if seed is None else seed + i

        result = run_single_trajectory(
            time_grid=time_grid,
            emit_params=emit_params,
            qfc_params=qfc_params,
            fiber_params=fiber_params,
            det_params=det_params,
            chi_max=chi_max,
            seed=traj_seed,
        )

        if result.success:
            success_count += 1
            rho_success_list.append(result.rho_atom)

            # Compute fidelity
            fidelity = np.real(np.vdot(target_bell.flatten(),
                                       result.rho_atom @ target_bell.flatten()))
            fidelity_list.append(fidelity)

    # Compute statistics
    p_succ = success_count / n_traj
    p_succ_stderr = np.sqrt(p_succ * (1 - p_succ) / n_traj)

    if success_count > 0:
        rho_cond = np.mean(rho_success_list, axis=0)
        F_cond = np.mean(fidelity_list)
        F_cond_stderr = sem(fidelity_list) if len(fidelity_list) > 1 else 0.0
    else:
        rho_cond = np.zeros((9, 9), dtype=complex)
        F_cond = 0.0
        F_cond_stderr = 0.0

    return SimulationResult(
        p_succ=p_succ,
        p_succ_stderr=p_succ_stderr,
        rho_cond=rho_cond,
        F_cond=F_cond,
        F_cond_stderr=F_cond_stderr,
        n_succ=success_count,
        n_traj=n_traj,
    )


__all__ = [
    'SimulationResult',
    'run_simulation',
]
