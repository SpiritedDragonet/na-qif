"""
Single Trajectory Execution

This module implements the "conveyor belt" main loop for time-bin simulation.
Each time-bin is processed sequentially: emission, QFC, loss, Jones, BS, detection.
"""

from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, field
import numpy as np

from ..core.mps import MPSState, create_timebin_mps
from ..config import TimeGrid, EmitParams, QFCParams, FiberParams, DetParams
from ..physics.gates import (
    emission_gate, qfc_gate, bs_gate, jones_gate_from_array, swap_gate
)
from ..physics.channels import (
    loss_channel_1517, detection_channel_two_mode, dephasing_channel_from_rate
)


@dataclass
class TrajectoryResult:
    """
    Result of a single trajectory run.

    Attributes
    ----------
    success : bool
        Whether the trajectory resulted in a success pattern
    rho_atom : np.ndarray
        Atomic density matrix at the end (9x9 for two atoms)
    outcome : Optional[Tuple[int, int, int, int]]
        Detector click pattern (d1_H, d1_V, d2_H, d2_V)
    success_bin : Optional[int]
        Which bin produced the success (None if no success)
    record : List[Tuple[int, int, int, int]]
        Full record of detector outcomes for all bins
    """
    success: bool
    rho_atom: np.ndarray
    outcome: Optional[Tuple[int, int, int, int]] = None
    success_bin: Optional[int] = None
    record: List[Tuple[int, int, int, int]] = field(default_factory=list)


class TrajectoryRunner:
    """
    Executes a single trajectory of the time-bin protocol.

    Implements the "conveyor belt" algorithm where each bin is processed:
    1. Emission (atom -> photon)
    2. QFC (780 -> 1517 conversion)
    3. Jones rotation (polarization)
    4. Loss channel
    5. Beam splitter (A_n with B_n)
    6. Detection

    Chain layout: A0 - B0 - A1 - B1 - A2 - B2 - ...
    where A0, B0 are atoms and A_n, B_n are time-bins.
    """

    def __init__(
        self,
        time_grid: TimeGrid,
        emit_params: EmitParams,
        qfc_params: QFCParams,
        fiber_params: FiberParams,
        det_params: DetParams,
        chi_max: int = 100,
        seed: Optional[int] = None,
    ):
        """
        Initialize trajectory runner.

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
        chi_max : int
            Maximum bond dimension
        seed : int, optional
            Random seed for reproducibility
        """
        self.time_grid = time_grid
        self.emit = emit_params
        self.qfc = qfc_params
        self.fiber = fiber_params
        self.det = det_params
        self.chi_max = chi_max

        # Random number generator
        self.rng = np.random.default_rng(seed)

        # Cached gates (computed once)
        self._cached_gates: Dict[str, np.ndarray] = {}

    def initialize_mps(self) -> MPSState:
        """
        Initialize MPS with atoms in excited state, bins in vacuum.

        Chain: A0(3D) - B0(3D) - A1(18D) - B1(18D) - ...

        Returns
        -------
        MPSState
            Initialized MPS state
        """
        # System: two atoms (3D each)
        system_dim = 9  # atom_A ⊗ atom_B

        # Or use separate atom sites:
        # Chain layout: A0(3) - B0(3) - A1(18) - B1(18) - ...
        local_dims = [3, 3] + [18] * (2 * self.time_grid.N)

        # Initial state: both atoms excited, all bins vacuum
        # Atom basis: |0>, |1>, |e> with |e> at index 2
        init_state = [2, 2] + [0] * (2 * self.time_grid.N)

        mps = MPSState(local_dims, init_state=init_state, max_bond=self.chi_max)
        return mps

    def run_bin(
        self,
        mps: MPSState,
        n: int,
    ) -> Tuple[MPSState, Tuple[int, int, int, int]]:
        """
        Process a single time-bin n.

        Steps:
        1. Emission on A0-An and B0-Bn
        2. QFC on An and Bn
        3. Jones on An and Bn
        4. Loss on An and Bn
        5. BS on An-Bn
        6. Detection on An-Bn
        7. Finalize An-Bn

        Parameters
        ----------
        mps : MPSState
            Current MPS state
        n : int
            Bin index (1-indexed, so n=1 corresponds to sites 2,3 in chain)

        Returns
        -------
        Tuple[MPSState, Tuple[int, int, int, int]]
            Updated MPS and detector outcome (d1_H, d1_V, d2_H, d2_V)
        """
        # Site indices: A0=0, B0=1, A1=2, B1=3, A2=4, B2=4, ...
        # For bin n (1-indexed), sites are at indices 2n and 2n+1
        site_A = 2 * n
        site_B = 2 * n + 1

        t = self.time_grid.t[n-1]  # n is 1-indexed

        # (1) Emission: two-site unitary (atom, bin)
        U_emit_A = emission_gate(
            gamma=self.emit.get_gamma_A(t),
            dt=self.time_grid.dt,
            Alpha=self.emit.Alpha_A,
            which_atom='A'
        )
        # Emission gate acts on atom(3D) ⊗ 780(3D), need to embed to full 18D bin
        # For now, use a simplified version
        mps.apply_bond_op(0, U_emit_A)  # A0-An emission (simplified)

        U_emit_B = emission_gate(
            gamma=self.emit.get_gamma_B(t),
            dt=self.time_grid.dt,
            Alpha=self.emit.Alpha_B,
            which_atom='B'
        )
        mps.apply_bond_op(1, U_emit_B)  # B0-Bn emission (simplified)

        # (2) QFC: one-site unitary on An, Bn
        U_qfc = qfc_gate(theta_H=self.qfc.theta_H, theta_V=self.qfc.theta_V)
        mps.apply_one_site_gate(site_A, U_qfc)
        mps.apply_one_site_gate(site_B, U_qfc)

        # (3) Jones rotation: one-site unitary
        U_pol_A = jones_gate_from_array(self.fiber.Jones_A)
        U_pol_B = jones_gate_from_array(self.fiber.Jones_B)

        # Embed into 18D bin space (only acts on 1517 subspace)
        # For now, apply directly (assumes proper embedding)
        mps.apply_one_site_gate(site_A, U_pol_A)
        mps.apply_one_site_gate(site_B, U_pol_B)

        # (4) Loss: one-site Kraus
        K_loss_A = loss_channel_1517(
            eta_H=self.fiber.eta_fiber_A * self.qfc.eta_ins_H,
            eta_V=self.fiber.eta_fiber_A * self.qfc.eta_ins_V
        )
        mps.apply_kraus_one_site(site_A, K_loss_A, self.rng)

        K_loss_B = loss_channel_1517(
            eta_H=self.fiber.eta_fiber_B * self.qfc.eta_ins_H,
            eta_V=self.fiber.eta_fiber_B * self.qfc.eta_ins_V
        )
        mps.apply_kraus_one_site(site_B, K_loss_B, self.rng)

        # (5) Beam splitter: two-site unitary
        U_bs = bs_gate()
        mps.apply_bond_op(site_A, U_bs)

        # (6) Detection: two-site measurement Kraus
        K_det, outcomes = detection_channel_two_mode(
            eta_det=self.det.eta_det,
            p_dark=self.det.p_dark
        )
        mu = mps.apply_kraus_two_site(site_A, K_det, self.rng)
        outcome = outcomes[mu]

        # (7) Finalize measured bins
        mps.finalize_bin_pair(site_A)

        return mps, outcome

    def run(self) -> TrajectoryResult:
        """
        Run a complete trajectory over all time bins.

        Returns
        -------
        TrajectoryResult
            Result of the trajectory
        """
        mps = self.initialize_mps()
        record = []

        for n in range(1, self.time_grid.N + 1):
            mps, outcome = self.run_bin(mps, n)
            record.append(outcome)

            # Check for success
            if self.det.is_success(outcome):
                # Extract atomic state
                rho_atom = mps.get_reduced_density([0, 1])  # A0, B0

                return TrajectoryResult(
                    success=True,
                    rho_atom=rho_atom,
                    outcome=outcome,
                    success_bin=n,
                    record=record
                )

        # No success in any bin
        rho_atom = mps.get_reduced_density([0, 1])
        return TrajectoryResult(
            success=False,
            rho_atom=rho_atom,
            outcome=None,
            success_bin=None,
            record=record
        )

    def extract_wave_packet(self, mps: MPSState) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract wave packet information from MPS.

        Returns intensity envelope p_n and complex amplitudes xi_n for each bin.

        Parameters
        ----------
        mps : MPSState
            Current MPS state

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (p_n, xi_n) where p_n is intensity and xi_n are complex amplitudes
        """
        p_n = np.zeros(self.time_grid.N)
        xi_n_H = np.zeros(self.time_grid.N, dtype=complex)
        xi_n_V = np.zeros(self.time_grid.N, dtype=complex)

        # Number operators for 1517 telecom subspace
        # Need to construct these operators
        # For now, return zeros as placeholder
        # TODO: Implement using operators.number_op() on 1517 subspace

        return p_n, xi_n_H + xi_n_V


def run_single_trajectory(
    time_grid: TimeGrid,
    emit_params: EmitParams,
    qfc_params: QFCParams,
    fiber_params: FiberParams,
    det_params: DetParams,
    chi_max: int = 100,
    seed: Optional[int] = None,
) -> TrajectoryResult:
    """
    Convenience function to run a single trajectory.

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
    chi_max : int
        Maximum bond dimension
    seed : int, optional
        Random seed

    Returns
    -------
    TrajectoryResult
        Result of the trajectory
    """
    runner = TrajectoryRunner(
        time_grid=time_grid,
        emit_params=emit_params,
        qfc_params=qfc_params,
        fiber_params=fiber_params,
        det_params=det_params,
        chi_max=chi_max,
        seed=seed,
    )
    return runner.run()
