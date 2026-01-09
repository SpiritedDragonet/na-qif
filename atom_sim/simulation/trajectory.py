"""
Single Trajectory Execution

This module implements the "conveyor belt" main loop for time-bin simulation.
Each time-bin is processed sequentially: emission, QFC, loss, Jones, BS, detection.
"""

from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, field
import numpy as np

from ..core.mps import MPSState
from ..config import TimeGrid, EmitParams, QFCParams, FiberParams, DetParams
from ..hilbert.basis import BIN_SPACE, SUBSPACE_780, SUBSPACE_1517
from ..physics.gates import (
    emission_gate, qfc_gate, bs_gate, jones_gate_from_array, swap_gate
)
from ..physics.channels import (
    loss_channel_1517, loss_channel_both_subspaces,
    detection_channel_two_mode, detection_povm_single_site,
    dephasing_channel_from_rate
)


# Dimension constants for clarity
DIM_ATOM = 3
DIM_BIN = BIN_SPACE.dim  # 18
DIM_780 = SUBSPACE_780.dim  # 3
DIM_1517 = SUBSPACE_1517.dim  # 6

# Bin subspace indices (780 x 1517 product space)
# index = i_780 * DIM_1517 + i_1517
IDX_780_VAC = 0  # |vac> in 780
IDX_780_H = 1    # |H> in 780
IDX_780_V = 2    # |V> in 780

# 780H block in 18D bin space: indices DIM_1517 * 1 to DIM_1511 * 2 - 1
IDX_BIN_780H_START = DIM_1517 * IDX_780_H  # 6
IDX_BIN_780H_END = DIM_1517 * (IDX_780_H + 1)  # 12

# 780V block in 18D bin space
IDX_BIN_780V_START = DIM_1517 * IDX_780_V  # 12
IDX_BIN_780V_END = DIM_1517 * (IDX_780_V + 1)  # 18


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


@dataclass
class EmissionResult:
    """
    Result of dual-atom emission simulation (emission-only stage).

    Chain layout after emission: A1, B1, A2, B2, ..., AN, BN, atomA, atomB
    (bins at the front, atoms at the end)

    Attributes
    ----------
    mps : MPSState
        Final MPS state after emission.
    time_grid : TimeGrid
        Time grid used for simulation
    per_bin_prob_A : np.ndarray
        Emission probability for each bin in arm A (shape: n_bins)
    per_bin_prob_B : np.ndarray
        Emission probability for each bin in arm B (shape: n_bins)
    atom_states : dict
        Final atomic states {'A': rho_A, 'B': rho_B}
    atom_A_state_evolution : np.ndarray
        Atomic state evolution for atom A (shape: 3 x 2*n_bins)
        Rows: P(|0>), P(|1>), P(|e>)
        Columns: after each SWAP (odd: after A SWAP, even: after B SWAP)
    atom_B_state_evolution : np.ndarray
        Atomic state evolution for atom B (shape: 3 x 2*n_bins)
        Rows: P(|0>), P(|1>), P(|e>)
        Columns: after each SWAP (odd: after A SWAP, even: after B SWAP)
    """
    mps: MPSState
    time_grid: TimeGrid
    per_bin_prob_A: np.ndarray
    per_bin_prob_B: np.ndarray
    atom_states: dict
    atom_A_state_evolution: np.ndarray = field(default_factory=lambda: np.zeros((3, 1)))
    atom_B_state_evolution: np.ndarray = field(default_factory=lambda: np.zeros((3, 1)))

    def get_bin_indices(self, n: int) -> Tuple[int, int]:
        """
        Get the MPS site indices for bin n in arms A and B.

        After SWAP conveyor belt:
        - A1, B1, A2, B2, ..., AN, BN, atomA, atomB
        - A_n is at site 2*n, B_n is at site 2*n + 1

        Parameters
        ----------
        n : int
            Bin index (0-based)

        Returns
        -------
        Tuple[int, int]
            (site_A, site_B) - MPS site indices for A_n and B_n
        """
        n_bins = len(self.per_bin_prob_A)
        # After SWAP: A1(0), B1(1), A2(2), B2(3), ..., AN, BN, atomA, atomB
        # A_n is at site 2*n, B_n is at site 2*n + 1
        return 2 * n, 2 * n + 1

    def get_atom_site_indices(self) -> Tuple[int, int]:
        """
        Get the MPS site indices for atoms A and B.

        After SWAP conveyor belt, atoms are at the end.

        Returns
        -------
        Tuple[int, int]
            (site_A, site_B) - MPS site indices for atomA and atomB
        """
        n_bins = len(self.per_bin_prob_A)
        # Atoms are at sites 2*n_bins and 2*n_bins + 1
        return 2 * n_bins, 2 * n_bins + 1

    def get_n_bins(self) -> int:
        """Get the number of time bins."""
        return len(self.per_bin_prob_A)

    def get_mps_for_next_stage(self) -> MPSState:
        """
        Get the MPS state ready for the next stage (e.g., QFC, BSM).

        The current layout is: A1, B1, A2, B2, ..., AN, BN, atomA, atomB
        where each A_n, B_n pair is adjacent for operations.

        Returns
        -------
        MPSState
            The MPS state ready for next processing stage
        """
        return self.mps


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
            dt=self.time_grid.dt * 1e9,  # Convert seconds to nanoseconds
            Alpha=self.emit.Alpha_A,
            which_atom='A'
        )
        # Emission gate acts on atom(3D) ⊗ 780(3D), need to embed to full 18D bin
        # For now, use a simplified version
        mps.apply_bond_op(0, U_emit_A)  # A0-An emission (simplified)

        U_emit_B = emission_gate(
            gamma=self.emit.get_gamma_B(t),
            dt=self.time_grid.dt * 1e9,  # Convert seconds to nanoseconds
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

    def run_emission(
        self,
        verbose: bool = True,
    ) -> EmissionResult:
        """
        Run emission-only stage using SWAP conveyor belt protocol.

        This implements the correct dual-atom emission where each bin couples
        exactly once with its corresponding atom, preventing re-absorption.

        Chain structure (initial): atomA, atomB, A1, B1, A2, B2, ..., AN, BN
        Chain structure (final):   A1, B1, A2, B2, ..., AN, BN, atomA, atomB

        Parameters
        ----------
        verbose : bool
            Whether to print progress information

        Returns
        -------
        EmissionResult
            Container with emission simulation results
        """
        if verbose:
            print("=" * 70)
            print("Dual-Atom Emission Simulation")
            print("=" * 70)
            print(f"\nParameters:")
            print(f"  n_bins = {self.time_grid.N}, dt = {self.time_grid.dt * 1e9:.1f} ns")

        # Initialize MPS: atomA(3), atomB(3), A1(18), B1(18), ..., AN(18), BN(18)
        local_dims = [DIM_ATOM, DIM_ATOM] + [DIM_BIN, DIM_BIN] * self.time_grid.N
        # Initial state: atoms excited (index 2), bins vacuum (index 0)
        init_state = [2, 2] + [0] * (2 * self.time_grid.N)
        mps = MPSState(local_dims=local_dims, init_state=init_state, max_bond=self.chi_max)

        per_bin_prob_A = np.zeros(self.time_grid.N)
        per_bin_prob_B = np.zeros(self.time_grid.N)

        # Record atomic state evolution after each SWAP
        # Shape: (3, 2 * n_bins) where 3 rows are P(|0>), P(|1>), P(|e>)
        # Columns: record after each atom SWAPs past a bin
        #   - Even indices (0, 2, 4, ...): after atomA SWAP for bin n/2
        #   - Odd indices (1, 3, 5, ...): after atomB SWAP for bin (n-1)/2
        atom_A_state_evolution = np.zeros((3, 2 * self.time_grid.N))
        atom_B_state_evolution = np.zeros((3, 2 * self.time_grid.N))

        # Record initial atomic states
        rho_A_init = mps.get_reduced_density([0])
        rho_B_init = mps.get_reduced_density([1])
        atom_A_state_evolution[0, 0] = rho_A_init[0, 0].real  # P(|0>)
        atom_A_state_evolution[1, 0] = rho_A_init[1, 1].real  # P(|1>)
        atom_A_state_evolution[2, 0] = rho_A_init[2, 2].real  # P(|e>)
        atom_B_state_evolution[0, 0] = rho_B_init[0, 0].real
        atom_B_state_evolution[1, 0] = rho_B_init[1, 1].real
        atom_B_state_evolution[2, 0] = rho_B_init[2, 2].real

        if verbose:
            print(f"\nRunning SWAP conveyor belt...")
            print(f"  Initial: [atomA, atomB, A1, B1, A2, B2, ...]")
            print(f"  Target:  [A1, B1, A2, B2, ..., AN, BN, atomA, atomB]")

        # Process bins one by one
        for n in range(self.time_grid.N):
            t = self.time_grid.t[n]

            # === Atom A emission ===
            atom_sites = mps.find_sites_by_dim(DIM_ATOM)
            site_A = atom_sites[0]
            target_A = 2 + 2 * n  # Original position of A_n

            # Move atomA right until adjacent to target bin
            while site_A + 1 < target_A:
                mps.swap_sites(site_A)
                site_A += 1

            # Apply emission gate
            gamma_A = self.emit.get_gamma_A(t)
            if gamma_A >= 1e-6 and site_A + 1 < len(mps.d):
                U_emit_A = emission_gate(
                    gamma=gamma_A,
                    dt=self.time_grid.dt * 1e9,  # Convert seconds to nanoseconds
                    Alpha=self.emit.Alpha_A,
                    which_atom='A'
                )
                mps.apply_bond_op(site_A, U_emit_A)

                # Extract emission probability for this bin
                rho_A_n = mps.get_reduced_density([site_A + 1])
                p_A_H = rho_A_n[IDX_BIN_780H_START:IDX_BIN_780H_END,
                               IDX_BIN_780H_START:IDX_BIN_780H_END].sum().real
                p_A_V = rho_A_n[IDX_BIN_780V_START:IDX_BIN_780V_END,
                               IDX_BIN_780V_START:IDX_BIN_780V_END].sum().real
                per_bin_prob_A[n] = p_A_H + p_A_V

            # SWAP atomA right (past the processed bin)
            # Allow moving all the way to the end of the chain
            # After SWAP, atom should be at or beyond the original bin position
            if site_A + 1 < len(mps.d):
                # Check if we need to swap past the last bin (for n = N-1)
                # The last bin A_N is at site 2*N, B_N at site 2*N+1
                # We want to move atoms all the way past these bins
                mps.swap_sites(site_A)
                site_A += 1

            # Record atom A state after SWAP
            atom_sites_after_A = mps.find_sites_by_dim(DIM_ATOM)
            site_A_after = atom_sites_after_A[0]
            rho_A_after = mps.get_reduced_density([site_A_after])
            col_idx = 2 * n + 1  # After atomA SWAP for bin n
            atom_A_state_evolution[0, col_idx] = rho_A_after[0, 0].real
            atom_A_state_evolution[1, col_idx] = rho_A_after[1, 1].real
            atom_A_state_evolution[2, col_idx] = rho_A_after[2, 2].real

            # === Atom B emission ===
            atom_sites = mps.find_sites_by_dim(DIM_ATOM)
            site_B = atom_sites[1]
            target_B = 3 + 2 * n  # Original position of B_n

            # Move atomB right until adjacent to target bin
            while site_B + 1 < target_B:
                mps.swap_sites(site_B)
                site_B += 1

            # Apply emission gate
            gamma_B = self.emit.get_gamma_B(t)
            if gamma_B >= 1e-6 and site_B + 1 < len(mps.d):
                U_emit_B = emission_gate(
                    gamma=gamma_B,
                    dt=self.time_grid.dt * 1e9,  # Convert seconds to nanoseconds
                    Alpha=self.emit.Alpha_B,
                    which_atom='B'
                )
                mps.apply_bond_op(site_B, U_emit_B)

                # Extract emission probability for this bin
                rho_B_n = mps.get_reduced_density([site_B + 1])
                p_B_H = rho_B_n[IDX_BIN_780H_START:IDX_BIN_780H_END,
                               IDX_BIN_780H_START:IDX_BIN_780H_END].sum().real
                p_B_V = rho_B_n[IDX_BIN_780V_START:IDX_BIN_780V_END,
                               IDX_BIN_780V_START:IDX_BIN_780V_END].sum().real
                per_bin_prob_B[n] = p_B_H + p_B_V

            # SWAP atomB right
            # Allow moving all the way to the end of the chain
            if site_B + 1 < len(mps.d):
                mps.swap_sites(site_B)
                site_B += 1

            # Record atom B state after SWAP
            atom_sites_after_B = mps.find_sites_by_dim(DIM_ATOM)
            site_B_after = atom_sites_after_B[1]
            rho_B_after = mps.get_reduced_density([site_B_after])
            col_idx = 2 * n + 1  # After atomB SWAP for bin n
            atom_B_state_evolution[0, col_idx] = rho_B_after[0, 0].real
            atom_B_state_evolution[1, col_idx] = rho_B_after[1, 1].real
            atom_B_state_evolution[2, col_idx] = rho_B_after[2, 2].real

            if verbose and (n + 1) % 50 == 0:
                atom_sites_curr = mps.find_sites_by_dim(DIM_ATOM)
                # Print current atomic states
                print(f"  Processed {n + 1}/{self.time_grid.N} bins... "
                      f"(atomA@{atom_sites_curr[0]}, atomB@{atom_sites_curr[1]})")
                print(f"    Atom A: P(|0>)={atom_A_state_evolution[0, col_idx]:.3f}, "
                      f"P(|1>)={atom_A_state_evolution[1, col_idx]:.3f}, "
                      f"P(|e>)={atom_A_state_evolution[2, col_idx]:.3f}")
                print(f"    Atom B: P(|0>)={atom_B_state_evolution[0, col_idx]:.3f}, "
                      f"P(|1>)={atom_B_state_evolution[1, col_idx]:.3f}, "
                      f"P(|e>)={atom_B_state_evolution[2, col_idx]:.3f}")

        # Final pass: move atoms all the way to the end of the chain
        # This ensures bins occupy sites 0 to 2*N-1, atoms at 2*N and 2*N+1
        if verbose:
            print(f"\n  Final pass: moving atoms to end of chain...")

        # Move atomA to site 2*N (second to last)
        while True:
            atom_sites = mps.find_sites_by_dim(DIM_ATOM)
            site_A = atom_sites[0]
            target_A = 2 * self.time_grid.N  # Site 2*N
            if site_A >= target_A:
                break
            mps.swap_sites(site_A)

        # Move atomB to site 2*N+1 (last)
        while True:
            atom_sites = mps.find_sites_by_dim(DIM_ATOM)
            site_B = atom_sites[1]
            target_B = 2 * self.time_grid.N + 1  # Site 2*N+1
            if site_B >= target_B:
                break
            mps.swap_sites(site_B)

        if verbose:
            atom_sites_final = mps.find_sites_by_dim(DIM_ATOM)
            print(f"  After final pass: atomA@{atom_sites_final[0]}, atomB@{atom_sites_final[1]}")

        # Get final atom states
        atom_sites_final = mps.find_sites_by_dim(DIM_ATOM)
        rho_atom_A = mps.get_reduced_density([atom_sites_final[0]])
        rho_atom_B = mps.get_reduced_density([atom_sites_final[1]])

        atom_states = {'A': rho_atom_A, 'B': rho_atom_B}

        if verbose:
            print(f"  Complete!")
            print(f"  Final: atomA@{atom_sites_final[0]}, atomB@{atom_sites_final[1]}")
            print(f"  Final chi: {mps.get_bond_dimensions()}")
            print(f"  Norm: {mps.norm():.6f}")
            print(f"\nFinal atomic states:")
            print(f"  Atom A: P(|e>)={rho_atom_A[2,2].real:.4f}")
            print(f"  Atom B: P(|e>)={rho_atom_B[2,2].real:.4f}")
            print(f"\nEmission statistics:")
            print(f"  Arm A total: {per_bin_prob_A.sum():.4f}, peak: {per_bin_prob_A.max():.4f}")
            print(f"  Arm B total: {per_bin_prob_B.sum():.4f}, peak: {per_bin_prob_B.max():.4f}")

        return EmissionResult(
            mps=mps,
            time_grid=self.time_grid,
            per_bin_prob_A=per_bin_prob_A,
            per_bin_prob_B=per_bin_prob_B,
            atom_states=atom_states,
            atom_A_state_evolution=atom_A_state_evolution,
            atom_B_state_evolution=atom_B_state_evolution,
        )


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


def run_emission_only(
    time_grid: TimeGrid,
    emit_params: EmitParams,
    qfc_params: Optional[QFCParams] = None,
    fiber_params: Optional[FiberParams] = None,
    det_params: Optional[DetParams] = None,
    chi_max: int = 100,
    verbose: bool = True,
) -> EmissionResult:
    """
    Run emission-only simulation using SWAP conveyor belt protocol.

    This is a convenience function for the first stage of the total simulation:
    - Two atoms (A and B) in excited state
    - Emission to time bins (780nm only, no QFC yet)
    - Final state ready for next gate (BSM at station)

    Parameters
    ----------
    time_grid : TimeGrid
        Time discretization
    emit_params : EmitParams
        Emission parameters (gamma_A, gamma_B, Alpha_A, Alpha_B)
    qfc_params : QFCParams, optional
        Not used in emission-only, but kept for interface consistency
    fiber_params : FiberParams, optional
        Not used in emission-only, but kept for interface consistency
    det_params : DetParams, optional
        Not used in emission-only, but kept for interface consistency
    chi_max : int
        Maximum bond dimension
    verbose : bool
        Whether to print progress information

    Returns
    -------
    EmissionResult
        Container with emission simulation results
    """
    # Create default params if not provided
    if qfc_params is None:
        from ..config import QFCParams as _QFCParams
        qfc_params = _QFCParams()
    if fiber_params is None:
        from ..config import FiberParams as _FiberParams
        fiber_params = _FiberParams()
    if det_params is None:
        from ..config import DetParams as _DetParams
        det_params = _DetParams()

    runner = TrajectoryRunner(
        time_grid=time_grid,
        emit_params=emit_params,
        qfc_params=qfc_params,
        fiber_params=fiber_params,
        det_params=det_params,
        chi_max=chi_max,
    )
    return runner.run_emission(verbose=verbose)


# ============================================================================
# Unified Processor Functions (apply_* pattern)
# All functions follow the same interface:
#   - Input: mps (MPSState), params, verbose (bool)
#   - Output: mps (MPSState)
#   - Print format: consistent across all functions
# ============================================================================

def apply_qfc(
    mps: MPSState,
    n_bins: int,
    theta_H: float = np.pi/4,
    theta_V: float = np.pi/4,
    verbose: bool = True,
) -> MPSState:
    """
    Apply QFC gate to all bins.

    Parameters
    ----------
    mps : MPSState
        MPS state (layout: A1, B1, A2, B2, ..., AN, BN, atomA, atomB)
    n_bins : int
        Number of time bins
    theta_H : float
        QFC angle for H polarization (sin² = conversion probability)
    theta_V : float
        QFC angle for V polarization
    verbose : bool
        Whether to print progress

    Returns
    -------
    MPSState
        MPS state with QFC applied (modified in-place)
    """
    from ..physics.gates import qfc_gate

    _print_header("QFC", verbose)
    if verbose:
        print(f"  theta_H = {theta_H:.4f} (sin² = {np.sin(theta_H)**2:.3f})")
        print(f"  theta_V = {theta_V:.4f} (sin² = {np.sin(theta_V)**2:.3f})")

    # Get QFC gate (18x18, acts on single bin)
    U_qfc = qfc_gate(theta_H=theta_H, theta_V=theta_V)

    if verbose:
        print(f"  U_qfc shape: {U_qfc.shape}")
        print(f"  n_bins={n_bins}, MPS L={mps.L}")
        print(f"  MPS d[:5]={mps.d[:5]}, d[-5:]={mps.d[-5:]}")

    # Apply QFC to each bin
    for n in range(n_bins):
        site_A = 2 * n
        site_B = 2 * n + 1

        mps.apply_one_site_gate(site_A, U_qfc)
        mps.apply_one_site_gate(site_B, U_qfc)

        _print_progress(n + 1, n_bins, verbose)

    _print_footer(mps, verbose, stage="QFC")
    return mps


def apply_jones(
    mps: MPSState,
    n_bins: int,
    Jones_A: np.ndarray,
    Jones_B: np.ndarray,
    verbose: bool = True,
) -> MPSState:
    """
    Apply Jones polarization rotation to all bins.

    Parameters
    ----------
    mps : MPSState
        MPS state (layout: A1, B1, A2, B2, ..., AN, BN, atomA, atomB)
    n_bins : int
        Number of time bins
    Jones_A : np.ndarray
        2x2 Jones matrix for arm A
    Jones_B : np.ndarray
        2x2 Jones matrix for arm B
    verbose : bool
        Whether to print progress

    Returns
    -------
    MPSState
        MPS state with Jones rotation applied (modified in-place)
    """
    from ..physics.gates import jones_gate_from_array

    _print_header("Jones", verbose)
    if verbose:
        print(f"  Jones_A: {Jones_A}")
        print(f"  Jones_B: {Jones_B}")

    # Get Jones gates (18x18, embedded)
    U_J_A = jones_gate_from_array(Jones_A)
    U_J_B = jones_gate_from_array(Jones_B)

    # Apply Jones to each bin
    for n in range(n_bins):
        site_A = 2 * n
        site_B = 2 * n + 1
        mps.apply_one_site_gate(site_A, U_J_A)
        mps.apply_one_site_gate(site_B, U_J_B)

        _print_progress(n + 1, n_bins, verbose)

    _print_footer(mps, verbose, stage="Jones")
    return mps


def apply_loss(
    mps: MPSState,
    n_bins: int,
    eta_H_A: float,
    eta_V_A: float,
    eta_H_B: float,
    eta_V_B: float,
    rng: np.random.Generator,
    verbose: bool = True,
) -> MPSState:
    """
    Apply loss channel to all bins.

    Parameters
    ----------
    mps : MPSState
        MPS state (layout: A1, B1, A2, B2, ..., AN, BN, atomA, atomB)
    n_bins : int
        Number of time bins
    eta_H_A, eta_V_A : float
        Transmissivity for arm A (H, V polarizations)
    eta_H_B, eta_V_B : float
        Transmissivity for arm B (H, V polarizations)
    rng : np.random.Generator
        Random number generator for Kraus sampling
    verbose : bool
        Whether to print progress

    Returns
    -------
    MPSState
        MPS state with loss applied (modified in-place)
    """
    _print_header("Loss", verbose)
    if verbose:
        print(f"  Arm A: eta_H={eta_H_A:.3f}, eta_V={eta_V_A:.3f}")
        print(f"  Arm B: eta_H={eta_H_B:.3f}, eta_V={eta_V_B:.3f}")

    # Get loss Kraus operators (18x18, embedded)
    K_loss_A = loss_channel_1517(eta_H_A, eta_V_A)
    K_loss_B = loss_channel_1517(eta_H_B, eta_V_B)

    # Apply loss to each bin
    for n in range(n_bins):
        site_A = 2 * n
        site_B = 2 * n + 1
        mps.apply_kraus_one_site(site_A, K_loss_A, rng)
        mps.apply_kraus_one_site(site_B, K_loss_B, rng)

        _print_progress(n + 1, n_bins, verbose)

    _print_footer(mps, verbose, stage="Loss")
    return mps


def apply_loss_combined(
    mps: MPSState,
    n_bins: int,
    eta_780: float,
    eta_H_1517: float,
    eta_V_1517: float,
    rng: np.random.Generator,
    verbose: bool = True,
) -> MPSState:
    """
    Apply combined loss channel to all bins (both 780 and 1517 subspaces).

    For QFC applications: typically eta_780=0 (100% filtered),
    eta_1517=0.5~0.8 (normal transmission loss).

    Parameters
    ----------
    mps : MPSState
        MPS state (layout: A1, B1, A2, B2, ..., AN, BN, atomA, atomB)
    n_bins : int
        Number of time bins
    eta_780 : float
        Transmissivity for 780nm subspace (0 = 100% loss/filtered)
    eta_H_1517 : float
        Transmissivity for 1517nm H polarization
    eta_V_1517 : float
        Transmissivity for 1517nm V polarization
    rng : np.random.Generator
        Random number generator for Kraus sampling
    verbose : bool
        Whether to print progress

    Returns
    -------
    MPSState
        MPS state with loss applied (modified in-place)
    """
    _print_header("Loss", verbose)
    if verbose:
        print(f"  780nm: eta={eta_780:.3f} ({'100% filtered' if eta_780==0 else 'partial loss'})")
        print(f"  1517nm: eta_H={eta_H_1517:.3f}, eta_V={eta_V_1517:.3f}")

    # Get combined Kraus operators (18x18, both subspaces)
    K_list = loss_channel_both_subspaces(eta_780, eta_H_1517, eta_V_1517)

    # Apply loss to each bin (same for both arms)
    for n in range(n_bins):
        site_A = 2 * n
        site_B = 2 * n + 1
        mps.apply_kraus_one_site(site_A, K_list, rng)
        mps.apply_kraus_one_site(site_B, K_list, rng)

        _print_progress(n + 1, n_bins, verbose)

    _print_footer(mps, verbose, stage="Loss")
    return mps


def apply_bs(
    mps: MPSState,
    n_bins: int,
    verbose: bool = True,
) -> MPSState:
    """
    Apply beam splitter to each A_n, B_n pair.

    Parameters
    ----------
    mps : MPSState
        MPS state (layout: A1, B1, A2, B2, ..., AN, BN, atomA, atomB)
    n_bins : int
        Number of time bins
    verbose : bool
        Whether to print progress

    Returns
    -------
    MPSState
        MPS state with BS applied (modified in-place)
    """
    from ..physics.gates import bs_gate

    _print_header("BS", verbose)

    # Get BS gate (36x36, acts on 1517_A × 1517_B)
    U_bs = bs_gate()

    # Apply BS to each A_n, B_n pair
    for n in range(n_bins):
        site_A = 2 * n
        site_B = 2 * n + 1
        mps.apply_bond_op(site_A, U_bs)

        _print_progress(n + 1, n_bins, verbose)

    _print_footer(mps, verbose, stage="BS")
    return mps


# Helper functions for consistent printing
def _print_header(stage: str, verbose: bool):
    """Print stage header in consistent format."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"{stage:>56} <<<")
        print(f"{'='*60}")

def _print_progress(current: int, total: int, verbose: bool):
    """Print progress in consistent format."""
    if verbose and (current % 50 == 0 or current == total):
        print(f"  Processed {current}/{total} bins...")

def _print_footer(mps: MPSState, verbose: bool, stage: str = ""):
    """Print stage footer in consistent format."""
    if verbose:
        print(f"  Final chi: {mps.get_bond_dimensions()}")
        print(f"{stage} complete.")


def apply_fiber_channel(
    mps: MPSState,
    n_bins: int,
    fiber_params,
    rng: np.random.Generator,
    verbose: bool = True,
) -> tuple:
    """
    Apply fiber channel effects: Jones rotation + loss (with random sampling).

    This combines apply_jones and apply_loss_combined, but samples parameters
    from FiberChannelParams for each trajectory (simulating fiber drift).

    Parameters
    ----------
    mps : MPSState
        MPS state (layout: A1, B1, A2, B2, ..., AN, BN, atomA, atomB)
    n_bins : int
        Number of time bins
    fiber_params : FiberChannelParams
        Fiber channel parameters (will sample new Jones matrices and eta)
    rng : np.random.Generator
        Random number generator
    verbose : bool
        Whether to print progress

    Returns
    -------
    tuple
        (mps, sampled_params) where sampled_params = (U_A, U_B, eta, phase)
    """
    from ..physics.channels import FiberChannelParams

    _print_header("Fiber Channel", verbose)

    # Sample parameters for this trajectory
    U_A, U_B, eta, phase = fiber_params.sample_all(rng)

    if verbose:
        print(f"  Sampled Jones_A:\n{U_A}")
        print(f"  Sampled Jones_B:\n{U_B}")
        print(f"  Phase drift: {phase:.4f} rad")
        print(f"  Sampled eta: {eta:.4f}")

    # Apply Jones rotation
    from ..physics.gates import jones_gate_from_array
    U_J_A = jones_gate_from_array(U_A)
    U_J_B = jones_gate_from_array(U_B)

    for n in range(n_bins):
        site_A = 2 * n
        site_B = 2 * n + 1
        mps.apply_one_site_gate(site_A, U_J_A)
        mps.apply_one_site_gate(site_B, U_J_B)

        _print_progress(n + 1, n_bins, verbose)

    # Apply loss (780 filtered, 1517 with sampled eta)
    from ..physics.channels import loss_channel_both_subspaces
    K_list = loss_channel_both_subspaces(eta_780=0.0, eta_H_1517=eta, eta_V_1517=eta)

    for n in range(n_bins):
        site_A = 2 * n
        site_B = 2 * n + 1
        mps.apply_kraus_one_site(site_A, K_list, rng)
        mps.apply_kraus_one_site(site_B, K_list, rng)

        _print_progress(n + 1, n_bins, verbose)

    _print_footer(mps, verbose, stage="Fiber Channel")

    return mps, (U_A, U_B, eta, phase)


def apply_detection(
    mps: MPSState,
    n_bins: int,
    eta_det: float = 1.0,
    p_dark: float = 0.0,
    rng: np.random.Generator = None,
    verbose: bool = True,
) -> Tuple[MPSState, List[Tuple[int, int, int, int]]]:
    """
    Apply detection POVM to all bin pairs after beam splitter.

    This measures photons at each (A_n, B_n) pair and returns click patterns.
    Each site has H and V detectors, giving 4 outcomes per pair (16 total combinations).

    Parameters
    ----------
    mps : MPSState
        MPS state (layout: A1, B1, A2, B2, ..., AN, BN, atomA, atomB)
    n_bins : int
        Number of time bins
    eta_det : float
        Detection efficiency (0 <= eta_det <= 1)
    p_dark : float
        Dark count probability per detector
    rng : np.random.Generator
        Random number generator for measurement sampling
    verbose : bool
        Whether to print progress

    Returns
    -------
    Tuple[MPSState, List[Tuple[int, int, int, int]]]
        (mps, outcomes) where outcomes[n] = (dA_H, dA_V, dB_H, dB_V)
        for bin n, and d=0 means no click, d=1 means click.

    Notes
    -----
    For BSM (Bell State Measurement), success patterns are:
        - (1,0,0,1) or (0,1,1,0): Psi+ heralding (one H, one V in different ports)
        - (0,1,0,1) or (1,0,1,0): Psi- heralding (same polarization in different ports)

    The measurement is destructive: after detection, the photon state collapses.
    """
    if rng is None:
        rng = np.random.default_rng()

    _print_header("Detection", verbose)
    if verbose:
        print(f"  eta_det = {eta_det:.3f}, p_dark = {p_dark:.6f}")

    # Get single-site detection POVM (4 outcomes per site)
    M_single, outcomes_single = detection_povm_single_site(eta_det, p_dark)
    # M_single[i] is 18x18, outcomes_single[i] is (d_H, d_V)

    all_outcomes = []

    for n in range(n_bins):
        site_A = 2 * n
        site_B = 2 * n + 1

        # Apply detection to site A, get outcome index
        mu_A = mps.apply_kraus_one_site(site_A, M_single, rng)
        dA_H, dA_V = outcomes_single[mu_A]

        # Apply detection to site B, get outcome index
        mu_B = mps.apply_kraus_one_site(site_B, M_single, rng)
        dB_H, dB_V = outcomes_single[mu_B]

        outcome = (dA_H, dA_V, dB_H, dB_V)
        all_outcomes.append(outcome)

        _print_progress(n + 1, n_bins, verbose)

    if verbose:
        # Count success patterns
        psi_plus = [(1,0,0,1), (0,1,1,0)]  # H-V or V-H in different ports
        psi_minus = [(0,1,0,1), (1,0,1,0)]  # Same pol in different ports (with sign)

        n_psi_plus = sum(1 for o in all_outcomes if o in psi_plus)
        n_psi_minus = sum(1 for o in all_outcomes if o in psi_minus)
        n_double_click = sum(1 for o in all_outcomes if sum(o) >= 2)

        print(f"  Results summary:")
        print(f"    Psi+ heralding: {n_psi_plus} bins")
        print(f"    Psi- heralding: {n_psi_minus} bins")
        print(f"    Multi-click: {n_double_click} bins")

    _print_footer(mps, verbose, stage="Detection")

    return mps, all_outcomes


def find_bsm_success(
    outcomes: List[Tuple[int, int, int, int]]
) -> Tuple[bool, int, str]:
    """
    Check if any bin has a BSM success pattern.

    Parameters
    ----------
    outcomes : List[Tuple[int, int, int, int]]
        Detection outcomes for all bins, each is (dA_H, dA_V, dB_H, dB_V)

    Returns
    -------
    Tuple[bool, int, str]
        (success, bin_index, bell_state) where:
        - success: True if BSM heralding found
        - bin_index: which bin (0-indexed), or -1 if no success
        - bell_state: "Psi+" or "Psi-" or ""
    """
    # BSM success patterns (single photon in each arm, different detectors)
    psi_plus_patterns = [(1,0,0,1), (0,1,1,0)]  # H_A V_B or V_A H_B
    psi_minus_patterns = [(1,0,1,0), (0,1,0,1)]  # H_A H_B or V_A V_B (with phase)

    for n, outcome in enumerate(outcomes):
        if outcome in psi_plus_patterns:
            return True, n, "Psi+"
        if outcome in psi_minus_patterns:
            return True, n, "Psi-"

    return False, -1, ""
