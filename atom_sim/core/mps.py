"""
MPS State Container (TeNPy Backend)

================================================================================
Wrapper around TeNPy's MPS for time-bin quantum simulation.

Key Design:
-----------
- Two-site operations use get_theta + set_svd_theta for local updates
- Kraus operations are fully localized (no canonical_form() sweeps)
- Reduced density matrices use get_rho_segment() for correctness

Requirements:
-------------
pip install physics-tenpy

References:
-----------
[1] TeNPy MPS: https://tenpy.readthedocs.io/en/stable/reference/tenpy.networks.mps.MPS.html
[2] TeNPy Array: https://tenpy.readthedocs.io/en/stable/reference/tenpy.linalg.np_conserved.html
================================================================================
"""

from typing import List, Optional, Union
import numpy as np

# TeNPy imports
from tenpy.networks.mps import MPS as TeNPy_MPS
from tenpy.networks.site import BosonSite
from tenpy.linalg.np_conserved import Array


class MPSState:
    """
    Matrix Product State using TeNPy.

    Parameters
    ----------
    local_dims : List[int]
        Local Hilbert space dimensions
    init_state : Optional[Union[List[int], np.ndarray]]
        - None: vacuum |0>...|0>
        - List[int]: product state
        - np.ndarray: full wavefunction (uses MPS.from_full)
    max_bond : int
        Maximum bond dimension for truncation
    """

    def __init__(
        self,
        local_dims: List[int],
        init_state: Optional[Union[List[int], np.ndarray]] = None,
        max_bond: int = 100,
    ):
        self.L = len(local_dims)
        self.d = local_dims
        self.max_bond = max_bond

        # Create TeNPy sites (bosonic, no charge conservation)
        sites = [BosonSite(dim - 1, None) for dim in self.d]

        # Initialize MPS based on init_state type
        if init_state is None:
            # Vacuum state |0>...|0>
            init_labels = ['0'] * self.L
            self._mps = TeNPy_MPS.from_product_state(sites, init_labels, bc='finite', form='B')
        elif isinstance(init_state, list):
            # Product state from basis indices
            init_labels = [str(s) for s in init_state]
            self._mps = TeNPy_MPS.from_product_state(sites, init_labels, bc='finite', form='B')
        elif isinstance(init_state, np.ndarray):
            # Full wavefunction - use TeNPy's from_full
            psi_reshaped = init_state.reshape(self.d + [1] * (self.L - len(self.d)))
            psi_array = Array.from_ndarray_trivial(psi_reshaped, labels=[f'p{i}' for i in range(self.L)])
            self._mps = TeNPy_MPS.from_full(psi_array, sites, bc='finite', form='B')
        else:
            raise ValueError(f"Invalid init_state: {type(init_state)}")

        self._mps.chi_max = self.max_bond

    # ========================================================================
    # Low-level Local Update (avoid canonical_form sweeps)
    # ========================================================================

    def _apply_two_site_op_local(
        self,
        i: int,
        op: Array,
        truncate: bool = True,
        normalize: bool = False,
    ) -> None:
        """
        Apply two-site operator using local update (get_theta + set_svd_theta).

        Avoids canonical_form() sweeps by only updating the local bond.

        Parameters
        ----------
        i : int
            Left site index
        op : Array
            TeNPy Array with labels ['p0', 'p1', 'p0*', 'p1*']
        truncate : bool
            Whether to truncate bond dimension
        normalize : bool
            If True, normalize after application (for Kraus results)
        """
        # Get theta for two sites: legs are (vL, p0, p1, vR)
        theta = self._mps.get_theta(i, n=2)

        # Convert to numpy for contraction (avoid LegCharge issues)
        theta_np = theta.to_ndarray()  # Shape: (chiL, d0, d1, chiR)
        op_np = op.to_ndarray()  # Shape: (d0, d1, d0, d1)

        # Contract: op @ theta where op[i,j,k,l] acts on theta's physical legs
        # Result: theta_new[a, i, j, b] = sum_{k,l} op[i, j, k, l] * theta[a, k, l, b]
        theta_new_np = np.einsum('ijkl,aklb->aijb', op_np, theta_np)

        # Convert back to TeNPy Array
        theta_new = Array.from_ndarray_trivial(theta_new_np, labels=['vL', 'p0', 'p1', 'vR'])

        # Combine legs for SVD: (vL.p0) and (p1.vR)
        theta_combined = theta_new.combine_legs(
            [['vL', 'p0'], ['p1', 'vR']],
            new_axes=[0, 1],
            qconj=[+1, -1]
        )

        # Set truncation parameters
        trunc_params = None
        if truncate:
            trunc_params = {'chi_max': self.max_bond, 'svd_min': 1e-13}

        # Write back via SVD
        self._mps.set_svd_theta(i, theta_combined, trunc_par=trunc_params)

        # Normalize if requested (for Kraus branch results)
        if normalize:
            self._mps.norm = 1.0

    # ========================================================================
    # Gate Operations
    # ========================================================================

    def apply_one_site_gate(self, site: int, gate: np.ndarray) -> None:
        """
        Apply single-site unitary gate.

        Parameters
        ----------
        site : int
            Site index
        gate : np.ndarray
            Unitary matrix of shape (d, d)
        """
        self._mps.apply_local_op(site, gate, unitary=True)

    def apply_two_site_gate(
        self,
        site_left: int,
        gate: np.ndarray,
        truncate: bool = True,
    ) -> None:
        """
        Apply two-site unitary gate using local update.

        Parameters
        ----------
        site_left : int
            Left site index
        gate : np.ndarray
            Unitary matrix of shape (d1*d2, d1*d2) or (d1, d2, d1, d2)
        truncate : bool
            Whether to truncate bond dimension
        """
        d1, d2 = self.d[site_left], self.d[site_left + 1]

        # Reshape gate to 4D: (d1, d2, d1, d2)
        gate = np.asarray(gate)
        if gate.ndim == 2:
            gate = gate.reshape(d1 * d2, d1 * d2)
        gate_4d = gate.reshape(d1, d2, d1, d2)

        # Create TeNPy Array with proper labels
        op_arr = Array.from_ndarray_trivial(gate_4d, labels=['p0', 'p1', 'p0*', 'p1*'])

        # Apply via local update (avoids canonical_form sweep)
        self._apply_two_site_op_local(site_left, op_arr, truncate=truncate, normalize=False)

    def apply_two_site_kraus(
        self,
        site_left: int,
        kraus_ops: List[np.ndarray],
        rng: Optional[np.random.Generator] = None,
    ) -> int:
        """
        Apply Kraus channel via quantum trajectory (Monte Carlo sampling).

        For each Kraus operator K_mu:
        1. Compute p_mu = ||K_mu @ theta||^2
        2. Sample mu according to probabilities {p_mu}
        3. Apply K_mu and normalize by sqrt(p_mu)

        This is fully local and does not trigger canonical_form().

        Parameters
        ----------
        site_left : int
            Left site index
        kraus_ops : List[np.ndarray]
            List of Kraus operators, each shape (d1*d2, d1*d2) or (d1, d2, d1, d2)
        rng : np.random.Generator, optional
            Random number generator

        Returns
        -------
        int
            Index of sampled Kraus operator
        """
        if rng is None:
            rng = np.random.default_rng()

        d1, d2 = self.d[site_left], self.d[site_left + 1]

        # Get current theta
        theta = self._mps.get_theta(site_left, n=2)
        theta_np = theta.to_ndarray()  # Shape: (chiL, d1, d2, chiR)

        # Compute probabilities and resulting states for each Kraus operator
        probs = []
        thetas_mu = []

        for K in kraus_ops:
            K = np.asarray(K)
            if K.ndim == 2:
                K = K.reshape(d1 * d2, d1 * d2)
            K_4d = K.reshape(d1, d2, d1, d2)

            # Apply K: contract K's input legs with theta's physical legs
            K_theta = np.einsum('ijkl,aklb->aijb', K_4d, theta_np)
            p_mu = np.linalg.norm(K_theta) ** 2
            probs.append(p_mu)
            thetas_mu.append(K_theta)

        # Normalize and sample
        probs = np.array(probs)
        p_total = np.sum(probs)

        if p_total < 1e-15:
            raise ValueError("Total probability is near zero - Kraus ops may be invalid")

        probs = probs / p_total
        mu = rng.choice(len(kraus_ops), p=probs)

        # Create normalized theta from selected branch
        theta_selected = thetas_mu[mu] / np.sqrt(probs[mu] * p_total)

        # Convert to TeNPy Array and write back
        theta_arr = Array.from_ndarray_trivial(theta_selected, labels=['vL', 'p0', 'p1', 'vR'])
        theta_combined = theta_arr.combine_legs(
            [['vL', 'p0'], ['p1', 'vR']],
            new_axes=[0, 1],
            qconj=[+1, -1]
        )

        trunc_params = {'chi_max': self.max_bond, 'svd_min': 1e-13}
        self._mps.set_svd_theta(site_left, theta_combined, trunc_par=trunc_params)
        self._mps.norm = 1.0

        return mu

    def swap_sites(self, i: int) -> None:
        """
        Swap adjacent sites i and i+1.

        Parameters
        ----------
        i : int
            Left site index (swaps i and i+1)
        """
        trunc_params = {'chi_max': self.max_bond, 'svd_min': 1e-13}
        self._mps.swap_sites(i, trunc_par=trunc_params)

    # ========================================================================
    # State Extraction
    # ========================================================================

    def get_reduced_density(self, sites: List[int]) -> np.ndarray:
        """
        Get reduced density matrix for specified sites.

        Uses TeNPy's get_rho_segment() which correctly handles
        Schmidt weights and gauge conditions.
        """
        rho_array = self._mps.get_rho_segment(sites)
        return rho_array.to_ndarray()

    def get_atom_state(self, system_site: int = 0) -> np.ndarray:
        """Get atomic density matrix from system site."""
        return self.get_reduced_density([system_site])

    def expectation_value(self, observable: np.ndarray, site: int) -> float:
        """
        Compute expectation value of an observable on a site.

        Parameters
        ----------
        observable : np.ndarray
            Observable matrix (d, d)
        site : int
            Site index

        Returns
        -------
        float
            Expectation value <O>
        """
        rho = self.get_reduced_density([site])
        return float(np.real(np.trace(observable @ rho)))

    # ========================================================================
    # Properties and Utility Methods
    # ========================================================================

    @property
    def chi(self) -> List[int]:
        """Bond dimensions."""
        return self._mps.chi.copy()

    def norm(self) -> float:
        """Get state norm."""
        return float(self._mps.norm)

    def get_bond_dimensions(self) -> List[int]:
        """Get bond dimensions."""
        return self._mps.chi.copy()

    def test_sanity(self) -> bool:
        """Run TeNPy's sanity check (verifies canonical form)."""
        return self._mps.test_sanity()

    def copy(self) -> 'MPSState':
        """Create a deep copy."""
        new_state = MPSState(self.d.copy(), max_bond=self.max_bond)
        new_state._mps = self._mps.copy()
        return new_state

    def __repr__(self) -> str:
        """String representation."""
        chi_str = str(self.get_bond_dimensions())
        return f"MPSState(L={self.L}, d={self.d}, chi={chi_str})"


# ========================================================================
# Factory Functions
# ========================================================================

def create_timebin_mps(
    n_bins: int,
    system_dim: int = 9,
    bin_dim: int = 5,
    max_bond: int = 100,
) -> MPSState:
    """
    Create MPS for time-bin simulation.

    Structure: S - A1 - B1 - A2 - B2 - ... (system, then atom-photon bins)

    Parameters
    ----------
    n_bins : int
        Number of time bins
    system_dim : int
        System (atom) Hilbert space dimension
    bin_dim : int
        Bin (photon) Hilbert space dimension
    max_bond : int
        Maximum bond dimension
    """
    local_dims = [system_dim] + [bin_dim] * (2 * n_bins)
    return MPSState(local_dims, max_bond=max_bond)


def create_excited_atom_mps(
    n_bins: int,
    system_dim: int = 9,
    bin_dim: int = 5,
    max_bond: int = 100,
) -> MPSState:
    """
    Create MPS with atoms in excited state.

    Parameters
    ----------
    n_bins : int
        Number of time bins
    system_dim : int
        System (atom) Hilbert space dimension
    bin_dim : int
        Bin (photon) Hilbert space dimension
    max_bond : int
        Maximum bond dimension
    """
    local_dims = [system_dim] + [bin_dim] * (2 * n_bins)
    # Init with system in state |2> (excited), all bins in |0> (vacuum)
    init_state = [2] + [0] * (2 * n_bins)
    return MPSState(local_dims, init_state=init_state, max_bond=max_bond)
