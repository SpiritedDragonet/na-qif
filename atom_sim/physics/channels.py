"""
Kraus Channels for Non-Unitary Evolution

This module provides Kraus operator sets for various quantum channels:
- Amplitude damping (fiber loss)
- Detection POVM (on/off measurement)
- Atomic dephasing
"""

from typing import List, Tuple
import numpy as np

from ..hilbert.basis import SUBSPACE_1517, SUBSPACE_780, BIN_SPACE


def loss_channel_both_subspaces(
    eta_780: float,
    eta_H_1517: float,
    eta_V_1517: float
) -> List[np.ndarray]:
    """
    Combined loss channel acting on both 780 and 1517 subspaces.

    For QFC applications: 780nm channel typically has eta_780=0 (100% filtered),
    while 1517nm channel has normal transmission loss.

    The Kraus operators are tensor products: K_780^(k) ⊗ K_1517^(j)

    Parameters
    ----------
    eta_780 : float
        Transmissivity for 780nm subspace (0 = 100% loss/filtered)
    eta_H_1517 : float
        Transmissivity for 1517nm H polarization
    eta_V_1517 : float
        Transmissivity for 1517nm V polarization

    Returns
    -------
    List[np.ndarray]
        List of Kraus operators acting on 18D bin space
    """
    # Get Kraus operators for each subspace (NOT embedded)
    K_780_list = loss_channel_780_general(eta_780)  # 3x3 matrices
    K_1517_list = _loss_channel_1517_raw(eta_H_1517, eta_V_1517)  # 6x6 matrices

    # Form all tensor product combinations
    K_combined = []
    for K_780 in K_780_list:
        for K_1517 in K_1517_list:
            # K_780 is (3,3), K_1517 is (6,6), result is (18,18)
            K_combined.append(np.kron(K_780, K_1517))

    return K_combined


def _loss_channel_1517_raw(eta_H: float, eta_V: float) -> List[np.ndarray]:
    """
    Raw 1517nm loss channel (6x6 matrices, NOT embedded into 18D).

    This is an internal function used by loss_channel_both_subspaces.

    Parameters
    ----------
    eta_H : float
        Transmissivity for H polarization
    eta_V : float
        Transmissivity for V polarization

    Returns
    -------
    List[np.ndarray]
        List of 6x6 Kraus operators acting on 1517 subspace only
    """
    K_list_1517 = []

    basis = [
        (0, 0),  # 0: vac
        (1, 0),  # 1: H
        (0, 1),  # 2: V
        (2, 0),  # 3: 2H
        (0, 2),  # 4: 2V
        (1, 1),  # 5: HV
    ]

    for kH in range(3):
        for kV in range(3):
            K = np.zeros((6, 6), dtype=complex)

            for i, (nH, nV) in enumerate(basis):
                if nH < kH or nV < kV:
                    continue

                nH_new = nH - kH
                nV_new = nV - kV

                target = (nH_new, nV_new)
                if target in basis:
                    j = basis.index(target)

                    from math import comb
                    coeff_H = np.sqrt(comb(nH, kH)) * (eta_H ** ((nH - kH) / 2)) * ((1 - eta_H) ** (kH / 2))
                    coeff_V = np.sqrt(comb(nV, kV)) * (eta_V ** ((nV - kV) / 2)) * ((1 - eta_V) ** (kV / 2))
                    K[j, i] = coeff_H * coeff_V

            K_list_1517.append(K)

    # Remove all-zero operators
    K_list_1517 = [K for K in K_list_1517 if np.any(K != 0)]
    return K_list_1517


def loss_channel_780_general(eta: float) -> List[np.ndarray]:
    """
    General loss channel for 780nm subspace (up to 1 photon per mode).

    For eta=0: 100% loss (3 Kraus operators: |vac><vac|, |vac><H|, |vac><V|)
    For eta=1: no loss (K_0 = I)

    Parameters
    ----------
    eta : float
        Transmissivity (0 <= eta <= 1)

    Returns
    -------
    List[np.ndarray]
        Kraus operators for 780 subspace (3x3 matrices)
    """
    # Basis: |vac>, |H>, |V>

    K_list = []

    if eta == 0.0:
        # 100% loss: 3 Kraus operators
        # K_0 = |vac><vac| (vacuum stays vacuum)
        K0 = np.zeros((3, 3), dtype=complex)
        K0[0, 0] = 1.0
        K_list.append(K0)

        # K_1 = |vac><H| (H photon lost -> vacuum)
        K1 = np.zeros((3, 3), dtype=complex)
        K1[0, 1] = 1.0
        K_list.append(K1)

        # K_2 = |vac><V| (V photon lost -> vacuum)
        K2 = np.zeros((3, 3), dtype=complex)
        K2[0, 2] = 1.0
        K_list.append(K2)
    elif eta == 1.0:
        # No loss: identity
        K_list.append(np.eye(3, dtype=complex))
    else:
        # Partial loss: K_0 (no loss) and K_H, K_V (loss per mode)
        K0 = np.zeros((3, 3), dtype=complex)
        K0[0, 0] = 1.0
        K0[1, 1] = np.sqrt(eta)
        K0[2, 2] = np.sqrt(eta)
        K_list.append(K0)

        # Loss operators for each mode
        loss_amp = np.sqrt(1 - eta)

        K_H = np.zeros((3, 3), dtype=complex)
        K_H[0, 1] = loss_amp
        K_list.append(K_H)

        K_V = np.zeros((3, 3), dtype=complex)
        K_V[0, 2] = loss_amp
        K_list.append(K_V)

    return K_list


def loss_channel_1517(eta_H: float, eta_V: float) -> List[np.ndarray]:
    """
    Amplitude damping for the 1517nm telecom subspace (6D),
    embedded in the 18D bin space (I_780 ⊗ K_1517).

    Handles two polarization modes independently with possibly different loss.

    Parameters
    ----------
    eta_H : float
        Transmissivity for H polarization
    eta_V : float
        Transmissivity for V polarization

    Returns
    -------
    List[np.ndarray]
        List of Kraus operators acting on 18D bin space (780 × 1517)
    """
    # 1517 basis: vac, H, V, 2H, 2V, HV
    # Occupancy: (0,0), (1,0), (0,1), (2,0), (0,2), (1,1)

    # We need to construct Kraus operators for all combinations of loss
    # on H and V modes. For small truncation, enumerate explicitly.

    K_list_1517 = []

    # Basis with occupancy tuples
    basis = [
        (0, 0),  # 0: vac
        (1, 0),  # 1: H
        (0, 1),  # 2: V
        (2, 0),  # 3: 2H
        (0, 2),  # 4: 2V
        (1, 1),  # 5: HV
    ]

    # For each possible loss outcome (kH photons lost from H, kV from V)
    for kH in range(3):  # Can lose 0, 1, or 2 H photons
        for kV in range(3):
            K = np.zeros((6, 6), dtype=complex)

            for i, (nH, nV) in enumerate(basis):
                if nH < kH or nV < kV:
                    continue  # Can't lose more than we have

                nH_new = nH - kH
                nV_new = nV - kV

                # Find target index
                target = (nH_new, nV_new)
                if target in basis:
                    j = basis.index(target)

                    # Compute coefficient
                    # Product of independent H and V loss
                    from math import comb
                    coeff_H = np.sqrt(comb(nH, kH)) * (eta_H ** ((nH - kH) / 2)) * ((1 - eta_H) ** (kH / 2))
                    coeff_V = np.sqrt(comb(nV, kV)) * (eta_V ** ((nV - kV) / 2)) * ((1 - eta_V) ** (kV / 2))
                    K[j, i] = coeff_H * coeff_V

            K_list_1517.append(K)

    # Remove all-zero operators
    K_list_1517 = [K for K in K_list_1517 if np.any(K != 0)]

    # Embed each Kraus operator into 18D bin space: I_780 ⊗ K_1517
    I_780 = np.eye(3, dtype=complex)
    K_list_embedded = [np.kron(I_780, K) for K in K_list_1517]

    return K_list_embedded


def detection_channel(
    eta_det: float = 1.0,
    p_dark: float = 0.0
) -> Tuple[List[np.ndarray], List[int]]:
    """
    On/off detection POVM for photon number measurement.

    Models single-photon detector with:
    - Efficiency eta_det
    - Dark count probability p_dark per bin

    POVM elements:
        E_0 = (1-p_dark) * sum_n (1-eta_det)^n |n><n|  (no click)
        E_1 = I - E_0  (click)

    Kraus operators are M_r = sqrt(E_r).

    Parameters
    ----------
    eta_det : float
        Detection efficiency (0 <= eta_det <= 1)
    p_dark : float
        Dark count probability (0 <= p_dark <= 1)

    Returns
    -------
    Tuple[List[np.ndarray], List[int]]
        (Kraus operators, outcome labels)
        Outcome 0 = no click, Outcome 1 = click

    Examples
    --------
    >>> K, outcomes = detection_channel(eta_det=0.9, p_dark=0.001)
    >>> # K[0] = no-click Kraus, K[1] = click Kraus
    """
    if not 0 <= eta_det <= 1:
        raise ValueError(f"eta_det must be in [0, 1], got {eta_det}")
    if not 0 <= p_dark <= 1:
        raise ValueError(f"p_dark must be in [0, 1], got {p_dark}")

    # For the 6D 1517 subspace (n_max = 2)
    dim = 6

    # No-click POVM element
    # E_0 = (1-p_dark) * [(1-eta)^0*|0><0| + (1-eta)^1*|1><1| + (1-eta)^2*|2><2|]
    # But we have multiple single-photon states and multi-photon states

    # Basis: vac, H, V, 2H, 2V, HV
    # Need photon number for each basis state
    n_per_state = np.array([0, 1, 1, 2, 2, 2])  # Total photon number

    # No-click operator (diagonal)
    E0 = np.zeros((dim, dim), dtype=complex)
    for i, n in enumerate(n_per_state):
        prob_no_click = (1 - p_dark) * ((1 - eta_det) ** n)
        E0[i, i] = prob_no_click

    # Click operator
    I = np.eye(dim, dtype=complex)
    E1 = I - E0

    # Kraus operators are matrix square roots
    # For diagonal operators, this is just sqrt of diagonal elements
    M0 = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        M0[i, i] = np.sqrt(E0[i, i]) if E0[i, i] > 0 else 0

    # E1 may not be diagonal (due to I - E0), but since E0 is diagonal, E1 is also diagonal
    M1 = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        M1[i, i] = np.sqrt(E1[i, i]) if E1[i, i] > 0 else 0

    return [M0, M1], [0, 1]


def detection_povm_single_site(
    eta_det: float = 1.0,
    p_dark: float = 0.0
) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    """
    On/off detection POVM for a single bin site (H and V detectors).

    This acts on the 18D bin space (780 x 1517). Since 780nm is filtered,
    detection only responds to 1517nm photons.

    Each site has two detectors (H and V), giving 4 possible outcomes:
        (0, 0): neither clicks
        (1, 0): only H clicks
        (0, 1): only V clicks
        (1, 1): both click

    Parameters
    ----------
    eta_det : float
        Detection efficiency (0 <= eta_det <= 1)
    p_dark : float
        Dark count probability per detector per bin

    Returns
    -------
    Tuple[List[np.ndarray], List[Tuple[int, int]]]
        (Kraus operators [4 x (18,18)], outcome labels [(d_H, d_V)])

    Notes
    -----
    1517nm basis: vac, H, V, 2H, 2V, HV with photon numbers:
        - vac: n_H=0, n_V=0
        - H:   n_H=1, n_V=0
        - V:   n_H=0, n_V=1
        - 2H:  n_H=2, n_V=0
        - 2V:  n_H=0, n_V=2
        - HV:  n_H=1, n_V=1

    For on/off detector with efficiency eta:
        P(no click | n photons) = (1-eta)^n * (1-p_dark)  (ignoring dark counts for n>0)
        P(click | n photons) = 1 - (1-eta)^n + small dark count correction
    """
    if not 0 <= eta_det <= 1:
        raise ValueError(f"eta_det must be in [0, 1], got {eta_det}")
    if not 0 <= p_dark <= 1:
        raise ValueError(f"p_dark must be in [0, 1], got {p_dark}")

    # 1517nm basis photon numbers (n_H, n_V)
    photon_numbers = [
        (0, 0),  # vac
        (1, 0),  # H
        (0, 1),  # V
        (2, 0),  # 2H
        (0, 2),  # 2V
        (1, 1),  # HV
    ]

    # Build POVM elements for 1517nm subspace (6D)
    # E_{d_H, d_V} = P(d_H | n_H) * P(d_V | n_V) for each basis state

    E_list_1517 = []
    outcomes = []

    for d_H in range(2):  # 0 = no click, 1 = click
        for d_V in range(2):
            E = np.zeros((6, 6), dtype=complex)
            for i, (n_H, n_V) in enumerate(photon_numbers):
                # Probability of outcome (d_H, d_V) given (n_H, n_V) photons
                if d_H == 0:  # H no click
                    if n_H == 0:
                        P_H = 1 - p_dark  # No photon, no dark count
                    else:
                        P_H = (1 - eta_det) ** n_H  # All photons missed
                else:  # H click
                    if n_H == 0:
                        P_H = p_dark  # Dark count only
                    else:
                        P_H = 1 - (1 - eta_det) ** n_H  # At least one detected

                if d_V == 0:  # V no click
                    if n_V == 0:
                        P_V = 1 - p_dark
                    else:
                        P_V = (1 - eta_det) ** n_V
                else:  # V click
                    if n_V == 0:
                        P_V = p_dark
                    else:
                        P_V = 1 - (1 - eta_det) ** n_V

                E[i, i] = P_H * P_V

            E_list_1517.append(E)
            outcomes.append((d_H, d_V))

    # Kraus operators: M = sqrt(E) (diagonal, so element-wise sqrt)
    M_list_1517 = []
    for E in E_list_1517:
        M = np.zeros_like(E)
        for i in range(6):
            M[i, i] = np.sqrt(max(0, E[i, i].real))
        M_list_1517.append(M)

    # Embed into 18D bin space: I_780 ⊗ M_1517
    # After fiber filtering, 780nm is vacuum, so we just need identity on 780
    I_780 = np.eye(3, dtype=complex)
    M_list_embedded = [np.kron(I_780, M) for M in M_list_1517]

    return M_list_embedded, outcomes


def detection_channel_two_mode(
    eta_det: float = 1.0,
    p_dark: float = 0.0
) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
    """
    On/off detection POVM for two output ports (e.g., after beam splitter).

    This returns Kraus operators for detecting photons at two sites (A and B),
    each with H and V polarization detectors. Total 4 detectors, 16 outcomes.

    The Kraus operators are tensor products: M_A ⊗ M_B
    where M_A and M_B are single-site detection operators.

    Parameters
    ----------
    eta_det : float
        Detection efficiency (same for all detectors)
    p_dark : float
        Dark count probability per detector

    Returns
    -------
    Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]
        (Kraus operators [16 x (324,324)], outcome labels)
        Each outcome is (dA_H, dA_V, dB_H, dB_V) where d=0 means no click, d=1 means click

    Notes
    -----
    For BSM (Bell State Measurement), the relevant outcomes are:
        - (1,0,0,1) or (0,1,1,0): Psi+ heralding
        - (0,1,0,1) or (1,0,1,0): Psi- heralding
        - Other patterns: no successful heralding

    Examples
    --------
    >>> K, outcomes = detection_channel_two_mode(eta_det=0.9)
    >>> # K has 16 operators, one for each click pattern
    >>> # outcomes[i] gives (dA_H, dA_V, dB_H, dB_V) for K[i]
    """
    # Get single-site detection operators (4 operators for 4 outcomes)
    M_single, outcomes_single = detection_povm_single_site(eta_det, p_dark)
    # M_single[i] is 18x18, outcomes_single[i] is (d_H, d_V)

    # Build tensor products for all 16 combinations
    K_list = []
    outcomes = []

    for iA, (dA_H, dA_V) in enumerate(outcomes_single):
        for iB, (dB_H, dB_V) in enumerate(outcomes_single):
            # Tensor product: M_A ⊗ M_B (324 x 324)
            K = np.kron(M_single[iA], M_single[iB])
            K_list.append(K)
            outcomes.append((dA_H, dA_V, dB_H, dB_V))

    return K_list, outcomes


def dephasing_channel(
    p_phi: float,
    dim: int = 3
) -> List[np.ndarray]:
    """
    Pure dephasing channel for atomic qubits.

        E(rho) = (1 - p_phi) * rho + p_phi * Z * rho * Z

    where Z = |0><0| - |1><1| flips the phase in the {|0>, |1>} subspace.

    Parameters
    ----------
    p_phi : float
        Dephasing probability (0 <= p_phi <= 1)
    dim : int
        Dimension of the atomic subspace (default: 3 for |0>, |1>, |e>)

    Returns
    -------
    List[np.ndarray]
        Kraus operators [K0, K1] where:
        K0 = sqrt(1 - p_phi) * I
        K1 = sqrt(p_phi) * Z

    Examples
    --------
    >>> K = dephasing_channel(p_phi=0.01)
    """
    if not 0 <= p_phi <= 1:
        raise ValueError(f"p_phi must be in [0, 1], got {p_phi}")

    K0 = np.sqrt(1 - p_phi) * np.eye(dim, dtype=complex)

    K1 = np.zeros((dim, dim), dtype=complex)
    # Z = |0><0| - |1><1| in the {|0>, |1>, |e>} basis
    K1[0, 0] = 1.0   # |0><0|
    K1[1, 1] = -1.0  # -|1><1|
    # |e> is unchanged by dephasing
    K1[2, 2] = 1.0 if dim >= 3 else 0
    K1 = np.sqrt(p_phi) * K1

    return [K0, K1]


def dephasing_channel_from_rate(
    gamma_phi: float,
    tau: float,
    dim: int = 3
) -> List[np.ndarray]:
    """
    Dephasing channel from continuous dephasing rate.

    p_phi = 1 - exp(-gamma_phi * tau)

    Parameters
    ----------
    gamma_phi : float
        Dephasing rate (1/time)
    tau : float
        Time duration
    dim : int
        Dimension of the atomic subspace

    Returns
    -------
    List[np.ndarray]
        Kraus operators
    """
    p_phi = 1 - np.exp(-gamma_phi * tau)
    return dephasing_channel(p_phi, dim)


# =============================================================================
# Fiber Channel Parameters (for realistic fiber transmission simulation)
# =============================================================================

class FiberChannelParams:
    """
    Parameters for fiber channel transmission with random polarization drift.

    This class models:
    - Jones matrix polarization drift (SU(2) random matrices)
    - Phase drift between arms
    - Loss with small fluctuations
    - PMD (polarization mode dispersion)

    Each trajectory samples new random parameters from the distributions.

    Parameters
    ----------
    U_mean_A : np.ndarray
        Mean Jones matrix for arm A (2x2 unitary)
    U_mean_B : np.ndarray
        Mean Jones matrix for arm B (2x2 unitary)
    polarization_model : str
        "haar" - fully random SU(2) (uncompensated fiber)
        "perturb" - small random rotation around mean (compensated fiber)
        "euler" - random Euler angles (intermediate)
    polarization_sigma : float
        For "perturb" model: standard deviation of rotation angle (radians)
    eta_mean : float
        Mean transmissivity (0 to 1)
    eta_std : float
        Standard deviation of transmissivity
    phase_drift_std : float
        Standard deviation of phase drift between arms (radians)
    pmd_enabled : bool
        Whether to include PMD effect
    pmd_delay_bins : int
        PMD delay in number of bins (integer shift)

    Examples
    --------
    >>> # Uncompensated long fiber
    >>> params = FiberChannelParams(polarization_model="haar")
    >>> # Compensated fiber with small drift
    >>> params = FiberChannelParams(polarization_model="perturb", polarization_sigma=0.1)
    >>> # Sample for one trajectory
    >>> U_A, U_B, eta, phase = params.sample_all(rng)
    """

    def __init__(
        self,
        U_mean_A: np.ndarray = None,
        U_mean_B: np.ndarray = None,
        polarization_model: str = "perturb",
        polarization_sigma: float = 0.1,
        eta_mean: float = 0.6,
        eta_std: float = 0.02,
        phase_drift_std: float = 0.2,
        pmd_enabled: bool = False,
        pmd_delay_bins: int = 0,
    ):
        if U_mean_A is None:
            U_mean_A = np.eye(2, dtype=complex)
        if U_mean_B is None:
            U_mean_B = np.eye(2, dtype=complex)

        self.U_mean_A = np.asarray(U_mean_A, dtype=complex)
        self.U_mean_B = np.asarray(U_mean_B, dtype=complex)
        self.polarization_model = polarization_model
        self.polarization_sigma = polarization_sigma
        self.eta_mean = eta_mean
        self.eta_std = eta_std
        self.phase_drift_std = phase_drift_std
        self.pmd_enabled = pmd_enabled
        self.pmd_delay_bins = pmd_delay_bins

    def sample_jones_A(self, rng: np.random.Generator) -> np.ndarray:
        """Sample a Jones matrix for arm A."""
        return self._sample_jones(self.U_mean_A, rng)

    def sample_jones_B(self, rng: np.random.Generator) -> np.ndarray:
        """Sample a Jones matrix for arm B."""
        return self._sample_jones(self.U_mean_B, rng)

    def _sample_jones(self, U_mean: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Sample a Jones matrix given the mean matrix."""
        model = self.polarization_model

        if model == "haar":
            # Fully random SU(2) from Haar measure
            # Use quaternion parameterization
            x = rng.standard_normal(4)
            x = x / np.linalg.norm(x)
            a, b, c, d = x
            U = np.array([
                [a + 1j*b, c + 1j*d],
                [-c + 1j*d, a - 1j*b]
            ], dtype=complex)

        elif model == "perturb":
            # Small random rotation around mean
            # Generate random axis on Bloch sphere
            axis = rng.standard_normal(3)
            axis = axis / np.linalg.norm(axis)

            # Random rotation angle
            delta_theta = rng.normal(0, self.polarization_sigma)

            # Build rotation: U = U_mean @ exp(i * delta_theta * (axis·sigma/2))
            from scipy.linalg import expm
            sigma = [
                np.array([[0, 1], [1, 0]], dtype=complex),   # sigma_x
                np.array([[0, -1j], [1j, 0]], dtype=complex), # sigma_y
                np.array([[1, 0], [0, -1]], dtype=complex)   # sigma_z
            ]
            generator = sum(a * s for a, s in zip(axis, sigma)) / 2
            delta_U = expm(1j * delta_theta * generator)
            U = U_mean @ delta_U

        elif model == "euler":
            # Random Euler angles
            # U = R_z(alpha) @ R_y(beta) @ R_z(gamma)
            alpha = rng.uniform(0, 2*np.pi)
            beta = rng.uniform(0, np.pi)
            gamma = rng.uniform(0, 2*np.pi)

            Rz_a = np.array([
                [np.exp(-1j*alpha/2), 0],
                [0, np.exp(1j*alpha/2)]
            ], dtype=complex)
            Ry_b = np.array([
                [np.cos(beta/2), -np.sin(beta/2)],
                [np.sin(beta/2), np.cos(beta/2)]
            ], dtype=complex)
            Rz_g = np.array([
                [np.exp(-1j*gamma/2), 0],
                [0, np.exp(1j*gamma/2)]
            ], dtype=complex)
            U = Rz_a @ Ry_b @ Rz_g

        else:
            raise ValueError(f"Unknown polarization_model: {model}")

        return U

    def sample_eta(self, rng: np.random.Generator) -> float:
        """Sample transmissivity from truncated normal distribution."""
        eta = rng.normal(self.eta_mean, self.eta_std)
        return np.clip(eta, 0, 1)

    def sample_phase_drift(self, rng: np.random.Generator) -> float:
        """Sample phase drift between arms (in radians)."""
        return rng.normal(0, self.phase_drift_std)

    def sample_all(self, rng: np.random.Generator) -> tuple:
        """
        Sample all parameters for one trajectory.

        Returns
        -------
        tuple
            (U_A, U_B, eta, phase_drift) where:
            - U_A: Jones matrix for arm A (2x2)
            - U_B: Jones matrix for arm B (2x2, with possible phase drift)
            - eta: transmissivity (0 to 1)
            - phase_drift: relative phase between arms (radians)
        """
        U_A = self.sample_jones_A(rng)
        U_B = self.sample_jones_B(rng)
        eta = self.sample_eta(rng)
        phase = self.sample_phase_drift(rng)

        # Apply phase drift to arm B (global phase affects interference)
        U_B = np.exp(1j * phase) * U_B

        return U_A, U_B, eta, phase
