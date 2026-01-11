"""
Event-Driven Detection Simulation (Quantum Jump Method)

This module implements physically correct two-photon detection following the
quantum jump / quantum trajectory method described in the expert document.

Key Concepts:
-------------
1. Jump operators J_{alpha,n} for detector alpha at time bin n
2. First-hit / first-jump sampling: exactly 0, 1, or 2 clicks total
3. Method B: threshold-based cumulative sampling for time ordering
4. MPS collapse after each detection event

Detection Modes (after BS + PBS):
---------------------------------
- H1: port1 H-polarization (from c_{H,n})
- V1: port1 V-polarization (from c_{V,n})
- H2: port2 H-polarization (from d_{H,n})
- V2: port2 V-polarization (from d_{V,n})

BSM Success Patterns:
---------------------
- Psi+: (H1, V2) or (V1, H2) - cross-port different polarization
- Psi-: (H1, H2) or (V1, V2) - cross-port same polarization
"""

from typing import Tuple, List, Optional
from dataclasses import dataclass
import numpy as np

from ..core.mps import MPSState
from ..hilbert.basis import SUBSPACE_1517


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class DetectionEvent:
    """
    A single detection event.

    Attributes
    ----------
    detector : str
        Detector label: "H1", "V1", "H2", "V2"
    bin_index : int
        Time bin index where click occurred
    site : int
        MPS site index (for internal use)
    """
    detector: str
    bin_index: int
    site: int


@dataclass
class TwoPhotonDetectionResult:
    """
    Result of a two-photon detection trial.

    Attributes
    ----------
    clicks : List[DetectionEvent]
        List of detection events (0, 1, or 2 clicks)
    success : bool
        Whether a BSM success pattern was found
    bell_state : str
        "Psi+" or "Psi-" if success, empty string otherwise
    spin_state : np.ndarray
        4x4 spin density matrix rho_AB after detection
        Basis: |00>, |01>, |10>, |11> where 0=down, 1=up
    spin_amplitudes : np.ndarray
        4D complex amplitude vector (if pure state extraction possible)
    p_click_first : float
        Probability of first click (for diagnostics)
    p_click_second : float
        Probability of second click (for diagnostics)
    """
    clicks: List[DetectionEvent]
    success: bool
    bell_state: str
    spin_state: np.ndarray
    spin_amplitudes: Optional[np.ndarray] = None
    p_click_first: float = 0.0
    p_click_second: float = 0.0


# =============================================================================
# Jump Operators (annihilation operators for detection)
# =============================================================================

def build_jump_operators_1517() -> dict:
    """
    Build jump (annihilation) operators for 1517nm detection modes.

    After BS, each output port has H and V polarization modes.
    The jump operator annihilates one photon of specific polarization.

    For bucket-type SNSPD:
    - J_H = |n_H - 1><n_H| for H-polarization (projects to lower H occupation)
    - J_V = |n_V - 1><n_V| for V-polarization (projects to lower V occupation)

    Returns
    -------
    dict
        {"H": J_H, "V": J_V} - each is a 6x6 matrix for 1517nm subspace
    """
    # 1517nm basis: (n_H, n_V)
    # 0: vac (0,0), 1: H (1,0), 2: V (0,1), 3: 2H (2,0), 4: 2V (0,2), 5: HV (1,1)
    basis = [(0, 0), (1, 0), (0, 1), (2, 0), (0, 2), (1, 1)]
    dim = 6

    # H annihilation: a_H |n_H, n_V> = sqrt(n_H) |n_H-1, n_V>
    J_H = np.zeros((dim, dim), dtype=complex)
    for j, (n_H, n_V) in enumerate(basis):
        if n_H > 0:
            target = (n_H - 1, n_V)
            if target in basis:
                i = basis.index(target)
                J_H[i, j] = np.sqrt(n_H)

    # V annihilation: a_V |n_H, n_V> = sqrt(n_V) |n_H, n_V-1>
    J_V = np.zeros((dim, dim), dtype=complex)
    for j, (n_H, n_V) in enumerate(basis):
        if n_V > 0:
            target = (n_H, n_V - 1)
            if target in basis:
                i = basis.index(target)
                J_V[i, j] = np.sqrt(n_V)

    return {"H": J_H, "V": J_V}


def build_jump_operators_18d() -> dict:
    """
    Build jump operators embedded in 18D bin space (780 x 1517).

    Since 780nm is filtered, we only detect 1517nm photons.
    J = I_780 ⊗ J_1517

    Returns
    -------
    dict
        {"H": J_H, "V": J_V} - each is 18x18 matrix
    """
    J_1517 = build_jump_operators_1517()
    I_780 = np.eye(3, dtype=complex)

    J_H_18 = np.kron(I_780, J_1517["H"])
    J_V_18 = np.kron(I_780, J_1517["V"])

    return {"H": J_H_18, "V": J_V_18}


# =============================================================================
# Core Detection Algorithm (Method B: Event-Driven Quantum Jump)
# =============================================================================

def compute_click_probability(
    mps: MPSState,
    site: int,
    jump_op: np.ndarray,
) -> float:
    """
    Compute the probability of a click at a specific site and polarization.

    For the jump operator J (annihilation operator), the click probability is:
    p = <psi| J^dagger J |psi> = <psi| n |psi>

    where n is the number operator for that mode. This gives the expected
    photon number at that mode.

    Note: The sum over all modes can exceed 1 because:
    - Multiple photons can exist in different modes
    - The same photon can have amplitude in H and V modes

    Parameters
    ----------
    mps : MPSState
        Current MPS state
    site : int
        Site index
    jump_op : np.ndarray
        Jump operator (annihilation operator), shape (d, d)

    Returns
    -------
    float
        Expected photon number (click "rate") at this mode
    """
    # Get reduced density matrix for the site
    rho = mps.get_reduced_density([site])

    # p = Tr(J^dagger J rho) = <n>
    JdJ = jump_op.conj().T @ jump_op
    p = np.real(np.trace(JdJ @ rho))

    return max(0.0, p)  # Ensure non-negative


def compute_all_click_probabilities(
    mps: MPSState,
    n_bins: int,
    eta_det: float = 1.0,
    jump_ops_18d: dict = None,
) -> np.ndarray:
    """
    Compute click probabilities for all 4 detectors at all bins.

    Parameters
    ----------
    mps : MPSState
        Current MPS state (layout: A1, B1, A2, B2, ..., AN, BN, atomA, atomB)
    n_bins : int
        Number of time bins
    eta_det : float
        Detection efficiency
    jump_ops_18d : dict, optional
        Pre-computed jump operators

    Returns
    -------
    np.ndarray
        Shape (n_bins, 4) - probabilities for [H1, V1, H2, V2] at each bin
    """
    if jump_ops_18d is None:
        jump_ops_18d = build_jump_operators_18d()

    J_H = jump_ops_18d["H"]
    J_V = jump_ops_18d["V"]

    probs = np.zeros((n_bins, 4))

    for n in range(n_bins):
        site_1 = 2 * n      # Port 1 (from arm A)
        site_2 = 2 * n + 1  # Port 2 (from arm B)

        # H1, V1 at site_1 (port 1)
        probs[n, 0] = eta_det * compute_click_probability(mps, site_1, J_H)  # H1
        probs[n, 1] = eta_det * compute_click_probability(mps, site_1, J_V)  # V1

        # H2, V2 at site_2 (port 2)
        probs[n, 2] = eta_det * compute_click_probability(mps, site_2, J_H)  # H2
        probs[n, 3] = eta_det * compute_click_probability(mps, site_2, J_V)  # V2

    return probs


def apply_jump_and_collapse(
    mps: MPSState,
    site: int,
    jump_op: np.ndarray,
) -> Tuple[MPSState, float]:
    """
    Apply jump operator to MPS and collapse the state.

    |psi'> = J |psi> / ||J|psi>||

    Parameters
    ----------
    mps : MPSState
        Current MPS state (will be modified in-place)
    site : int
        Site index where detection occurred
    jump_op : np.ndarray
        Jump operator (18x18)

    Returns
    -------
    Tuple[MPSState, float]
        (collapsed MPS, probability of this jump)
    """
    from tenpy.linalg.np_conserved import Array

    # Get the site tensor
    theta = mps._mps.get_theta(site, n=1)  # (vL, p, vR)
    theta_np = theta.to_ndarray()

    d = mps.d[site]
    J = jump_op.reshape(d, d)

    # Apply jump: J @ theta (contract over physical index)
    # theta shape: (chiL, d, chiR)
    J_theta = np.einsum('ij,ajb->aib', J, theta_np)

    # Compute norm squared (= probability)
    p = np.linalg.norm(J_theta) ** 2

    if p < 1e-15:
        # No probability for this jump - state becomes invalid
        raise ValueError(f"Jump probability is zero at site {site}")

    # Normalize
    J_theta_normalized = J_theta / np.sqrt(p)

    # Write back to MPS
    theta_arr = Array.from_ndarray_trivial(J_theta_normalized, labels=['vL', 'p', 'vR'])
    mps._mps.set_B(site, theta_arr, form='Th')
    mps._mps.canonical_form_finite(renormalize=True)

    return mps, p


def sample_first_click_method_b(
    mps: MPSState,
    n_bins: int,
    eta_det: float,
    rng: np.random.Generator,
    jump_ops_18d: dict = None,
    verbose: bool = False,
) -> Tuple[Optional[DetectionEvent], float, np.ndarray]:
    """
    Sample the first click using Method B (threshold cumulative).

    For a two-photon state, the total expected photon number sum(p_{alpha,n})
    equals approximately 2 (one photon in each arm). We normalize this to get
    the probability distribution for the first click.

    Algorithm:
    1. Compute expected photon numbers for all modes
    2. Normalize to get probability distribution
    3. Draw threshold u ~ U(0,1)
    4. Accumulate probabilities C = sum p_{alpha,n} in time order
    5. First bin where C >= u is the click location

    Parameters
    ----------
    mps : MPSState
        Current MPS state
    n_bins : int
        Number of time bins
    eta_det : float
        Detection efficiency
    rng : np.random.Generator
        Random number generator
    jump_ops_18d : dict, optional
        Pre-computed jump operators
    verbose : bool
        Whether to print debug info

    Returns
    -------
    Tuple[Optional[DetectionEvent], float, np.ndarray]
        (event, total_photon_number, all_probs_normalized)
        event is None if no click (efficiency loss)
    """
    if jump_ops_18d is None:
        jump_ops_18d = build_jump_operators_18d()

    # Compute all click probabilities (expected photon numbers)
    all_probs = compute_all_click_probabilities(mps, n_bins, eta_det, jump_ops_18d)

    # Total expected photon number (should be ~2 for two-photon state)
    total_photon_number = all_probs.sum()

    if verbose:
        print(f"  First click: total_photon_number = {total_photon_number:.6f}")

    # Normalize to get probability distribution for first click
    if total_photon_number < 1e-15:
        # No photons at all
        return None, 0.0, all_probs

    # First decide if a click happens at all (based on detection efficiency)
    # For ideal detector (eta=1), first click should always happen if there are photons
    # The probabilities are already scaled by eta_det
    p_click = min(1.0, total_photon_number)  # Probability of at least one click

    u = rng.uniform(0, 1)
    if u > p_click:
        if verbose:
            print(f"  No first click (u={u:.4f} > p_click={p_click:.4f})")
        return None, total_photon_number, all_probs

    # Normalize probabilities for selecting which mode
    probs_normalized = all_probs / total_photon_number

    # Draw threshold for mode selection
    u_mode = rng.uniform(0, 1)

    # Find first bin where cumulative sum exceeds u_mode
    C = 0.0
    detector_labels = ["H1", "V1", "H2", "V2"]

    for n in range(n_bins):
        p_bin = probs_normalized[n].sum()

        if C + p_bin >= u_mode:
            # Click in this bin - select detector proportionally
            p_in_bin = probs_normalized[n]
            if p_in_bin.sum() < 1e-15:
                C += p_bin
                continue

            p_in_bin_renorm = p_in_bin / p_in_bin.sum()

            det_idx = rng.choice(4, p=p_in_bin_renorm)
            detector = detector_labels[det_idx]

            # Determine site
            site = 2 * n if det_idx < 2 else 2 * n + 1

            event = DetectionEvent(
                detector=detector,
                bin_index=n,
                site=site,
            )

            if verbose:
                print(f"  First click: {detector} at bin {n} (p={all_probs[n, det_idx]:.6f})")

            return event, total_photon_number, probs_normalized

        C += p_bin

    # Should not reach here if probabilities sum correctly
    return None, total_photon_number, probs_normalized


def sample_second_click_method_b(
    mps: MPSState,
    n_bins: int,
    eta_det: float,
    rng: np.random.Generator,
    first_event: DetectionEvent,
    jump_ops_18d: dict = None,
    verbose: bool = False,
) -> Tuple[Optional[DetectionEvent], float]:
    """
    Sample the second click after the first click has occurred.

    After the first jump, the MPS has been collapsed to a single-photon state.
    The total expected photon number should now be ~1.

    Parameters
    ----------
    mps : MPSState
        Collapsed MPS (single-photon state + atoms)
    n_bins : int
        Number of time bins
    eta_det : float
        Detection efficiency
    rng : np.random.Generator
        Random number generator
    first_event : DetectionEvent
        The first detection event
    jump_ops_18d : dict, optional
        Pre-computed jump operators
    verbose : bool
        Whether to print debug info

    Returns
    -------
    Tuple[Optional[DetectionEvent], float]
        (event, total_photon_number)
        event is None if no second click
    """
    if jump_ops_18d is None:
        jump_ops_18d = build_jump_operators_18d()

    # Recompute probabilities after collapse
    all_probs = compute_all_click_probabilities(mps, n_bins, eta_det, jump_ops_18d)

    # Total expected photon number (should be ~1 for single-photon state)
    total_photon_number = all_probs.sum()

    if verbose:
        print(f"  Second click: total_photon_number = {total_photon_number:.6f}")

    if total_photon_number < 1e-15:
        return None, 0.0

    # Probability of second click
    p_click = min(1.0, total_photon_number)

    # Draw random number for "does click happen"
    u = rng.uniform(0, 1)
    if u > p_click:
        if verbose:
            print(f"  No second click (u={u:.4f} > p_click={p_click:.4f})")
        return None, total_photon_number

    # Normalize and select mode
    probs_normalized = all_probs / total_photon_number
    u_mode = rng.uniform(0, 1)

    C = 0.0
    detector_labels = ["H1", "V1", "H2", "V2"]

    for n in range(n_bins):
        p_bin = probs_normalized[n].sum()

        if C + p_bin >= u_mode:
            p_in_bin = probs_normalized[n]
            if p_in_bin.sum() < 1e-15:
                C += p_bin
                continue

            p_in_bin_renorm = p_in_bin / p_in_bin.sum()
            det_idx = rng.choice(4, p=p_in_bin_renorm)
            detector = detector_labels[det_idx]

            site = 2 * n if det_idx < 2 else 2 * n + 1

            event = DetectionEvent(
                detector=detector,
                bin_index=n,
                site=site,
            )

            if verbose:
                print(f"  Second click: {detector} at bin {n} (p={all_probs[n, det_idx]:.6f})")

            return event, total_photon_number

        C += p_bin

    return None, total_photon_number


# =============================================================================
# Main Detection Function
# =============================================================================

def run_two_photon_detection(
    mps: MPSState,
    n_bins: int,
    eta_det: float = 0.85,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = True,
) -> TwoPhotonDetectionResult:
    """
    Run complete two-photon detection with quantum jump method.

    This implements the physically correct detection simulation:
    1. Sample first click using Method B
    2. Collapse MPS with jump operator
    3. Sample second click from collapsed state
    4. Collapse again and extract spin state

    Parameters
    ----------
    mps : MPSState
        MPS state after BS (layout: A1, B1, A2, B2, ..., atomA, atomB)
    n_bins : int
        Number of time bins
    eta_det : float
        Detection efficiency (typical SNSPD: 0.85)
    rng : np.random.Generator, optional
        Random number generator
    verbose : bool
        Whether to print progress

    Returns
    -------
    TwoPhotonDetectionResult
        Detection result including clicks, success status, and spin state
    """
    if rng is None:
        rng = np.random.default_rng()

    if verbose:
        print("\n" + "=" * 60)
        print("Two-Photon Detection (Quantum Jump Method)")
        print("=" * 60)
        print(f"  eta_det = {eta_det:.3f}")
        print(f"  n_bins = {n_bins}")

    # Build jump operators
    jump_ops = build_jump_operators_18d()

    # Make a copy of MPS for collapse operations
    mps_work = mps.copy()

    clicks = []

    # --- First Click ---
    first_event, p1_total, _ = sample_first_click_method_b(
        mps=mps_work,
        n_bins=n_bins,
        eta_det=eta_det,
        rng=rng,
        jump_ops_18d=jump_ops,
        verbose=verbose,
    )

    if first_event is None:
        # No first click - return empty result
        spin_state = extract_spin_state(mps_work, n_bins)
        return TwoPhotonDetectionResult(
            clicks=[],
            success=False,
            bell_state="",
            spin_state=spin_state,
            p_click_first=p1_total,
            p_click_second=0.0,
        )

    clicks.append(first_event)

    # Apply jump operator and collapse
    pol1 = "H" if "H" in first_event.detector else "V"
    J1 = jump_ops[pol1]
    mps_work, _ = apply_jump_and_collapse(mps_work, first_event.site, J1)

    # --- Second Click ---
    second_event, p2_total = sample_second_click_method_b(
        mps=mps_work,
        n_bins=n_bins,
        eta_det=eta_det,
        rng=rng,
        first_event=first_event,
        jump_ops_18d=jump_ops,
        verbose=verbose,
    )

    if second_event is not None:
        clicks.append(second_event)

        # Apply second jump operator (for probability calculation only)
        pol2 = "H" if "H" in second_event.detector else "V"
        J2 = jump_ops[pol2]
        mps_work, _ = apply_jump_and_collapse(mps_work, second_event.site, J2)

    # Check for BSM success first
    success, bell_state = check_bsm_success(clicks)

    # Extract final spin state
    # For BSM success with two clicks, use conditional extraction on ORIGINAL MPS
    # This properly projects onto the detected photon modes
    if len(clicks) == 2:
        spin_state = extract_conditional_spin_state(
            mps=mps,  # Use original MPS, not collapsed one
            n_bins=n_bins,
            click1=clicks[0],
            click2=clicks[1],
        )
    else:
        # For 0 or 1 click, use the collapsed MPS
        spin_state = extract_spin_state(mps_work, n_bins)

    if verbose:
        print(f"\n  Result:")
        print(f"    Clicks: {[(c.detector, c.bin_index) for c in clicks]}")
        print(f"    Success: {success}")
        if success:
            print(f"    Bell state: {bell_state}")

    return TwoPhotonDetectionResult(
        clicks=clicks,
        success=success,
        bell_state=bell_state,
        spin_state=spin_state,
        p_click_first=p1_total,
        p_click_second=p2_total,
    )


# =============================================================================
# Helper Functions
# =============================================================================

def extract_spin_state(mps: MPSState, n_bins: int) -> np.ndarray:
    """
    Extract the two-atom spin density matrix from MPS.

    After detection, atoms are at sites 2*n_bins and 2*n_bins + 1.

    Parameters
    ----------
    mps : MPSState
        MPS state after detection
    n_bins : int
        Number of time bins

    Returns
    -------
    np.ndarray
        4x4 density matrix in computational basis |00>, |01>, |10>, |11>
        where 0 = |down> (index 0 in 3D atom), 1 = |up> (index 1 in 3D atom)
    """
    site_A = 2 * n_bins
    site_B = 2 * n_bins + 1

    # Get full 9x9 two-atom density matrix
    rho_full = mps.get_reduced_density([site_A, site_B])

    # rho_full shape should be (3, 3, 3, 3) or (9, 9) depending on implementation
    # Reshape to (9, 9) if needed
    if rho_full.ndim == 4:
        rho_full = rho_full.reshape(9, 9)

    # Extract 4x4 qubit subspace (|0>, |1> for each atom, ignoring |e>)
    # Full basis: |00>, |01>, |0e>, |10>, |11>, |1e>, |e0>, |e1>, |ee>
    # (row-major: first index is atom A, second is atom B)
    # Qubit basis: |00> (idx 0), |01> (idx 1), |10> (idx 3), |11> (idx 4)
    qubit_indices = [0, 1, 3, 4]

    rho_qubit = np.zeros((4, 4), dtype=complex)
    for i, qi in enumerate(qubit_indices):
        for j, qj in enumerate(qubit_indices):
            rho_qubit[i, j] = rho_full[qi, qj]

    # Renormalize (in case there's population in |e>)
    trace = np.trace(rho_qubit)
    if trace > 1e-10:
        rho_qubit = rho_qubit / trace

    return rho_qubit


def extract_conditional_spin_state(
    mps: MPSState,
    n_bins: int,
    click1: DetectionEvent,
    click2: Optional[DetectionEvent] = None,
) -> np.ndarray:
    """
    Extract spin state conditioned on detection events.

    When photons are detected at specific modes, the spin state is the
    post-measurement state after the photon is annihilated.

    This function:
    1. Applies annihilation (jump) operators at the detected modes
    2. Normalizes the resulting state
    3. Extracts the atomic reduced density matrix

    Parameters
    ----------
    mps : MPSState
        MPS state after BS (original, not modified)
    n_bins : int
        Number of time bins
    click1 : DetectionEvent
        First detection event
    click2 : DetectionEvent, optional
        Second detection event (if two clicks)

    Returns
    -------
    np.ndarray
        4x4 spin density matrix conditioned on detection
    """
    from tenpy.linalg.np_conserved import Array

    # Work on a copy
    mps_cond = mps.copy()

    # Get jump operators
    jump_ops = build_jump_operators_18d()

    def get_jump_op_for_detector(detector: str) -> Tuple[int, np.ndarray]:
        """Get site index and jump operator for a detector."""
        pol = "H" if "H" in detector else "V"
        return jump_ops[pol]

    # Apply jump operator for click1
    pol1 = "H" if "H" in click1.detector else "V"
    J1 = jump_ops[pol1]

    # Get site tensor and apply jump
    theta1 = mps_cond._mps.get_theta(click1.site, n=1)
    theta1_np = theta1.to_ndarray()
    d1 = theta1_np.shape[1]
    J1_d = J1.reshape(d1, d1)
    J1_theta = np.einsum('ij,ajb->aib', J1_d, theta1_np)

    # Check if norm is non-zero
    norm1 = np.linalg.norm(J1_theta)
    if norm1 < 1e-15:
        # No amplitude for this detection pattern - return maximally mixed
        return np.eye(4, dtype=complex) / 4

    J1_theta /= norm1
    theta1_arr = Array.from_ndarray_trivial(J1_theta, labels=['vL', 'p', 'vR'])
    mps_cond._mps.set_B(click1.site, theta1_arr, form='Th')

    # Apply jump operator for click2 if exists
    if click2 is not None:
        pol2 = "H" if "H" in click2.detector else "V"
        J2 = jump_ops[pol2]

        theta2 = mps_cond._mps.get_theta(click2.site, n=1)
        theta2_np = theta2.to_ndarray()
        d2 = theta2_np.shape[1]
        J2_d = J2.reshape(d2, d2)
        J2_theta = np.einsum('ij,ajb->aib', J2_d, theta2_np)

        norm2 = np.linalg.norm(J2_theta)
        if norm2 < 1e-15:
            return np.eye(4, dtype=complex) / 4

        J2_theta /= norm2
        theta2_arr = Array.from_ndarray_trivial(J2_theta, labels=['vL', 'p', 'vR'])
        mps_cond._mps.set_B(click2.site, theta2_arr, form='Th')

    # Put MPS in canonical form
    mps_cond._mps.canonical_form_finite(renormalize=True)

    # Extract atomic state
    return extract_spin_state(mps_cond, n_bins)


def check_bsm_success(clicks: List[DetectionEvent]) -> Tuple[bool, str]:
    """
    Check if detection pattern indicates BSM success.

    BSM success requires:
    1. Exactly two clicks
    2. Both clicks at the SAME time bin (temporal overlap for HOM interference)
    3. Correct detector pattern:
       - Psi+: (H1, V2) or (V1, H2) - cross-port different polarization
       - Psi-: (H1, H2) or (V1, V2) - cross-port same polarization

    Parameters
    ----------
    clicks : List[DetectionEvent]
        List of detection events

    Returns
    -------
    Tuple[bool, str]
        (success, bell_state)
    """
    if len(clicks) != 2:
        return False, ""

    # Check if both clicks are at the same bin (required for HOM interference)
    if clicks[0].bin_index != clicks[1].bin_index:
        return False, ""  # Different bins = no proper interference

    detectors = {clicks[0].detector, clicks[1].detector}

    # Psi+ patterns: cross-port different polarization
    if detectors == {"H1", "V2"} or detectors == {"V1", "H2"}:
        return True, "Psi+"

    # Psi- patterns: cross-port same polarization
    if detectors == {"H1", "H2"} or detectors == {"V1", "V2"}:
        return True, "Psi-"

    return False, ""


def compute_photon_statistics(
    mps: MPSState,
    n_bins: int,
    verbose: bool = False,
) -> dict:
    """
    Compute photon number statistics for the MPS state.

    This helps quantify:
    - Probability of having 0, 1, or 2 photons
    - Loss probability (photons in vacuum)
    - Distribution across H/V polarizations

    Parameters
    ----------
    mps : MPSState
        MPS state after BS (layout: A1, B1, A2, B2, ..., atomA, atomB)
    n_bins : int
        Number of time bins
    verbose : bool
        Whether to print detailed statistics

    Returns
    -------
    dict
        Dictionary with photon statistics:
        - 'n_total': total expected photon number
        - 'p_0photon': probability of 0 photons
        - 'p_1photon': probability of exactly 1 photon
        - 'p_2photon': probability of 2+ photons
        - 'loss_prob': probability that at least one photon is lost
        - 'n_H': expected H-polarized photons
        - 'n_V': expected V-polarized photons
    """
    # Get jump operators
    jump_ops = build_jump_operators_18d()
    J_H = jump_ops["H"]
    J_V = jump_ops["V"]

    # Compute expected photon numbers
    n_H_total = 0.0
    n_V_total = 0.0

    for n in range(n_bins):
        # Port 1 (site 2n) and Port 2 (site 2n+1)
        for site in [2 * n, 2 * n + 1]:
            rho = mps.get_reduced_density([site])

            # n_H = Tr(J_H^dagger J_H rho)
            JdJ_H = J_H.conj().T @ J_H
            n_H = np.real(np.trace(JdJ_H @ rho))
            n_H_total += n_H

            # n_V = Tr(J_V^dagger J_V rho)
            JdJ_V = J_V.conj().T @ J_V
            n_V = np.real(np.trace(JdJ_V @ rho))
            n_V_total += n_V

    n_total = n_H_total + n_V_total

    # For a state with 0, 1, or 2 photons:
    # - n_total ≈ 0 means vacuum (both lost or never emitted)
    # - n_total ≈ 1 means one photon arrived, one lost
    # - n_total ≈ 2 means both photons arrived
    # We use n_total as a proxy for the probability

    # Loss probability: probability that fewer than 2 photons arrived
    # This is roughly: p_loss ≈ 2 - n_total (for n_total <= 2)
    loss_prob = max(0.0, 2.0 - n_total)

    stats = {
        'n_total': n_total,
        'n_H': n_H_total,
        'n_V': n_V_total,
        'loss_prob': loss_prob,
    }

    if verbose:
        print(f"\n  Photon Statistics:")
        print(f"    Total expected photons: {n_total:.4f}")
        print(f"    H-polarized: {n_H_total:.4f}")
        print(f"    V-polarized: {n_V_total:.4f}")
        print(f"    Loss probability (< 2 photons): {loss_prob:.4f}")
        print(f"    Two-photon arrival probability: {2 - loss_prob:.4f}")

    return stats


def compute_fidelity_with_bell(spin_state: np.ndarray, target_bell: str) -> float:
    """
    Compute fidelity of spin state with target Bell state.

    Parameters
    ----------
    spin_state : np.ndarray
        4x4 density matrix in computational basis
    target_bell : str
        "Psi+", "Psi-", "Phi+", or "Phi-"

    Returns
    -------
    float
        Fidelity F = <target|rho|target>
    """
    # Bell states in computational basis |00>, |01>, |10>, |11>
    bell_states = {
        "Phi+": np.array([1, 0, 0, 1]) / np.sqrt(2),   # (|00> + |11>)/sqrt(2)
        "Phi-": np.array([1, 0, 0, -1]) / np.sqrt(2),  # (|00> - |11>)/sqrt(2)
        "Psi+": np.array([0, 1, 1, 0]) / np.sqrt(2),   # (|01> + |10>)/sqrt(2)
        "Psi-": np.array([0, 1, -1, 0]) / np.sqrt(2),  # (|01> - |10>)/sqrt(2)
    }

    if target_bell not in bell_states:
        raise ValueError(f"Unknown Bell state: {target_bell}")

    psi = bell_states[target_bell]
    fidelity = np.real(psi.conj() @ spin_state @ psi)

    return float(fidelity)
