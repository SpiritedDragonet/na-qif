"""
Wave Packet Visualization

This module provides functions to extract and visualize wave packets
from MPS states, including intensity envelopes and single-photon probabilities.
"""

from typing import Tuple, Optional, List
import numpy as np
import matplotlib.pyplot as plt

from ..core.mps import MPSState
from ..config import TimeGrid


# ============================================================================
# Operators for Wave Packet Extraction
# ============================================================================

def _telecom_ops_1517():
    """
    Construct projection and number operators for the 1517nm subspace.

    1517 basis: [vac, 1H, 1V, 2H, 2V, HV]

    Returns
    -------
    Tuple of np.ndarray
        (P1_1517, N_1517, P1H_1517, P1V_1517, NH_1517, NV_1517)
        - P1_1517: Single-photon projection (all pol)
        - N_1517: Total photon number operator
        - P1H_1517: Single H-photon projection
        - P1V_1517: Single V-photon projection
        - NH_1517: H-photon number operator
        - NV_1517: V-photon number operator
    """
    # Basis: [vac, 1H, 1V, 2H, 2V, HV]
    # Single-photon projection (all pol)
    P1_1517 = np.diag([0, 1, 1, 0, 0, 0]).astype(complex)

    # Total photon number
    N_1517 = np.diag([0, 1, 1, 2, 2, 2]).astype(complex)

    # H/V single-photon projections
    P1H_1517 = np.diag([0, 1, 0, 0, 0, 0]).astype(complex)
    P1V_1517 = np.diag([0, 0, 1, 0, 0, 0]).astype(complex)

    # H/V photon numbers
    NH_1517 = np.diag([0, 1, 0, 2, 0, 1]).astype(complex)
    NV_1517 = np.diag([0, 0, 1, 0, 2, 1]).astype(complex)

    return P1_1517, N_1517, P1H_1517, P1V_1517, NH_1517, NV_1517


def telecom_ops_bin18():
    """
    Construct telecom operators embedded in the 18D bin space.

    Bin space = 780(3D) x 1517(6D) = 18D
    Assumes flatten order: |i_780> ⊗ |j_1517>, index = i_780 * 6 + j_1517

    Returns
    -------
    Tuple of np.ndarray
        (P1_bin, N_bin, P1H_bin, P1V_bin, NH_bin, NV_bin)
        Each is 18x18 acting on the full bin space
    """
    I_780 = np.eye(3, dtype=complex)

    P1_1517, N_1517, P1H_1517, P1V_1517, NH_1517, NV_1517 = _telecom_ops_1517()

    # Embed: I_780 ⊗ Op_1517
    P1_bin = np.kron(I_780, P1_1517)
    N_bin = np.kron(I_780, N_1517)
    P1H_bin = np.kron(I_780, P1H_1517)
    P1V_bin = np.kron(I_780, P1V_1517)
    NH_bin = np.kron(I_780, NH_1517)
    NV_bin = np.kron(I_780, NV_1517)

    return P1_bin, N_bin, P1H_bin, P1V_bin, NH_bin, NV_bin


# ============================================================================
# Wave Packet Extraction
# ============================================================================

def extract_wavepacket(
    mps: MPSState,
    n_bins: int,
    use_single_photon_prob: bool = True,
    polarized: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract wave packet data from MPS state.

    Chain layout: atomA, atomB, A1, B1, A2, B2, ..., AN, BN
    Site indices: A_n = 2 + 2*(n-1), B_n = 3 + 2*(n-1)

    Parameters
    ----------
    mps : MPSState
        MPS state (access mps._mps for TeNPy MPS)
    n_bins : int
        Number of time bins to extract
    use_single_photon_prob : bool
        If True, return single-photon probability q_n
        If False, return intensity <N_n>
    polarized : bool
        If True, return separate H and V components
        If False, return total (H+V)

    Returns
    -------
    Tuple of np.ndarray
        (data_A, data_B) where each is:
        - (n_bins,) if polarized=False
        - (n_bins, 2) if polarized=True [H, V columns]
    """
    P1_bin, N_bin, P1H_bin, P1V_bin, NH_bin, NV_bin = telecom_ops_bin18()

    # Choose operator based on mode
    if use_single_photon_prob:
        if polarized:
            OpA = P1H_bin  # Will use separate H/V
            OpB = P1V_bin
        else:
            OpA = P1_bin
            OpB = P1_bin
    else:
        if polarized:
            OpA = NH_bin
            OpB = NV_bin
        else:
            OpA = N_bin
            OpB = N_bin

    # Initialize arrays
    if polarized:
        data_A = np.zeros((n_bins, 2))
        data_B = np.zeros((n_bins, 2))
    else:
        data_A = np.zeros(n_bins)
        data_B = np.zeros(n_bins)

    for n in range(1, n_bins + 1):
        idx_A = 2 + 2 * (n - 1)  # A_n site index
        idx_B = 3 + 2 * (n - 1)  # B_n site index

        # Get reduced density matrices
        rhoA = mps.get_reduced_density([idx_A])
        rhoB = mps.get_reduced_density([idx_B])

        if polarized:
            # H component
            data_A[n - 1, 0] = float(np.real(np.trace(rhoA @ P1H_bin)))
            data_B[n - 1, 0] = float(np.real(np.trace(rhoB @ P1H_bin)))
            # V component
            data_A[n - 1, 1] = float(np.real(np.trace(rhoA @ P1V_bin)))
            data_B[n - 1, 1] = float(np.real(np.trace(rhoB @ P1V_bin)))
        else:
            data_A[n - 1] = float(np.real(np.trace(rhoA @ OpA)))
            data_B[n - 1] = float(np.real(np.trace(rhoB @ OpB)))

    return data_A, data_B


def extract_intensity_envelope(
    mps: MPSState,
    n_bins: int,
    polarized: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract intensity envelope <N_n> for each bin.

    Parameters
    ----------
    mps : MPSState
        MPS state
    n_bins : int
        Number of time bins
    polarized : bool
        If True, return H and V separately

    Returns
    -------
    Tuple of np.ndarray
        (pA, pB) intensity arrays
    """
    return extract_wavepacket(
        mps, n_bins,
        use_single_photon_prob=False,
        polarized=polarized
    )


def extract_single_photon_prob(
    mps: MPSState,
    n_bins: int,
    polarized: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract single-photon probability for each bin.

    Parameters
    ----------
    mps : MPSState
        MPS state
    n_bins : int
        Number of time bins
    polarized : bool
        If True, return H and V separately

    Returns
    -------
    Tuple of np.ndarray
        (qA, qB) probability arrays
    """
    return extract_wavepacket(
        mps, n_bins,
        use_single_photon_prob=True,
        polarized=polarized
    )


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_wavepacket(
    data_A: np.ndarray,
    data_B: np.ndarray,
    time_grid: Optional[TimeGrid] = None,
    polarized: bool = False,
    normalize: bool = False,
    title: str = "Wave Packet",
    labels: Optional[Tuple[str, str]] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot wave packet for both arms.

    Parameters
    ----------
    data_A : np.ndarray
        Arm A data (n_bins,) or (n_bins, 2) for polarized
    data_B : np.ndarray
        Arm B data
    time_grid : TimeGrid, optional
        Time grid for x-axis
    polarized : bool
        Whether data is polarized (H/V separate)
    normalize : bool
        If True, normalize to sum=1
    title : str
        Plot title
    labels : Tuple[str, str], optional
        Legend labels for arms A and B
    ax : plt.Axes, optional
        Existing axes to plot on

    Returns
    -------
    plt.Axes
        The axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    n_bins = len(data_A)
    if time_grid is None:
        x = np.arange(n_bins)
        xlabel = "Bin index"
    else:
        x = time_grid.t[:n_bins]
        xlabel = "Time (s)"

    # Normalize if requested
    if normalize:
        if polarized:
            data_A = data_A / (data_A.sum() + 1e-15)
            data_B = data_B / (data_B.sum() + 1e-15)
        else:
            data_A = data_A / (data_A.sum() + 1e-15)
            data_B = data_B / (data_B.sum() + 1e-15)

    # Plot
    if polarized:
        # data_A and data_B are (n_bins, 2) with [H, V] columns
        ax.plot(x, data_A[:, 0], '--', label=f"A: H" if labels is None else labels[0] + " H",
                color='tab:blue', alpha=0.7)
        ax.plot(x, data_A[:, 1], '--', label=f"A: V" if labels is None else labels[0] + " V",
                color='tab:blue', alpha=0.9)
        ax.plot(x, data_B[:, 0], '-', label=f"B: H" if labels is None else labels[1] + " H",
                color='tab:orange', alpha=0.7)
        ax.plot(x, data_B[:, 1], '-', label=f"B: V" if labels is None else labels[1] + " V",
                color='tab:orange', alpha=0.9)
    else:
        label_A = "Arm A" if labels is None else labels[0]
        label_B = "Arm B" if labels is None else labels[1]
        ax.plot(x, data_A, '-', label=label_A, color='tab:blue')
        ax.plot(x, data_B, '-', label=label_B, color='tab:orange')

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Probability" if normalize else "Intensity / <N>")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_intensity_envelope(
    mps: MPSState,
    n_bins: int,
    time_grid: Optional[TimeGrid] = None,
    polarized: bool = False,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot intensity envelope <N_n> for both arms.

    Parameters
    ----------
    mps : MPSState
        MPS state
    n_bins : int
        Number of time bins
    time_grid : TimeGrid, optional
        Time grid
    polarized : bool
        Whether to show H/V separately
    ax : plt.Axes, optional
        Existing axes

    Returns
    -------
    plt.Axes
    """
    data_A, data_B = extract_intensity_envelope(mps, n_bins, polarized=polarized)
    return plot_wavepacket(
        data_A, data_B, time_grid,
        polarized=polarized,
        normalize=False,
        title="Intensity Envelope <N>",
        ax=ax
    )


def plot_single_photon_prob(
    mps: MPSState,
    n_bins: int,
    time_grid: Optional[TimeGrid] = None,
    polarized: bool = False,
    normalize: bool = True,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot single-photon probability for both arms.

    Parameters
    ----------
    mps : MPSState
        MPS state
    n_bins : int
        Number of time bins
    time_grid : TimeGrid, optional
        Time grid
    polarized : bool
        Whether to show H/V separately
    normalize : bool
        Whether to normalize (recommended)
    ax : plt.Axes, optional
        Existing axes

    Returns
    -------
    plt.Axes
    """
    data_A, data_B = extract_single_photon_prob(mps, n_bins, polarized=polarized)
    return plot_wavepacket(
        data_A, data_B, time_grid,
        polarized=polarized,
        normalize=normalize,
        title="Single-Photon Probability",
        ax=ax
    )


def plot_mode_overlap(
    data_A: np.ndarray,
    data_B: np.ndarray,
    time_grid: Optional[TimeGrid] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot mode overlap between two arms.

    Overlap M = |sum_n (xi_A_n* xi_B_n)|^2

    Parameters
    ----------
    data_A : np.ndarray
        Complex amplitudes for arm A
    data_B : np.ndarray
        Complex amplitudes for arm B
    time_grid : TimeGrid, optional
        Time grid
    ax : plt.Axes, optional
        Existing axes

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    n_bins = len(data_A)
    if time_grid is None:
        x = np.arange(n_bins)
        xlabel = "Bin index"
    else:
        x = time_grid.t[:n_bins]
        xlabel = "Time (s)"

    # Compute overlap
    overlap = np.abs(np.sum(data_A * np.conj(data_B))) ** 2

    ax.plot(x, np.real(data_A), '--', label='Re(A)', color='tab:blue', alpha=0.7)
    ax.plot(x, np.imag(data_A), ':', label='Im(A)', color='tab:blue', alpha=0.5)
    ax.plot(x, np.real(data_B), '--', label='Re(B)', color='tab:orange', alpha=0.7)
    ax.plot(x, np.imag(data_B), ':', label='Im(B)', color='tab:orange', alpha=0.5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Mode Overlap M = {overlap:.4f}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax
