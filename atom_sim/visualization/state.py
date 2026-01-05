"""
State Visualization

This module provides functions to visualize quantum states,
including density matrices, Wigner functions, and Bloch spheres.
"""

from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def plot_density_matrix(
    rho: np.ndarray,
    title: str = "Density Matrix",
    labels: Optional[Tuple[str, str]] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (6, 5),
) -> plt.Axes:
    """
    Plot a density matrix as a heatmap.

    Parameters
    ----------
    rho : np.ndarray
        Density matrix (d, d)
    title : str
        Plot title
    labels : Tuple[str, str], optional
        Axis labels (x_label, y_label)
    ax : plt.Axes, optional
        Existing axes
    figsize : Tuple[int, int]
        Figure size if creating new figure

    Returns
    -------
    plt.Axes
        The axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Take real part for visualization
    rho_real = np.real(rho)

    im = ax.imshow(rho_real, cmap='RdBu_r', vmin=-rho_real.max(), vmax=rho_real.max())
    ax.set_title(title)

    if labels is not None:
        ax.set_xlabel(labels[1])
        ax.set_ylabel(labels[0])

    plt.colorbar(im, ax=ax, label='Re($\\rho$)')

    return ax


def plot_atomic_density(
    rho_atom: np.ndarray,
    atom_id: str = "AB",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot two-atomic density matrix.

    Parameters
    ----------
    rho_atom : np.ndarray
        9x9 density matrix for two atoms (|0>,|1>,|e>) each
    atom_id : str
        Identifier for which atom(s)
    ax : plt.Axes, optional
        Existing axes

    Returns
    -------
    plt.Axes
    """
    basis_labels = ['|0_A0>', '|0_A1>', '|0_Ae>',
                   '|1_A0>', '|1_A1>', '|1_Ae>',
                   '|e_A0>', '|e_A1>', '|e_Ae>']

    return plot_density_matrix(
        rho_atom,
        title=f"Atomic Density Matrix ({atom_id})",
        labels=("Basis", "Basis"),
        ax=ax
    )


def plot_fidelity_comparison(
    rho: np.ndarray,
    target_states: dict,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot fidelity with multiple target states.

    Parameters
    ----------
    rho : np.ndarray
        Density matrix
    target_states : dict
        Dictionary {name: target_state_matrix}
    ax : plt.Axes, optional
        Existing axes

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    names = []
    fidelities = []

    for name, target in target_states.items():
        fid = np.real(np.vdot(target.flatten(), rho @ target.flatten()))
        names.append(name)
        fidelities.append(fid)

    bars = ax.bar(names, fidelities, color='steelblue')
    ax.set_ylabel("Fidelity")
    ax.set_title("Fidelity with Target States")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars, fidelities):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom')

    return ax


def plot_bloch_vector(rho_qubit: np.ndarray, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot Bloch sphere representation of a qubit state.

    Parameters
    ----------
    rho_qubit : np.ndarray
        2x2 density matrix (qubit subspace only)
    ax : plt.Axes, optional
        Existing 3D axes

    Returns
    -------
    plt.Axes
        The 3D axes object
    """
    from mpl_toolkits.mplot3d import Axes3D

    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

    # Compute Bloch vector
    # sigma_x, sigma_y, sigma_z
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

    x = np.real(np.trace(rho_qubit @ sigma_x))
    y = np.real(np.trace(rho_qubit @ sigma_y))
    z = np.real(np.trace(rho_qubit @ sigma_z))

    # Draw sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x_sphere = 0.5 * np.outer(np.cos(u), np.sin(v))
    y_sphere = 0.5 * np.outer(np.sin(u), np.sin(v))
    z_sphere = 0.5 * np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(x_sphere, y_sphere, z_sphere, color='gray', alpha=0.1)

    # Draw axes
    ax.plot([-0.5, 0.5], [0, 0], [0, 0], 'k-', lw=1)
    ax.plot([0, 0], [-0.5, 0.5], [0, 0], 'k-', lw=1)
    ax.plot([0, 0], [0, 0], [-0.5, 0.5], 'k-', lw=1)

    # Plot vector
    ax.quiver(0, 0, 0, x, y, z, color='red', linewidth=3, arrow_length_ratio=0.1)

    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-0.5, 0.5])
    ax.set_zlim([-0.5, 0.5])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Bloch Vector: ({x:.3f}, {y:.3f}, {z:.3f})')

    return ax


def plot_concurrence(
    rho: np.ndarray,
    title: str = "Entanglement Measure",
    ax: Optional[plt.Axes] = None,
) -> float:
    """
    Calculate and plot concurrence (for two-qubit states).

    Parameters
    ----------
    rho : np.ndarray
        4x4 or 9x9 density matrix (truncated to qubit subspace)
    title : str
        Plot title
    ax : plt.Axes, optional
        Existing axes

    Returns
    -------
    float
        Concurrence value
    """
    # For now, just compute concurrence numerically
    # This is a simplified version

    # Extract 2x2 block if needed
    if rho.shape[0] > 4:
        # Assume first 4 elements are the qubit subspace
        rho = rho[:4, :4]

    # Compute concurrence using Wootters formula
    # C = max(0, sqrt1 - sqrt2 - sqrt3 - sqrt4)
    # where sqrt_i are eigenvalues of rho * (sigma_y ⊗ sigma_y) * rho* * (sigma_y ⊗ sigma_y)

    sigma_y = np.array([[0, -1j], [1j, 0]])
    sy_sy = np.kron(sigma_y, sigma_y)
    R = rho @ sy_sy @ rho.conj().T @ sy_sy

    evals = np.linalg.eigvals(R)
    evals = np.real(evals)
    evals = np.sort(evals)[::-1]  # Descending

    sqrt_evals = np.sqrt(np.maximum(0, evals))
    C = max(0, sqrt_evals[0] - sqrt_evals[1] - sqrt_evals[2] - sqrt_evals[3])

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

        # Create bar chart
        bars = ax.bar(['Concurrence'], [C], color='steelblue')
        ax.set_ylim(0, 1)
        ax.set_ylabel('Concurrence')
        ax.set_title(title)
        ax.grid(True, axis='y', alpha=0.3)

        # Add value label
        ax.text(0, C + 0.05, f'{C:.4f}', ha='center', va='bottom')

    return C
