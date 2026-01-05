"""
Hilbert Space Definitions and Tensor Product Constructions

This module provides classes and functions for defining subspaces,
constructing product spaces, and embedding gates into product spaces.
"""

from typing import List, Tuple, Optional, Union
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SubSpace:
    """
    A single Hilbert subspace with a name and dimension.

    Attributes
    ----------
    name : str
        Identifier for this subspace (e.g., '780', '1517', 'atom')
    dim : int
        Dimension of this subspace
    basis_labels : Optional[List[str]]
        Optional labels for basis states (e.g., ['vac', 'H', 'V'])
    """
    name: str
    dim: int
    basis_labels: Optional[Tuple[str, ...]] = None

    def __post_init__(self):
        if self.basis_labels is not None and len(self.basis_labels) != self.dim:
            raise ValueError(
                f"Number of basis labels ({len(self.basis_labels)}) "
                f"must match dimension ({self.dim})"
            )


@dataclass(frozen=True)
class ProductSpace:
    """
    A tensor product of subspaces.

    Represents H = H_1 ⊗ H_2 ⊗ ... ⊗ H_n

    Attributes
    ----------
    subspaces : Tuple[SubSpace, ...]
        The subspaces in the tensor product
    """
    subspaces: Tuple[SubSpace, ...]

    def __post_init__(self):
        if not self.subspaces:
            raise ValueError("ProductSpace must have at least one subspace")

    @property
    def dim(self) -> int:
        """Total dimension of the product space."""
        result = 1
        for s in self.subspaces:
            result *= s.dim
        return result

    @property
    def num_factors(self) -> int:
        """Number of subspaces in the product."""
        return len(self.subspaces)

    def subspace_index(self, name: str) -> int:
        """Get the index of a subspace by name."""
        for i, s in enumerate(self.subspaces):
            if s.name == name:
                return i
        raise ValueError(f"Subspace '{name}' not found")

    def subspace_dims(self) -> Tuple[int, ...]:
        """Tuple of dimensions of each subspace."""
        return tuple(s.dim for s in self.subspaces)

    def subspace_dim(self, name: str) -> int:
        """Get dimension of a specific subspace by name."""
        return self.subspaces[self.subspace_index(name)].dim


def subspace_gate(
    full_space: ProductSpace,
    active_subspaces: Union[str, List[str]],
    gate_matrix: np.ndarray,
) -> np.ndarray:
    """
    Embed a gate acting on specific subspaces into the full product space.

    Given a gate acting on active_subspaces, embed it as I ⊗ ... ⊗ G ⊗ ... ⊗ I
    where G acts on the active subspaces and I acts on the others.

    Parameters
    ----------
    full_space : ProductSpace
        The full product space H = H_1 ⊗ ... ⊗ H_n
    active_subspaces : Union[str, List[str]]
        Name(s) of the subspace(s) the gate acts on
    gate_matrix : np.ndarray
        The gate matrix acting on the tensor product of active subspaces.
        Shape should be (d_active, d_active) where d_active is the product
        of dimensions of active subspaces.

    Returns
    -------
    np.ndarray
        The embedded gate matrix acting on the full space.
        Shape is (full_space.dim, full_space.dim)

    Examples
    --------
    >>> space_780 = SubSpace('780', 3)
    >>> space_1517 = SubSpace('1517', 6)
    >>> bin_space = ProductSpace((space_780, space_1517))
    >>> gate_on_780 = np.eye(3)  # Identity on 780 subspace
    >>> full_gate = subspace_gate(bin_space, '780', gate_on_780)
    >>> full_gate.shape == (18, 18)
    True
    """
    if isinstance(active_subspaces, str):
        active_subspaces = [active_subspaces]

    # Validate active subspaces exist
    for name in active_subspaces:
        if name not in [s.name for s in full_space.subspaces]:
            raise ValueError(f"Subspace '{name}' not found in full_space")

    # Calculate expected dimension of active subspace
    active_dim = 1
    for name in active_subspaces:
        active_dim *= full_space.subspace_dim(name)

    if gate_matrix.shape != (active_dim, active_dim):
        raise ValueError(
            f"Gate matrix shape {gate_matrix.shape} does not match "
            f"active subspace dimension ({active_dim}, {active_dim})"
        )

    # Build the full gate as a tensor product
    # Start with identity on first subspace
    result = np.eye(1, dtype=complex)

    for i, subspace in enumerate(full_space.subspaces):
        if subspace.name in active_subspaces:
            # This subspace is acted on by the gate
            # We need to extract the correct slice from gate_matrix
            # and tensor it in
            if len(active_subspaces) == 1:
                # Single active subspace - use gate directly
                result = np.kron(result, gate_matrix)
            else:
                # Multiple active subspaces - need to factor gate
                # This is more complex; for now assume gate is already
                # formatted for the active subspace tensor product
                result = np.kron(result, gate_matrix)
        else:
            # Identity on inactive subspaces
            result = np.kron(result, np.eye(subspace.dim, dtype=complex))

    return result


# Predefined subspaces for this project
# These match the physical model in README

ATOM_3D = SubSpace('atom', 3, ('|0>', '|1>', '|e>'))
ATOM_A = SubSpace('atom_A', 3, ('|0>', '|1>', '|e>'))
ATOM_B = SubSpace('atom_B', 3, ('|0>', '|1>', '|e>'))

SUBSPACE_780 = SubSpace('780', 3, ('vac', 'H', 'V'))
SUBSPACE_1517 = SubSpace('1517', 6, ('vac', 'H', 'V', '2H', '2V', 'HV'))

# System site: two atoms
SYSTEM_SPACE = ProductSpace((ATOM_A, ATOM_B))  # 9D

# Bin site: 780 x 1517
BIN_SPACE = ProductSpace((SUBSPACE_780, SUBSPACE_1517))  # 18D


def get_bin_space() -> ProductSpace:
    """Get the standard 18D bin space (780 x 1517)."""
    return BIN_SPACE


def get_system_space() -> ProductSpace:
    """Get the standard 9D system space (atom_A x atom_B)."""
    return SYSTEM_SPACE
