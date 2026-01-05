"""
Custom Site Types for MPS Construction

This module provides FiniteDimSite, a generic d-dimensional site type
that is NOT based on TeNPy's BosonSite (which has incompatible semantics
for multi-mode Fock spaces with H/V polarization).
"""

from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np


@dataclass
class FiniteDimSite:
    """
    A generic finite-dimensional site for MPS construction.

    Unlike TeNPy's BosonSite which assumes single-mode bosonic occupation,
    this site can represent any finite-dimensional space with custom operators.

    Parameters
    ----------
    dim : int
        Hilbert space dimension
    name : str
        Identifier for this site type
    basis_labels : Optional[Tuple[str, ...]]
        Optional labels for basis states
    op_dict : Optional[Dict[str, np.ndarray]]
        Dictionary of operators acting on this site.
        Keys are operator names, values are (dim, dim) matrices.

    Attributes
    ----------
    dim : int
        Hilbert space dimension
    name : str
        Site identifier
    basis_labels : Tuple[str, ...]
        Labels for basis states
    operators : Dict[str, np.ndarray]
        Operator dictionary

    Examples
    --------
    >>> # Create a site for 780 photon (3D: vac, H, V)
    >>> site_780 = FiniteDimSite(3, name='780', basis_labels=('vac', 'H', 'V'))
    >>>
    >>> # Create a site with custom operators
    >>> import numpy as np
    >>> ops = {'n': np.diag([0, 1, 0]), 'a': np.array([[0,1,0],[0,0,0],[0,0,0]])}
    >>> site = FiniteDimSite(3, name='custom', op_dict=ops)
    """
    dim: int
    name: str = 'site'
    basis_labels: Optional[Tuple[str, ...]] = None
    op_dict: Dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        # Convert None to empty tuple for immutability
        if self.basis_labels is None:
            object.__setattr__(self, 'basis_labels', tuple(f'|{i}>' for i in range(self.dim)))

        # Validate basis labels
        if len(self.basis_labels) != self.dim:
            raise ValueError(
                f"Number of basis labels ({len(self.basis_labels)}) "
                f"must match dimension ({self.dim})"
            )

        # Validate operator dimensions
        for op_name, op_mat in self.op_dict.items():
            if op_mat.shape != (self.dim, self.dim):
                raise ValueError(
                    f"Operator '{op_name}' has shape {op_mat.shape} "
                    f"but expected ({self.dim}, {self.dim})"
                )

    def get_op(self, name: str) -> np.ndarray:
        """
        Get an operator by name.

        Parameters
        ----------
        name : str
            Operator name

        Returns
        -------
        np.ndarray
            Operator matrix

        Raises
        ------
        KeyError
            If operator name not found
        """
        if name not in self.op_dict:
            raise KeyError(f"Operator '{name}' not found. Available: {list(self.op_dict.keys())}")
        return self.op_dict[name]

    def add_op(self, name: str, op: np.ndarray) -> None:
        """
        Add an operator to the site.

        Parameters
        ----------
        name : str
            Operator name
        op : np.ndarray
            Operator matrix with shape (dim, dim)
        """
        if op.shape != (self.dim, self.dim):
            raise ValueError(
                f"Operator '{name}' has shape {op.shape} "
                f"but expected ({self.dim}, {self.dim})"
            )
        self.op_dict[name] = op

    def has_op(self, name: str) -> bool:
        """Check if an operator exists."""
        return name in self.op_dict

    def __repr__(self) -> str:
        return f"FiniteDimSite(name='{self.name}', dim={self.dim})"


# Convenience constructors for common sites

def bin_site() -> FiniteDimSite:
    """
    Create a standard 18D bin site (780 x 1517).

    Returns
    -------
    FiniteDimSite
        18-dimensional site with basis labels for 780 and 1517 product states
    """
    # Generate combined basis labels
    labels_780 = ('vac', 'H', 'V')
    labels_1517 = ('vac', 'H', 'V', '2H', '2V', 'HV')

    combined_labels = []
    for l7 in labels_780:
        for l15 in labels_1517:
            combined_labels.append(f'{l7}_x_{l15}')

    return FiniteDimSite(
        dim=18,
        name='bin',
        basis_labels=tuple(combined_labels)
    )


def atom_site(atom_id: str = 'A') -> FiniteDimSite:
    """
    Create a standard 3D atomic site.

    Parameters
    ----------
    atom_id : str
        Identifier for which atom ('A' or 'B')

    Returns
    -------
    FiniteDimSite
        3-dimensional site with basis |0>, |1>, |e>
    """
    return FiniteDimSite(
        dim=3,
        name=f'atom_{atom_id}',
        basis_labels=('|0>', '|1>', '|e>')
    )


def system_site() -> FiniteDimSite:
    """
    Create a 9D system site (two atoms combined).

    Returns
    -------
    FiniteDimSite
        9-dimensional site representing atom_A ⊗ atom_B
    """
    basis = []
    labels = ('|0>', '|1>', '|e>')
    for lA in labels:
        for lB in labels:
            basis.append(f'{lA}_A⊗{lB}_B')

    return FiniteDimSite(
        dim=9,
        name='system',
        basis_labels=tuple(basis)
    )
