"""
Hilbert Space Module

Provides classes and functions for defining Hilbert spaces,
operators, and site types for the time-bin MPS simulation.
"""

from .basis import (
    SubSpace,
    ProductSpace,
    subspace_gate,
    ATOM_3D,
    ATOM_A,
    ATOM_B,
    SUBSPACE_780,
    SUBSPACE_1517,
    SYSTEM_SPACE,
    BIN_SPACE,
    get_bin_space,
    get_system_space,
)
from .operators import (
    annihilation_op,
    creation_op,
    atom_transition,
    number_op,
)
from .sites import (
    FiniteDimSite,
    bin_site,
    atom_site,
    system_site,
)

__all__ = [
    # basis
    'SubSpace',
    'ProductSpace',
    'subspace_gate',
    'ATOM_3D',
    'ATOM_A',
    'ATOM_B',
    'SUBSPACE_780',
    'SUBSPACE_1517',
    'SYSTEM_SPACE',
    'BIN_SPACE',
    'get_bin_space',
    'get_system_space',
    # operators
    'annihilation_op',
    'creation_op',
    'atom_transition',
    'number_op',
    # sites
    'FiniteDimSite',
    'bin_site',
    'atom_site',
    'system_site',
]
