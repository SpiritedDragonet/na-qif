"""
Test Kraus Channel Completeness

Verifies that Kraus operators satisfy the completeness relation:
sum(K_mu^† K_mu) = I
"""

import numpy as np
from pytest import approx

from atom_sim.physics.channels import (
    loss_channel_1517,
    detection_channel,
    dephasing_channel,
)


def test_loss_kraus_completeness():
    """Test that loss Kraus operators sum to identity."""
    K_list = loss_channel_1517(eta_H=0.9, eta_V=0.9)

    # Sum K^† K
    dim = 6  # 1517 subspace dimension
    completeness = np.zeros((dim, dim), dtype=complex)
    for K in K_list:
        completeness += K.conj().T @ K

    # Should equal identity
    I = np.eye(dim, dtype=complex)
    diff = np.linalg.norm(completeness - I)
    assert diff == approx(0.0, abs=1e-10)

    print(f"Loss Kraus completeness: diff = {diff}")


def test_detection_kraus_completeness():
    """Test that detection Kraus operators sum to identity."""
    K_list, _ = detection_channel(eta_det=0.9, p_dark=0.001)

    # Sum K^† K
    dim = 6  # 1517 subspace dimension
    completeness = np.zeros((dim, dim), dtype=complex)
    for K in K_list:
        completeness += K.conj().T @ K

    # Should equal identity
    I = np.eye(dim, dtype=complex)
    diff = np.linalg.norm(completeness - I)
    assert diff == approx(0.0, abs=1e-10)

    print(f"Detection Kraus completeness: diff = {diff}")


def test_dephasing_kraus_completeness():
    """Test that dephasing Kraus operators sum to identity."""
    K_list = dephasing_channel(p_phi=0.1, dim=3)

    # Sum K^† K
    dim = 3
    completeness = np.zeros((dim, dim), dtype=complex)
    for K in K_list:
        completeness += K.conj().T @ K

    # Should equal identity
    I = np.eye(dim, dtype=complex)
    diff = np.linalg.norm(completeness - I)
    assert diff == approx(0.0, abs=1e-10)

    print(f"Dephasing Kraus completeness: diff = {diff}")


if __name__ == '__main__':
    test_loss_kraus_completeness()
    test_detection_kraus_completeness()
    test_dephasing_kraus_completeness()
    print("All Kraus completeness tests passed!")
