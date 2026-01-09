"""
Physics Module

Provides gate factories and Kraus channels for the time-bin simulation.
"""

from .gates import (
    emission_gate,
    qfc_gate,
    bs_gate,
    jones_gate,
    jones_gate_from_array,
    swap_gate,
)
from .channels import (
    loss_channel_1517,
    loss_channel_both_subspaces,
    detection_channel,
    detection_povm_single_site,
    detection_channel_two_mode,
    dephasing_channel,
    dephasing_channel_from_rate,
    FiberChannelParams,
)

__all__ = [
    # gates
    'emission_gate',
    'qfc_gate',
    'bs_gate',
    'jones_gate',
    'jones_gate_from_array',
    'swap_gate',
    # channels
    'loss_channel_1517',
    'loss_channel_both_subspaces',
    'detection_channel',
    'detection_povm_single_site',
    'detection_channel_two_mode',
    'dephasing_channel',
    'dephasing_channel_from_rate',
    'FiberChannelParams',
]
