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
    loss_channel,
    loss_channel_1517,
    detection_channel,
    detection_channel_two_mode,
    dephasing_channel,
    dephasing_channel_from_rate,
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
    'loss_channel',
    'loss_channel_1517',
    'detection_channel',
    'detection_channel_two_mode',
    'dephasing_channel',
    'dephasing_channel_from_rate',
]
