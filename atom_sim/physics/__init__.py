# -*- coding: utf-8 -*-
"""
物理模块

提供时间仓仿真的门工厂和Kraus通道。
"""

from .gates import (
    emission_gate,
    qfc_gate,
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
