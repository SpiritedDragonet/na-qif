# -*- coding: utf-8 -*-
"""
物理模块

提供时间仓仿真的门工厂和Kraus通道。
"""

from .gates import (
    emission_gate,
    qfc_gate,
    jones_gate,
)
from .channels import (
    loss_channel_both_subspaces,
    loss_channel_1517_single_photon,
    FiberChannelParams,
)

__all__ = [
    # gates
    'emission_gate',
    'qfc_gate',
    'jones_gate',
    # channels
    'loss_channel_both_subspaces',
    'loss_channel_1517_single_photon',
    'FiberChannelParams',
]
