# -*- coding: utf-8 -*-
"""
物理模块

提供时间仓仿真的门工厂和Kraus通道。
"""

from .gates import (
    build_emitter_operators_12d,
    emission_gate,
    qfc_gate,
)
from .channels import (
    kraus_from_collapse_ops,
    loss_channel_both_subspaces,
    loss_channel_1517_single_photon,
    FiberChannelParams,
)

__all__ = [
    # gates
    'build_emitter_operators_12d',
    'emission_gate',
    'qfc_gate',
    # channels
    'kraus_from_collapse_ops',
    'loss_channel_both_subspaces',
    'loss_channel_1517_single_photon',
    'FiberChannelParams',
]
