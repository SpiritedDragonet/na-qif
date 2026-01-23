# -*- coding: utf-8 -*-
"""
Atom Sim：中性原子量子接口的时间仓MPS仿真

本包提供时间仓MPS仿真的纯TeNPy实现，
用于中性原子量子纠缠协议。

主要导出：
- MPSState：张量网络态容器
- run_dual_atom_emission：双原子发射轨迹
- TimeGrid：时间网格
"""

from .core.mps import MPSState, create_timebin_mps
from .time_grid import TimeGrid
from .simulation import run_dual_atom_emission

__version__ = '0.1.0'

__all__ = [
    # core
    'MPSState',
    'create_timebin_mps',
    # time grid
    'TimeGrid',
    # simulation
    'run_dual_atom_emission',
]
