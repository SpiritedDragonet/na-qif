# -*- coding: utf-8 -*-
"""
福克空间和原子跃迁的基本算符工厂

本模块提供构造产生、湮灭、数算符和原子跃迁算符的函数。
"""

from typing import Tuple, Union
import numpy as np
from .basis import SubSpace, ProductSpace, SUBSPACE_780, SUBSPACE_1517, ATOM_3D


def annihilation_op(space: SubSpace, mode_id: int = 0) -> np.ndarray:
    """
    在福克子空间上构造湮灭算符 a[i]。

    对于占有数元组 (n_0, n_1, ..., n_{M-1}) 且 sum(n) <= n_max 的基，
    模 i 上的湮灭算符作用为：
        a[i] |..., n_i, ...> = sqrt(n_i) |..., n_i - 1, ...>

    Parameters
    ----------
    space : SubSpace
        构造算符的子空间。
        对于780：mode 0=H, 1=V
        对于1517：mode 0=H, 1=V
    mode_id : int
        要湮灭的模式（默认：0）

    Returns
    -------
    np.ndarray
        湮灭算符矩阵，形状为 (space.dim, space.dim)

    Examples
    --------
    >>> a_780_H = annihilation_op(SUBSPACE_780, mode_id=0)  # 780 H模式
    >>> a_1517_V = annihilation_op(SUBSPACE_1517, mode_id=1)  # 1517 V模式
    """
    dim = space.dim

    if space == SUBSPACE_780:
        # 780子空间：vac, H, V
        # mode_id=0 (H): a|H> = |vac>, a|vac> = 0, a|V> = 0
        # mode_id=1 (V): a|V> = |vac>, a|vac> = 0, a|H> = 0
        op = np.zeros((dim, dim), dtype=complex)
        if mode_id == 0:  # H模式
            op[0, 1] = 1.0  # |vac><H|
        elif mode_id == 1:  # V模式
            op[0, 2] = 1.0  # |vac><V|
        else:
            raise ValueError(f"780子空间的 mode_id {mode_id} 无效")
        return op

    elif space == SUBSPACE_1517:
        # 1517子空间：vac, H, V, 2H, 2V, HV
        # 占有数表示：
        # vac: (0,0), H: (1,0), V: (0,1), 2H: (2,0), 2V: (0,2), HV: (1,1)
        # mode_id=0 (H), mode_id=1 (V)
        op = np.zeros((dim, dim), dtype=complex)
        basis_order = [
            (0, 0),  # 0: vac
            (1, 0),  # 1: H
            (0, 1),  # 2: V
            (2, 0),  # 3: 2H
            (0, 2),  # 4: 2V
            (1, 1),  # 5: HV
        ]

        # 构建索引映射：(n_H, n_V) -> index
        idx_map = {occ: i for i, occ in enumerate(basis_order)}

        for i, (nH, nV) in enumerate(basis_order):
            if mode_id == 0:  # H模式
                nH_target = nH - 1
                nV_target = nV
            else:  # V模式
                nH_target = nH
                nV_target = nV - 1

            if nH_target < 0 or nV_target < 0:
                continue

            target = (nH_target, nV_target)
            if target in idx_map:
                j = idx_map[target]
                # 系数为 sqrt(n)，其中 n 是原始占有数
                n = nH if mode_id == 0 else nV
                op[j, i] = np.sqrt(n)

        return op

    else:
        raise ValueError(f"不支持的子空间：{space.name}。"
                        f"请使用 SUBSPACE_780 或 SUBSPACE_1517")


def creation_op(space: SubSpace, mode_id: int = 0) -> np.ndarray:
    """
    在福克子空间上构造产生算符 a^†[i]。

    这是湮灭算符的厄米共轭。

    Parameters
    ----------
    space : SubSpace
        构造算符的子空间
    mode_id : int
        要产生的模式（默认：0）

    Returns
    -------
    np.ndarray
        产生算符矩阵，形状为 (space.dim, space.dim)
    """
    a = annihilation_op(space, mode_id)
    return a.conj().T


def atom_transition(which: str) -> np.ndarray:
    """
    构造原子跃迁算符 S_+ 或 S_-。

    原子能级（3D）：
        |0>: 基态 (m_F = +1)
        |1>: 基态 (m_F = -1)
        |e>: 激发态 (m_F = 0)

    选择定则：
        |e> → |0>: Δm = +1 → σ+ 光子 (S_+ = |0><e|)
        |e> → |1>: Δm = -1 → σ- 光子 (S_- = |1><e|)

    Parameters
    ----------
    which : str
        '+' 表示 S_+，'-' 表示 S_-

    Returns
    -------
    np.ndarray
        跃迁算符矩阵，形状为 (3, 3)

    Examples
    --------
    >>> S_plus = atom_transition('+')  # |0><e|
    >>> S_minus = atom_transition('-')  # |1><e|
    """
    # 基顺序：|0>, |1>, |e>
    op = np.zeros((3, 3), dtype=complex)

    if which == '+':
        # S_+ = |0><e|
        op[0, 2] = 1.0
    elif which == '-':
        # S_- = |1><e|
        op[1, 2] = 1.0
    else:
        raise ValueError(f"which 必须是 '+' 或 '-'，得到 '{which}'")

    return op


def number_op(space: SubSpace, mode_id: int = 0) -> np.ndarray:
    """
    在福克子空间上构造数算符 N = a^† a。

    用于波包提取：<N> 给出光子数期望值。

    对于基为 (vac, H, V, 2H, 2V, HV) 的1517子空间：
        N_H = diag(0, 1, 0, 2, 0, 1)
        N_V = diag(0, 0, 1, 0, 2, 1)

    Parameters
    ----------
    space : SubSpace
        构造算符的子空间
    mode_id : int
        计数光子的模式（默认：0）

    Returns
    -------
    np.ndarray
        数算符矩阵（对角），形状为 (space.dim, space.dim)

    Examples
    --------
    >>> N_780_H = number_op(SUBSPACE_780, mode_id=0)  # 计数780 H光子
    >>> N_1517_H = number_op(SUBSPACE_1517, mode_id=0)  # 计数1517 H光子
    >>> N_1517_V = number_op(SUBSPACE_1517, mode_id=1)  # 计数1517 V光子
    """
    adag = creation_op(space, mode_id)
    a = annihilation_op(space, mode_id)
    return adag @ a
