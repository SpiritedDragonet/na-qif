# -*- coding: utf-8 -*-
"""
希尔伯特空间定义和张量积构造

本模块提供定义子空间与构造积空间的类和函数。
"""

from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass(frozen=True)
class SubSpace:
    """
    单个希尔伯特子空间，带有名称和维度。

    Attributes
    ----------
    name : str
        此子空间的标识符（如 '780', '1517', 'atom'）
    dim : int
        此子空间的维度
    basis_labels : Optional[List[str]]
        基态的可选标签（如 ['vac', 'H', 'V']）
    """
    name: str
    dim: int
    basis_labels: Optional[Tuple[str, ...]] = None

    def __post_init__(self):
        if self.basis_labels is not None and len(self.basis_labels) != self.dim:
            raise ValueError(
                f"基标签数量 ({len(self.basis_labels)}) "
                f"必须与维度 ({self.dim}) 匹配"
            )


@dataclass(frozen=True)
class ProductSpace:
    """
    子空间的张量积。

    表示 H = H_1 ⊗ H_2 ⊗ ... ⊗ H_n

    Attributes
    ----------
    subspaces : Tuple[SubSpace, ...]
        张量积中的子空间
    """
    subspaces: Tuple[SubSpace, ...]

    def __post_init__(self):
        if not self.subspaces:
            raise ValueError("ProductSpace必须至少有一个子空间")

    @property
    def dim(self) -> int:
        """积空间的总维度。"""
        result = 1
        for s in self.subspaces:
            result *= s.dim
        return result

    @property
    def num_factors(self) -> int:
        """积空间中的子空间数量。"""
        return len(self.subspaces)

    def subspace_index(self, name: str) -> int:
        """通过名称获取子空间的索引。"""
        for i, s in enumerate(self.subspaces):
            if s.name == name:
                return i
        raise ValueError(f"未找到子空间 '{name}'")

    def subspace_dims(self) -> Tuple[int, ...]:
        """每个子空间的维度元组。"""
        return tuple(s.dim for s in self.subspaces)

    def subspace_dim(self, name: str) -> int:
        """通过名称获取特定子空间的维度。"""
        return self.subspaces[self.subspace_index(name)].dim


# 本项目的预定义子空间
# 这些与 README 中的物理模型对应

ATOM_4D = SubSpace('atom', 4, ('|0>', '|1>', '|e>', '|u>'))  # |u>: 5S1/2, F=1, m_F=0
ATOM_A = SubSpace('atom_A', 4, ('|0>', '|1>', '|e>', '|u>'))
ATOM_B = SubSpace('atom_B', 4, ('|0>', '|1>', '|e>', '|u>'))

SUBSPACE_780 = SubSpace('780', 3, ('vac', 'H', 'V'))
SUBSPACE_1517 = SubSpace('1517', 6, ('vac', 'H', 'V', '2H', '2V', 'HV'))

# 系统格点：两个原子
SYSTEM_SPACE = ProductSpace((ATOM_A, ATOM_B))  # 16D

# Bin格点：780 x 1517
BIN_SPACE = ProductSpace((SUBSPACE_780, SUBSPACE_1517))  # 18D


def get_bin_space() -> ProductSpace:
    """获取标准的18D bin空间（780 x 1517）。"""
    return BIN_SPACE


def get_system_space() -> ProductSpace:
    """获取标准的9D系统空间（atom_A x atom_B）。"""
    return SYSTEM_SPACE
