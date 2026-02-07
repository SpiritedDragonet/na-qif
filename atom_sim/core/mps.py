# -*- coding: utf-8 -*-
"""
MPS态容器（TeNPy后端）

================================================================================
TeNPy的MPS封装，用于时间仓量子仿真。

核心设计：
-----------
- 两格点操作使用 get_theta + set_svd_theta 进行局域更新
- Kraus操作完全局域化（不需要 canonical_form() 扫描）
- 约化密度矩阵使用 get_rho_segment() 确保正确性

依赖：
-------------
pip install physics-tenpy

参考文献：
-----------
[1] TeNPy MPS: https://tenpy.readthedocs.io/en/stable/reference/tenpy.networks.mps.MPS.html
[2] TeNPy Array: https://tenpy.readthedocs.io/en/stable/reference/tenpy.linalg.np_conserved.html
================================================================================
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union
import numpy as np

# TeNPy imports
from tenpy.networks.mps import MPS as TeNPy_MPS
from tenpy.networks.site import BosonSite
from tenpy.linalg.np_conserved import Array


_TENPY_NOTICE_PRINTED = False


def _emit_tenpy_notice_once() -> None:
    global _TENPY_NOTICE_PRINTED
    if _TENPY_NOTICE_PRINTED:
        return
    print("[tenpy] 已启用 TeNPy 张量网络后端，正在进行 MPS 运算")
    _TENPY_NOTICE_PRINTED = True


class MPSState:
    """
    使用TeNPy的矩阵积态。

    Parameters
    ----------
    local_dims : List[int]
        局域希尔伯特空间维度
    init_state : Optional[Union[List[int], np.ndarray]]
        - None: 真空态 |0>...|0>
        - List[int]: 直积态
        - np.ndarray: 完整波函数（使用 MPS.from_full）
    max_bond : int
        截断的最大键维度
    """

    def __init__(
        self,
        local_dims: List[int],
        init_state: Optional[Union[List[int], np.ndarray]] = None,
        max_bond: int = 100,
    ):
        self.L = len(local_dims)
        self.d = local_dims
        self.max_bond = max_bond

        # 创建TeNPy格点（玻色型，无电荷守恒）
        sites = [BosonSite(dim - 1, None) for dim in self.d]
        unit_cell_width = len(sites)
        _emit_tenpy_notice_once()

        # 根据init_state类型初始化MPS
        if init_state is None:
            # 真空态 |0>...|0>
            init_labels = ['0'] * self.L
            self._mps = TeNPy_MPS.from_product_state(
                sites,
                init_labels,
                bc='finite',
                form='B',
                unit_cell_width=unit_cell_width,
            )
        elif isinstance(init_state, list):
            # 从基指标的直积态
            init_labels = [str(s) for s in init_state]
            self._mps = TeNPy_MPS.from_product_state(
                sites,
                init_labels,
                bc='finite',
                form='B',
                unit_cell_width=unit_cell_width,
            )
        elif isinstance(init_state, np.ndarray):
            # 完整波函数 - 使用TeNPy的from_full
            psi_reshaped = init_state.reshape(self.d + [1] * (self.L - len(self.d)))
            psi_array = Array.from_ndarray_trivial(psi_reshaped, labels=[f'p{i}' for i in range(self.L)])
            self._mps = TeNPy_MPS.from_full(
                psi_array,
                sites,
                bc='finite',
                form='B',
                unit_cell_width=unit_cell_width,
            )
        else:
            raise ValueError(f"Invalid init_state: {type(init_state)}")

        self._mps.chi_max = self.max_bond

    # ========================================================================
    # 底层局域更新（避免 canonical_form 扫描）
    # ========================================================================

    def _apply_two_site_op_local(
        self,
        i: int,
        op: Array,
        truncate: bool = True,
        normalize: bool = False,
    ) -> None:
        """
        使用局域更新应用两格点算符（get_theta + set_svd_theta）。

        通过只更新局域键来避免 canonical_form() 扫描。

        Parameters
        ----------
        i : int
            左格点索引
        op : Array
            带有标签 ['p0', 'p1', 'p0*', 'p1*'] 的TeNPy Array
        truncate : bool
            是否截断键维度
        normalize : bool
            若为True，应用后归一化（用于Kraus结果）
        """
        # ------------------------------------------------------------------
        # 核心思想：
        #   仅在局部 (i,i+1) 上做更新，避免全链 canonical sweep。
        #   等价于 TEBD 的“局部门 + SVD 截断”流程：
        #     1) 取出两点张量 theta
        #     2) 在物理腿上施加 op
        #     3) 合并腿做 SVD，写回左右张量
        # ------------------------------------------------------------------
        # 获取两格点的theta：腿为 (vL, p0, p1, vR)
        theta = self._mps.get_theta(i, n=2)

        # 转换为numpy进行收缩（避免LegCharge问题）
        theta_np = theta.to_ndarray()  # Shape: (chiL, d0, d1, chiR)
        op_np = op.to_ndarray()  # Shape: (d0, d1, d0, d1)

        # 收缩: op @ theta，其中 op[i,j,k,l] 作用于theta的物理腿
        # 结果: theta_new[a, i, j, b] = sum_{k,l} op[i, j, k, l] * theta[a, k, l, b]
        theta_new_np = np.einsum('ijkl,aklb->aijb', op_np, theta_np)

        # 转换回TeNPy Array
        theta_new = Array.from_ndarray_trivial(theta_new_np, labels=['vL', 'p0', 'p1', 'vR'])

        # 合并腿以进行SVD: (vL.p0) 和 (p1.vR)
        theta_combined = theta_new.combine_legs(
            [['vL', 'p0'], ['p1', 'vR']],
            new_axes=[0, 1],
            qconj=[+1, -1]
        )

        # 设置截断参数
        trunc_params = None
        if truncate:
            trunc_params = {'chi_max': self.max_bond, 'svd_min': 1e-13}

        # 通过SVD写回
        self._mps.set_svd_theta(i, theta_combined, trunc_par=trunc_params)

        # 若需要则归一化（用于Kraus分支结果）
        if normalize:
            self._mps.norm = 1.0

    # ========================================================================
    # 门操作
    # ========================================================================

    # ========================================================================
    # 核心API方法（仅局域，无 canonical_form 扫描）
    # ========================================================================

    def apply_bond_op(
        self,
        i: int,
        op: np.ndarray,
        truncate: bool = True,
    ) -> None:
        """
        通过局域更新应用两格点算符（幺正或非幺正）。

        使用 get_theta + set_svd_theta 来避免 canonical_form() 扫描。
        这是应用任何两格点门的主要方法。

        Parameters
        ----------
        i : int
            左格点索引（作用于格点i和i+1）
        op : np.ndarray
            算符矩阵，形状为 (d1*d2, d1*d2) 或 (d1, d2, d1, d2)
        truncate : bool
            是否截断键维度
        """
        d1, d2 = self.d[i], self.d[i + 1]

        # ------------------------------------------------------------------
        # 将任意 2-site 算符统一整理成 (d1,d2,d1,d2) 形式，
        # 以便与 TeNPy 的标签约定对齐。
        # ------------------------------------------------------------------
        # 重塑为4D: (d1, d2, d1, d2)
        op = np.asarray(op)
        if op.ndim == 2:
            op = op.reshape(d1 * d2, d1 * d2)
        op_4d = op.reshape(d1, d2, d1, d2)

        # 创建带正确标签的TeNPy Array
        op_arr = Array.from_ndarray_trivial(op_4d, labels=['p0', 'p1', 'p0*', 'p1*'])

        # 通过局域更新应用
        self._apply_two_site_op_local(i, op_arr, truncate=truncate, normalize=False)

    def apply_kraus_one_site(
        self,
        site: int,
        kraus_ops: List[np.ndarray],
        rng: Optional[np.random.Generator] = None,
    ) -> int:
        """
        通过量子轨迹应用单格点Kraus信道。

        采样一个Kraus算符并归一化后应用。

        Parameters
        ----------
        site : int
            格点索引
        kraus_ops : List[np.ndarray]
            Kraus算符列表，每个形状为 (d, d)
        rng : np.random.Generator, optional
            随机数生成器

        Returns
        -------
        int
            采样的Kraus算符索引
        """
        # ------------------------------------------------------------------
        # 单格点量子轨迹 (quantum trajectory)：
        #   1) 对每个 Kraus K_mu 计算概率 p_mu = ||K_mu |psi>||^2
        #   2) 按 p_mu 采样一个分支
        #   3) 归一化并写回该格点的张量
        #
        # 这等价于对 CPTP 信道进行一次“测量记录”的随机展开。
        # ------------------------------------------------------------------
        if rng is None:
            rng = np.random.default_rng()

        d = self.d[site]

        # 获取当前格点张量
        theta = self._mps.get_theta(site, n=1)  # Shape: (chiL, d, chiR)
        theta_np = theta.to_ndarray()

        # 计算每个Kraus算符的概率
        probs = []
        thetas_mu = []

        for K in kraus_ops:
            K = np.asarray(K).reshape(d, d)
            # 应用K: K @ theta（在物理指标上收缩）
            K_theta = np.einsum('ij,ajb->aib', K, theta_np)
            p_mu = np.linalg.norm(K_theta) ** 2
            probs.append(p_mu)
            thetas_mu.append(K_theta)

        # 归一化并采样
        probs = np.array(probs)
        p_total = np.sum(probs)

        if p_total < 1e-15:
            raise ValueError("总概率接近零 - Kraus算符可能无效")

        probs = probs / p_total
        mu = rng.choice(len(kraus_ops), p=probs)

        # 从选中分支创建归一化的theta
        theta_selected = thetas_mu[mu] / np.sqrt(probs[mu] * p_total)

        # 转换为TeNPy Array并直接写回格点张量
        # 对于单格点操作，直接设置B张量
        theta_arr = Array.from_ndarray_trivial(theta_selected, labels=['vL', 'p', 'vR'])

        # 直接设置格点张量（单格点不需要SVD）
        self._mps.set_B(site, theta_arr, form='Th')

        return mu

    def canonicalize(self, renormalize: bool = True) -> None:
        """显式执行一次全链规范化。"""
        self._mps.canonical_form_finite(renormalize=renormalize)

    def swap_sites(self, i: int) -> None:
        """
        交换相邻格点i和i+1。

        这会同时交换张量索引和局域维度（self.d），
        使得后续的 apply_bond_op 调用使用正确的维度。

        Parameters
        ----------
        i : int
            左格点索引（交换i和i+1）
        """
        # ------------------------------------------------------------------
        # 这一步非常关键：
        #   - TeNPy 的 swap_sites 会更新张量，但不会自动更新我们维护的 self.d。
        #   - 若不同步 self.d，会导致后续门的维度错配。
        # ------------------------------------------------------------------
        trunc_params = {'chi_max': self.max_bond, 'svd_min': 1e-13}
        self._mps.swap_sites(i, trunc_par=trunc_params)

        # 关键修复：同时更新self.d数组，交换两个位置的局域维度
        # 这是apply_bond_op正确计算维度所必需的
        self.d[i], self.d[i + 1] = self.d[i + 1], self.d[i]

    # ========================================================================
    # 态提取
    # ========================================================================

    def get_reduced_density(self, sites: List[int]) -> np.ndarray:
        """
        获取指定格点的约化密度矩阵。

        使用TeNPy的get_rho_segment()，它能正确处理
        施密特权重和规范条件。
        """
        # get_rho_segment 会自动包含左右环境的施密特系数，
        # 比直接收缩 B 张量更稳健（尤其在非规范态时）。
        rho_array = self._mps.get_rho_segment(sites)
        return rho_array.to_ndarray()

    # ========================================================================
    # 属性和工具方法
    # ========================================================================

    @property
    def chi(self) -> List[int]:
        """键维度。"""
        return self._mps.chi.copy()

    def norm(self) -> float:
        """获取态的模长。"""
        return float(self._mps.norm)

    def get_bond_dimensions(self) -> List[int]:
        """获取所有键维度列表，等价于 self.chi。"""
        return self.chi

    def copy(self) -> 'MPSState':
        """创建深拷贝。"""
        new_state = MPSState(self.d.copy(), max_bond=self.max_bond)
        new_state._mps = self._mps.copy()
        return new_state

    def __repr__(self) -> str:
        """字符串表示。"""
        chi_str = str(self.get_bond_dimensions())
        return f"MPSState(L={self.L}, d={self.d}, chi={chi_str})"


def compute_joint_arrival_probabilities(
    state: MPSState,
    n_bins: int,
    bin_start: int,
    proj_A: Tuple[np.ndarray, np.ndarray, np.ndarray],
    proj_B: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> Tuple[float, float, float, float, float]:
    """
    计算两臂联合到达统计：p(1,1), p(2,0), p(0,2)。

    返回顺序：
      p_arrive, p_arrive_11, p_arrive_20, p_arrive_02, p_arrive_same_arm
    """
    pi0_a, pi1_a, pi2_a = proj_A
    pi0_b, pi1_b, pi2_b = proj_B
    bin_dim = state.d[bin_start]

    w_bin_a = np.zeros((3, 3, bin_dim, bin_dim), dtype=complex)
    w_bin_a[0, 0] = pi0_a
    w_bin_a[0, 1] = pi1_a
    w_bin_a[0, 2] = pi2_a
    w_bin_a[1, 1] = pi0_a
    w_bin_a[1, 2] = pi1_a
    w_bin_a[2, 2] = pi0_a

    w_bin_b = np.zeros((3, 3, bin_dim, bin_dim), dtype=complex)
    w_bin_b[0, 0] = pi0_b
    w_bin_b[0, 1] = pi1_b
    w_bin_b[0, 2] = pi2_b
    w_bin_b[1, 1] = pi0_b
    w_bin_b[1, 2] = pi1_b
    w_bin_b[2, 2] = pi0_b

    counter_dim = 9
    w_bin_a_joint = np.zeros((counter_dim, counter_dim, bin_dim, bin_dim), dtype=complex)
    w_bin_b_joint = np.zeros((counter_dim, counter_dim, bin_dim, bin_dim), dtype=complex)
    for n_a in range(3):
        for n_a_next in range(n_a, 3):
            for n_b in range(3):
                idx = n_a * 3 + n_b
                idx_next = n_a_next * 3 + n_b
                w_bin_a_joint[idx, idx_next] = w_bin_a[n_a, n_a_next]
        for n_b in range(3):
            for n_b_next in range(n_b, 3):
                idx = n_a * 3 + n_b
                idx_next = n_a * 3 + n_b_next
                w_bin_b_joint[idx, idx_next] = w_bin_b[n_b, n_b_next]

    bin_sites = set()
    for n in range(n_bins):
        site_a = bin_start + 2 * n
        site_b = bin_start + 2 * n + 1
        if site_b >= state.L:
            raise ValueError(f"n_bins={n_bins} 超出MPS长度 {state.L}")
        bin_sites.add(site_a)
        bin_sites.add(site_b)

    env = np.zeros((counter_dim, 1, 1), dtype=complex)
    env[0, 0, 0] = 1.0
    w_identity_cache = {}
    for site in range(state.L):
        b_tensor = state._mps.get_B(site, form='B').to_ndarray()
        bc_tensor = b_tensor.conj()

        if site in bin_sites:
            offset = site - bin_start
            if offset < 0:
                raise ValueError(f"Unexpected bin-site index mapping for site={site}")
            pair_idx = offset // 2
            if pair_idx >= n_bins:
                raise ValueError(f"Unexpected bin-site index mapping for site={site}")
            weight = w_bin_a_joint if (offset % 2) == 0 else w_bin_b_joint
        else:
            dim = b_tensor.shape[1]
            if dim not in w_identity_cache:
                w_id = np.zeros((counter_dim, counter_dim, dim, dim), dtype=complex)
                eye = np.eye(dim, dtype=complex)
                for n_a in range(3):
                    for n_b in range(3):
                        idx = n_a * 3 + n_b
                        w_id[idx, idx] = eye
                w_identity_cache[dim] = w_id
            weight = w_identity_cache[dim]
        env = np.einsum('aij,ipk,jql,abpq->bkl', env, b_tensor, bc_tensor, weight, optimize=True)

    p_arrive_11 = float(env[4, 0, 0].real)
    p_arrive_20 = float(env[6, 0, 0].real)
    p_arrive_02 = float(env[2, 0, 0].real)
    p_arrive_same_arm = max(0.0, p_arrive_20 + p_arrive_02)
    p_arrive = max(0.0, p_arrive_11 + p_arrive_20 + p_arrive_02)
    return p_arrive, p_arrive_11, p_arrive_20, p_arrive_02, p_arrive_same_arm


@dataclass
class DetectionContractionEngine:
    """
    双点击 POVM 收缩引擎。

    将按 (atomA,atomB),(A1,B1),... 分组后的 MPS 收缩逻辑集中到核心层，
    避免在中层 `detection.py` 中堆叠大量张量收缩细节。
    """

    b_list: List[np.ndarray]
    bc_list: List[np.ndarray]
    e_no_list: List[np.ndarray]
    zero_effect: np.ndarray
    detector_order_fn: Callable[[List[str]], Tuple[str, ...]]
    n_bins: int
    dim_atom: int
    right_envs: List[np.ndarray]
    left_envs_identity: List[np.ndarray]
    qubit_indices: List[int]
    _left_envs_qubit: Optional[List[List[List[np.ndarray]]]] = None

    @staticmethod
    def _prepare_grouped_pairs(state: MPSState) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        psi = state._mps.copy()
        if psi.L % 2 != 0:
            raise ValueError("MPS sites 数量必须为偶数，才能按 (atomA,atomB),(A1,B1),... 分组")
        psi.group_sites(n=2)
        psi.canonical_form_finite(renormalize=True)
        psi.norm = 1.0
        b_list_local = []
        bc_list_local = []
        for idx in range(psi.L):
            b_tensor = psi.get_B(idx, form='B').to_ndarray()
            b_list_local.append(b_tensor)
            bc_list_local.append(b_tensor.conj())
        return b_list_local, bc_list_local

    @staticmethod
    def _apply_env_left(
        b_tensor: np.ndarray,
        bc_tensor: np.ndarray,
        op: np.ndarray,
        env_left: np.ndarray,
    ) -> np.ndarray:
        return np.einsum('ij,ipk,jql,pq->kl', env_left, b_tensor, bc_tensor, op, optimize=True)

    @staticmethod
    def _apply_env_right(
        b_tensor: np.ndarray,
        bc_tensor: np.ndarray,
        op: np.ndarray,
        env_right: np.ndarray,
    ) -> np.ndarray:
        return np.einsum('ipk,jql,pq,kl->ij', b_tensor, bc_tensor, op, env_right, optimize=True)

    @classmethod
    def from_mps(
        cls,
        state: MPSState,
        n_bins: int,
        e_no_list: List[np.ndarray],
        zero_effect: np.ndarray,
        detector_order_fn: Callable[[List[str]], Tuple[str, ...]],
    ) -> 'DetectionContractionEngine':
        b_list, bc_list = cls._prepare_grouped_pairs(state)
        grouped_bins = len(b_list) - 1
        if grouped_bins != n_bins:
            raise ValueError(f"n_bins={n_bins} 与分组后bin数量 {grouped_bins} 不一致")

        dim_atom = b_list[0].shape[1]
        if dim_atom != 16:
            raise ValueError(f"Atom pair site dimension {dim_atom} != 16")

        single_dim = int(round(np.sqrt(dim_atom)))
        if single_dim * single_dim != dim_atom:
            raise ValueError(f"Unexpected atom-pair dimension: {dim_atom}")
        qubit_indices = [
            0 * single_dim + 0,
            0 * single_dim + 1,
            1 * single_dim + 0,
            1 * single_dim + 1,
        ]

        dummy = cls(
            b_list=b_list,
            bc_list=bc_list,
            e_no_list=e_no_list,
            zero_effect=zero_effect,
            detector_order_fn=detector_order_fn,
            n_bins=n_bins,
            dim_atom=dim_atom,
            right_envs=[],
            left_envs_identity=[],
            qubit_indices=qubit_indices,
        )
        dummy.right_envs = dummy.build_right_envs()
        atom_identity = np.eye(dim_atom, dtype=complex)
        dummy.left_envs_identity = dummy.build_left_envs(atom_identity)
        return dummy

    def order_detectors(self, detectors: List[str]) -> Tuple[str, ...]:
        return self.detector_order_fn(detectors)

    def build_left_envs(self, atom_op: np.ndarray) -> List[np.ndarray]:
        length = len(self.b_list)
        left_envs = [None] * (length + 1)
        left_envs[0] = np.array([[1.0 + 0.0j]])
        left_envs[1] = self._apply_env_left(
            self.b_list[0],
            self.bc_list[0],
            atom_op,
            left_envs[0],
        )
        for site in range(1, length):
            bin_idx = site - 1
            left_envs[site + 1] = self._apply_env_left(
                self.b_list[site],
                self.bc_list[site],
                self.e_no_list[bin_idx],
                left_envs[site],
            )
        return left_envs

    def build_right_envs(self) -> List[np.ndarray]:
        length = len(self.b_list)
        right_envs = [None] * (length + 1)
        right_envs[length] = np.array([[1.0 + 0.0j]])
        for site in range(length - 1, 0, -1):
            bin_idx = site - 1
            right_envs[site] = self._apply_env_right(
                self.b_list[site],
                self.bc_list[site],
                self.e_no_list[bin_idx],
                right_envs[site + 1],
            )
        return right_envs

    def sum_same_bin(
        self,
        left_envs: List[np.ndarray],
        effects_by_bin: List[dict],
        key_pair: Tuple[str, ...],
    ) -> float:
        total = 0.0
        for site in range(1, self.n_bins + 1):
            op_pair = effects_by_bin[site - 1].get(key_pair, self.zero_effect)
            env_mid = self._apply_env_left(
                self.b_list[site],
                self.bc_list[site],
                op_pair,
                left_envs[site],
            )
            total += float(np.einsum('ij,ij->', env_mid, self.right_envs[site + 1]).real)
        return total

    def sum_diff_bins(
        self,
        left_envs: List[np.ndarray],
        effects_by_bin: List[dict],
        key_first: Tuple[str, ...],
        key_second: Tuple[str, ...],
        window_bins: Optional[int],
    ) -> float:
        total = 0.0
        for first_site in range(1, self.n_bins):
            op_first = effects_by_bin[first_site - 1].get(key_first, self.zero_effect)
            env_mid = self._apply_env_left(
                self.b_list[first_site],
                self.bc_list[first_site],
                op_first,
                left_envs[first_site],
            )
            j_end = self.n_bins
            if window_bins is not None:
                j_end = min(self.n_bins, first_site + window_bins)
            for second_site in range(first_site + 1, j_end + 1):
                op_second = effects_by_bin[second_site - 1].get(key_second, self.zero_effect)
                env_j = self._apply_env_left(
                    self.b_list[second_site],
                    self.bc_list[second_site],
                    op_second,
                    env_mid,
                )
                total += float(np.einsum('ij,ij->', env_j, self.right_envs[second_site + 1]).real)
                if second_site < j_end:
                    env_mid = self._apply_env_left(
                        self.b_list[second_site],
                        self.bc_list[second_site],
                        self.e_no_list[second_site - 1],
                        env_mid,
                    )
        return total

    def collect_same_bin_records(
        self,
        effects_by_bin: List[dict],
        det_a: str,
        det_b: str,
        weight_eps: float,
    ) -> List[Tuple[str, str, int, int, float]]:
        key_pair = self.order_detectors([det_a, det_b])
        records = []
        for site in range(1, self.n_bins + 1):
            op_pair = effects_by_bin[site - 1].get(key_pair, self.zero_effect)
            env_mid = self._apply_env_left(
                self.b_list[site],
                self.bc_list[site],
                op_pair,
                self.left_envs_identity[site],
            )
            weight = float(np.einsum('ij,ij->', env_mid, self.right_envs[site + 1]).real)
            if weight > weight_eps:
                records.append((det_a, det_b, site - 1, site - 1, weight))
        return records

    def collect_diff_bin_records(
        self,
        effects_by_bin: List[dict],
        det_first: str,
        det_second: str,
        weight_eps: float,
        window_bins: Optional[int],
    ) -> List[Tuple[str, str, int, int, float]]:
        key_first = self.order_detectors([det_first])
        key_second = self.order_detectors([det_second])
        records = []
        for first_site in range(1, self.n_bins):
            op_first = effects_by_bin[first_site - 1].get(key_first, self.zero_effect)
            env_mid = self._apply_env_left(
                self.b_list[first_site],
                self.bc_list[first_site],
                op_first,
                self.left_envs_identity[first_site],
            )
            j_end = self.n_bins
            if window_bins is not None:
                j_end = min(self.n_bins, first_site + window_bins)
            for second_site in range(first_site + 1, j_end + 1):
                op_second = effects_by_bin[second_site - 1].get(key_second, self.zero_effect)
                env_j = self._apply_env_left(
                    self.b_list[second_site],
                    self.bc_list[second_site],
                    op_second,
                    env_mid,
                )
                weight = float(np.einsum('ij,ij->', env_j, self.right_envs[second_site + 1]).real)
                if weight > weight_eps:
                    records.append((det_first, det_second, first_site - 1, second_site - 1, weight))
                if second_site < j_end:
                    env_mid = self._apply_env_left(
                        self.b_list[second_site],
                        self.bc_list[second_site],
                        self.e_no_list[second_site - 1],
                        env_mid,
                    )
        return records

    def contract_record(
        self,
        left_envs: List[np.ndarray],
        effects_by_bin: List[dict],
        det_a: str,
        det_b: str,
        bin_a: int,
        bin_b: int,
    ) -> complex:
        if bin_a == bin_b:
            site = bin_a + 1
            key_pair = self.order_detectors([det_a, det_b])
            op_pair = effects_by_bin[bin_a].get(key_pair, self.zero_effect)
            env_mid = self._apply_env_left(
                self.b_list[site],
                self.bc_list[site],
                op_pair,
                left_envs[site],
            )
            return np.einsum('ij,ij->', env_mid, self.right_envs[site + 1])

        if bin_a < bin_b:
            first_site = bin_a + 1
            second_site = bin_b + 1
            key_first = self.order_detectors([det_a])
            key_second = self.order_detectors([det_b])
            op_first = effects_by_bin[bin_a].get(key_first, self.zero_effect)
            op_second = effects_by_bin[bin_b].get(key_second, self.zero_effect)
        else:
            first_site = bin_b + 1
            second_site = bin_a + 1
            key_first = self.order_detectors([det_b])
            key_second = self.order_detectors([det_a])
            op_first = effects_by_bin[bin_b].get(key_first, self.zero_effect)
            op_second = effects_by_bin[bin_a].get(key_second, self.zero_effect)

        env_mid = self._apply_env_left(
            self.b_list[first_site],
            self.bc_list[first_site],
            op_first,
            left_envs[first_site],
        )
        for site in range(first_site + 1, second_site):
            env_mid = self._apply_env_left(
                self.b_list[site],
                self.bc_list[site],
                self.e_no_list[site - 1],
                env_mid,
            )
        env_mid = self._apply_env_left(
            self.b_list[second_site],
            self.bc_list[second_site],
            op_second,
            env_mid,
        )
        return np.einsum('ij,ij->', env_mid, self.right_envs[second_site + 1])

    def _ensure_left_envs_qubit(self) -> None:
        if self._left_envs_qubit is not None:
            return
        self._left_envs_qubit = [[None for _ in range(4)] for _ in range(4)]
        for i, qi in enumerate(self.qubit_indices):
            for j, qj in enumerate(self.qubit_indices):
                atom_op = np.zeros((self.dim_atom, self.dim_atom), dtype=complex)
                atom_op[qi, qj] = 1.0
                self._left_envs_qubit[i][j] = self.build_left_envs(atom_op)

    def compute_record_qubit_state(
        self,
        effects_by_bin: List[dict],
        det_a: str,
        det_b: str,
        bin_a: int,
        bin_b: int,
    ) -> np.ndarray:
        self._ensure_left_envs_qubit()
        sigma = np.zeros((4, 4), dtype=complex)
        for i in range(4):
            for j in range(4):
                left_envs = self._left_envs_qubit[i][j]
                sigma[i, j] = self.contract_record(
                    left_envs=left_envs,
                    effects_by_bin=effects_by_bin,
                    det_a=det_a,
                    det_b=det_b,
                    bin_a=bin_a,
                    bin_b=bin_b,
                )
        return sigma

    def weight_record_masked(
        self,
        effects_mask_by_bin: List[dict],
        det_a: str,
        det_b: str,
        bin_a: int,
        bin_b: int,
        dark_mask: Tuple[str, ...],
        empty_key: Tuple[str, ...],
    ) -> float:
        mask_set = set(dark_mask)
        if bin_a == bin_b:
            site = bin_a + 1
            key_pair = self.order_detectors([det_a, det_b])
            op_pair = effects_mask_by_bin[bin_a].get(key_pair, {}).get(dark_mask, self.zero_effect)
            env_mid = self._apply_env_left(
                self.b_list[site],
                self.bc_list[site],
                op_pair,
                self.left_envs_identity[site],
            )
            return float(np.einsum('ij,ij->', env_mid, self.right_envs[site + 1]).real)

        if bin_a < bin_b:
            first_site = bin_a + 1
            second_site = bin_b + 1
            det_first = det_a
            det_second = det_b
            first_bin = bin_a
            second_bin = bin_b
        else:
            first_site = bin_b + 1
            second_site = bin_a + 1
            det_first = det_b
            det_second = det_a
            first_bin = bin_b
            second_bin = bin_a

        key_first = self.order_detectors([det_first])
        key_second = self.order_detectors([det_second])
        mask_first = self.order_detectors([det_first]) if det_first in mask_set else empty_key
        mask_second = self.order_detectors([det_second]) if det_second in mask_set else empty_key
        op_first = effects_mask_by_bin[first_bin].get(key_first, {}).get(mask_first, self.zero_effect)
        op_second = effects_mask_by_bin[second_bin].get(key_second, {}).get(mask_second, self.zero_effect)

        env_mid = self._apply_env_left(
            self.b_list[first_site],
            self.bc_list[first_site],
            op_first,
            self.left_envs_identity[first_site],
        )
        for site in range(first_site + 1, second_site):
            env_mid = self._apply_env_left(
                self.b_list[site],
                self.bc_list[site],
                self.e_no_list[site - 1],
                env_mid,
            )
        env_mid = self._apply_env_left(
            self.b_list[second_site],
            self.bc_list[second_site],
            op_second,
            env_mid,
        )
        return float(np.einsum('ij,ij->', env_mid, self.right_envs[second_site + 1]).real)
