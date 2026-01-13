# -*- coding: utf-8 -*-
"""
单轨迹执行模块

本模块实现时间仓仿真的"传送带"主循环。
每个时间仓按顺序处理：发射、QFC、损耗、琼斯旋转、分束器、探测。
"""

from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, field
import numpy as np

from ..core.mps import MPSState
from ..config import TimeGrid, EmitParams, QFCParams, FiberParams, DetParams
from ..hilbert.basis import BIN_SPACE, SUBSPACE_780, SUBSPACE_1517
from ..physics.gates import (
    emission_gate, qfc_gate, bs_gate_bin18, jones_gate_from_array, swap_gate
)
from ..physics.channels import (
    loss_channel_1517, loss_channel_both_subspaces,
    detection_channel_two_mode, detection_povm_single_site,
    dephasing_channel_from_rate
)


# 维度常量，便于代码阅读
DIM_ATOM = 3
DIM_BIN = BIN_SPACE.dim  # 18
DIM_780 = SUBSPACE_780.dim  # 3
DIM_1517 = SUBSPACE_1517.dim  # 6

# 仓子空间索引（780 x 1517 积空间）
# index = i_780 * DIM_1517 + i_1517
IDX_780_VAC = 0  # 780子空间中的 |vac>
IDX_780_H = 1    # 780子空间中的 |H>
IDX_780_V = 2    # 780子空间中的 |V>

# 18维仓空间中的780H块：索引范围 DIM_1517 * 1 到 DIM_1511 * 2 - 1
IDX_BIN_780H_START = DIM_1517 * IDX_780_H  # 6
IDX_BIN_780H_END = DIM_1517 * (IDX_780_H + 1)  # 12

# 18维仓空间中的780V块
IDX_BIN_780V_START = DIM_1517 * IDX_780_V  # 12
IDX_BIN_780V_END = DIM_1517 * (IDX_780_V + 1)  # 18


@dataclass
class TrajectoryResult:
    """
    单次轨迹运行的结果。

    Attributes
    ----------
    success : bool
        轨迹是否产生成功模式
    rho_atom : np.ndarray
        末态原子密度矩阵（两原子为9x9矩阵）
    outcome : Optional[Tuple[int, int, int, int]]
        探测器点击模式 (d1_H, d1_V, d2_H, d2_V)
    success_bin : Optional[int]
        产生成功的仓索引（无成功则为None）
    record : List[Tuple[int, int, int, int]]
        所有仓的探测器结果完整记录
    """
    success: bool
    rho_atom: np.ndarray
    outcome: Optional[Tuple[int, int, int, int]] = None
    success_bin: Optional[int] = None
    record: List[Tuple[int, int, int, int]] = field(default_factory=list)


@dataclass
class EmissionResult:
    """
    双原子发射仿真的结果（仅发射阶段）。

    发射后的链布局：atomA, atomB, A1, B1, A2, B2, ..., AN, BN
    （原子在前，仓在后）

    Attributes
    ----------
    mps : MPSState
        发射后的最终MPS态
    time_grid : TimeGrid
        仿真使用的时间网格
    per_bin_prob_A : np.ndarray
        A臂每个仓的发射概率（形状：n_bins）
    per_bin_prob_B : np.ndarray
        B臂每个仓的发射概率（形状：n_bins）
    atom_states : dict
        最终原子状态 {'A': rho_A, 'B': rho_B}
    atom_A_state_evolution : np.ndarray
        原子A的状态演化（形状：3 x 2*n_bins）
        行：P(|0>), P(|1>), P(|e>)
        列：每次SWAP后的记录
    atom_B_state_evolution : np.ndarray
        原子B的状态演化（形状：3 x 2*n_bins）
        行：P(|0>), P(|1>), P(|e>)
        列：每次SWAP后的记录
    """
    mps: MPSState
    time_grid: TimeGrid
    per_bin_prob_A: np.ndarray
    per_bin_prob_B: np.ndarray
    atom_states: dict
    atom_A_state_evolution: np.ndarray = field(default_factory=lambda: np.zeros((3, 1)))
    atom_B_state_evolution: np.ndarray = field(default_factory=lambda: np.zeros((3, 1)))

    def get_bin_indices(self, n: int) -> Tuple[int, int]:
        """
        获取仓n在A臂和B臂的MPS格点索引。

        发射后的链布局：
        - atomA, atomB, A1, B1, A2, B2, ..., AN, BN
        - A_n 位于格点 2 + 2*n，B_n 位于格点 2 + 2*n + 1

        Parameters
        ----------
        n : int
            仓索引（从0开始）

        Returns
        -------
        Tuple[int, int]
            (site_A, site_B) - A_n和 B_n的MPS格点索引
        """
        # 原子在格点 0, 1
        # 仓从格点 2 开始：A1(2), B1(3), A2(4), B2(5), ...
        return 2 + 2 * n, 2 + 2 * n + 1

    def get_atom_site_indices(self) -> Tuple[int, int]:
        """
        获取原子A和B的MPS格点索引。

        发射后，原子位于链的最左端。

        Returns
        -------
        Tuple[int, int]
            (site_A, site_B) - atomA和atomB的MPS格点索引
        """
        return 0, 1

    def get_n_bins(self) -> int:
        """获取时间仓的数量。"""
        return len(self.per_bin_prob_A)

    def get_mps_for_next_stage(self) -> MPSState:
        """
        获取准备进入下一阶段的MPS态（如QFC、BSM）。

        当前布局为：atomA, atomB, A1, B1, A2, B2, ..., AN, BN
        其中每对 A_n, B_n 相邻以便进行操作。

        Returns
        -------
        MPSState
            准备好进行下一处理的MPS态
        """
        return self.mps


class TrajectoryRunner:
    """
    执行时间仓协议的单次轨迹。

    实现"传送带"算法，每个仓按以下方式处理：
    1. 发射（原子 -> 光子）
    2. QFC（780 -> 1517 频率转换）
    3. 琼斯旋转（偏振）
    4. 损耗通道
    5. 分束器（A_n 与 B_n）
    6. 探测

    链布局：A0 - B0 - A1 - B1 - A2 - B2 - ...
    其中 A0, B0 是原子，A_n, B_n 是时间仓。
    """

    def __init__(
        self,
        time_grid: TimeGrid,
        emit_params: EmitParams,
        qfc_params: QFCParams,
        fiber_params: FiberParams,
        det_params: DetParams,
        chi_max: int = 100,
        seed: Optional[int] = None,
    ):
        """
        初始化轨迹运行器。

        Parameters
        ----------
        time_grid : TimeGrid
            时间离散化
        emit_params : EmitParams
            发射参数
        qfc_params : QFCParams
            QFC参数
        fiber_params : FiberParams
            光纤/琼斯/PMD参数
        det_params : DetParams
            探测参数
        chi_max : int
            最大键维度
        seed : int, optional
            用于可重复性的随机种子
        """
        self.time_grid = time_grid
        self.emit = emit_params
        self.qfc = qfc_params
        self.fiber = fiber_params
        self.det = det_params
        self.chi_max = chi_max

        # 随机数生成器
        self.rng = np.random.default_rng(seed)

        # 缓存的门（仅计算一次）
        self._cached_gates: Dict[str, np.ndarray] = {}

    def initialize_mps(self) -> MPSState:
        """
        初始化MPS，原子处于激发态，仓处于真空态。

        新架构链布局：A1(18D) - B1(18D) - ... - AN(18D) - BN(18D) - atomA(3D) - atomB(3D)
        仓在前，原子在后。

        Returns
        -------
        MPSState
            初始化的MPS态
        """
        # 链布局：仓在前，原子在后
        local_dims = [DIM_BIN, DIM_BIN] * self.time_grid.N + [DIM_ATOM, DIM_ATOM]

        # 初态：仓真空，原子激发
        # 原子基：|0>, |1>, |e>，|e> 在索引2
        init_state = [0] * (2 * self.time_grid.N) + [2, 2]

        mps = MPSState(local_dims, init_state=init_state, max_bond=self.chi_max)
        return mps

    def run_bin(
        self,
        mps: MPSState,
        n: int,
    ) -> Tuple[MPSState, Tuple[int, int, int, int]]:
        """
        处理单个时间仓n。

        步骤：
        1. A0-An 和 B0-Bn 上的发射
        2. An 和 Bn 上的QFC
        3. An 和 Bn 上的琼斯旋转
        4. An 和 Bn 上的损耗
        5. An-Bn 上的分束器
        6. An-Bn 上的探测
        7. 完成 An-Bn

        Parameters
        ----------
        mps : MPSState
            当前MPS态
        n : int
            仓索引（从1开始，n=1对应链中格点2,3）

        Returns
        -------
        Tuple[MPSState, Tuple[int, int, int, int]]
            更新后的MPS和探测器结果 (d1_H, d1_V, d2_H, d2_V)
        """
        # 格点索引：A0=0, B0=1, A1=2, B1=3, A2=4, B2=4, ...
        # 对于仓n（从1开始），格点位于索引 2n 和 2n+1
        site_A = 2 * n
        site_B = 2 * n + 1

        t = self.time_grid.t[n-1]  # n从1开始

        # (1) 发射：两格点酉门（原子，仓）
        U_emit_A = emission_gate(
            gamma=self.emit.get_gamma_A(t),
            dt=self.time_grid.dt * 1e9,  # 秒转换为纳秒
            Alpha=self.emit.Alpha_A,
            which_atom='A'
        )
        # 发射门作用于 atom(3D) ⊗ 780(3D)，需要嵌入到完整的18维仓空间
        # 暂时使用简化版本
        mps.apply_bond_op(0, U_emit_A)  # A0-An发射（简化）

        U_emit_B = emission_gate(
            gamma=self.emit.get_gamma_B(t),
            dt=self.time_grid.dt * 1e9,  # 秒转换为纳秒
            Alpha=self.emit.Alpha_B,
            which_atom='B'
        )
        mps.apply_bond_op(1, U_emit_B)  # B0-Bn发射（简化）

        # (2) QFC：An, Bn上的单格点酉门
        U_qfc = qfc_gate(theta_H=self.qfc.theta_H, theta_V=self.qfc.theta_V)
        mps.apply_one_site_gate(site_A, U_qfc)
        mps.apply_one_site_gate(site_B, U_qfc)

        # (3) 琼斯旋转：单格点酉门
        U_pol_A = jones_gate_from_array(self.fiber.Jones_A)
        U_pol_B = jones_gate_from_array(self.fiber.Jones_B)

        # 嵌入到18维仓空间（仅作用于1517子空间）
        # 暂时直接应用（假设已正确嵌入）
        mps.apply_one_site_gate(site_A, U_pol_A)
        mps.apply_one_site_gate(site_B, U_pol_B)

        # (4) 损耗：单格点Kraus算符
        K_loss_A = loss_channel_1517(
            eta_H=self.fiber.eta_fiber_A * self.qfc.eta_ins_H,
            eta_V=self.fiber.eta_fiber_A * self.qfc.eta_ins_V
        )
        mps.apply_kraus_one_site(site_A, K_loss_A, self.rng)

        K_loss_B = loss_channel_1517(
            eta_H=self.fiber.eta_fiber_B * self.qfc.eta_ins_H,
            eta_V=self.fiber.eta_fiber_B * self.qfc.eta_ins_V
        )
        mps.apply_kraus_one_site(site_B, K_loss_B, self.rng)

        # (5) 分束器：两格点酉门
        U_bs = bs_gate_bin18()
        mps.apply_bond_op(site_A, U_bs)

        # (6) 探测：两格点测量Kraus算符
        K_det, outcomes = detection_channel_two_mode(
            eta_det=self.det.eta_det,
            p_dark=self.det.p_dark
        )
        mu = mps.apply_kraus_two_site(site_A, K_det, self.rng)
        outcome = outcomes[mu]

        # (7) 完成已测量的仓对
        mps.finalize_bin_pair(site_A)

        return mps, outcome

    def run(self) -> TrajectoryResult:
        """
        在所有时间仓上运行完整轨迹。

        Returns
        -------
        TrajectoryResult
            轨迹的结果
        """
        mps = self.initialize_mps()
        record = []

        for n in range(1, self.time_grid.N + 1):
            mps, outcome = self.run_bin(mps, n)
            record.append(outcome)

            # 检查是否成功
            if self.det.is_success(outcome):
                # 提取原子态
                rho_atom = mps.get_reduced_density([0, 1])  # A0, B0

                return TrajectoryResult(
                    success=True,
                    rho_atom=rho_atom,
                    outcome=outcome,
                    success_bin=n,
                    record=record
                )

        # 没有任何仓成功
        rho_atom = mps.get_reduced_density([0, 1])
        return TrajectoryResult(
            success=False,
            rho_atom=rho_atom,
            outcome=None,
            success_bin=None,
            record=record
        )

    def run_emission(
        self,
        verbose: bool = True,
        delay_bins_B: int = 0,
    ) -> EmissionResult:
        """
        运行仅发射阶段（原子向左移动版本）。

        链结构（初始）：A1, B1, ..., AN, BN, atomA, atomB
        链结构（最终）：atomA, atomB, A1, B1, ..., AN, BN

        原子从右向左移动，依次与每个仓对相互作用。
        关键：发射前原子先 swap 到仓左侧，使门顺序为 (atom, bin)。
        """
        start_bin_A = max(0, -delay_bins_B)
        start_bin_B = max(0, delay_bins_B)

        if verbose:
            print("=" * 70)
            print("Dual-Atom Emission Simulation")
            print("=" * 70)
            print(f"\nParameters:")
            print(f"  n_bins = {self.time_grid.N}, dt = {self.time_grid.dt * 1e9:.1f} ns")
            if delay_bins_B != 0:
                print(f"  delay_bins_B = {delay_bins_B}")
                print(f"    -> Atom A starts at bin {start_bin_A}")
                print(f"    -> Atom B starts at bin {start_bin_B}")

        # 初始化MPS：仓在前，原子在后
        local_dims = [DIM_BIN, DIM_BIN] * self.time_grid.N + [DIM_ATOM, DIM_ATOM]
        init_state = [0] * (2 * self.time_grid.N) + [2, 2]
        mps = MPSState(local_dims=local_dims, init_state=init_state, max_bond=self.chi_max)

        per_bin_prob_A = np.zeros(self.time_grid.N)
        per_bin_prob_B = np.zeros(self.time_grid.N)
        atom_A_state_evolution = np.zeros((3, 2 * self.time_grid.N))
        atom_B_state_evolution = np.zeros((3, 2 * self.time_grid.N))

        # 记录初始原子态（原子在最后两个格点）
        site_A = 2 * self.time_grid.N
        site_B = 2 * self.time_grid.N + 1
        rho_A_init = mps.get_reduced_density([site_A])
        rho_B_init = mps.get_reduced_density([site_B])
        atom_A_state_evolution[0, 0] = rho_A_init[0, 0].real
        atom_A_state_evolution[1, 0] = rho_A_init[1, 1].real
        atom_A_state_evolution[2, 0] = rho_A_init[2, 2].real
        atom_B_state_evolution[0, 0] = rho_B_init[0, 0].real
        atom_B_state_evolution[1, 0] = rho_B_init[1, 1].real
        atom_B_state_evolution[2, 0] = rho_B_init[2, 2].real

        if verbose:
            print(f"\nRunning emission (atoms move left)...")
            print(f"  Initial: [A1, B1, ..., AN, BN, atomA, atomB]")
            print(f"  Target:  [atomA, atomB, A1, B1, ..., AN, BN]")

        # 逐个处理仓（从最后一个仓开始，向左移动）
        for n in reversed(range(self.time_grid.N)):
            t = self.time_grid.t[n]

            # === 原子B发射 ===
            # 目标仓 B_n 在格点 2*n + 1
            target_B = 2 * n + 1

            # 策略：atomB 最终需要到达 site_B = target_B（在仓左侧）
            # 但在此之前，需要确保 atomA 不阻挡

            # 步骤1：如果 atomA 阻挡在 target_B 或 target_B+1，先将其移到 target_B - 1
            if site_A >= target_B:
                # atomA 在 target_B 或更右边，需要向左移
                while site_A > target_B - 1 and site_A > 0:
                    if site_A - 1 == site_B:
                        # atomB 在左边，先交换
                        mps.swap_sites(site_A - 1)
                        # swap 后 atomB 到 site_A，atomA 到 site_A-1
                        old_site_B = site_B
                        site_B = site_A  # atomB 新位置
                        site_A = old_site_B  # atomA 新位置
                    else:
                        mps.swap_sites(site_A - 1)
                        site_A -= 1

            # 步骤2：现在 atomA 在 target_B - 1 或更左边，移动 atomB
            while site_B > target_B + 1:
                if site_B - 1 == site_A:
                    # atomA 在左边，交换
                    mps.swap_sites(site_B - 1)
                    # swap 后 atomB 到 site_B-1，atomA 到 site_B
                    site_A = site_B  # atomA 新位置
                    site_B = site_B - 1  # atomB 新位置
                else:
                    mps.swap_sites(site_B - 1)
                    site_B -= 1

            # 步骤3：现在 site_B = target_B + 1，需要 swap 使 atomB 在仓左侧
            if site_B - 1 == site_A:
                # atomA 在 target_B 位置，这是特殊情况
                # 交换 atomB 和 atomA
                mps.swap_sites(site_B - 1)
                temp = site_A
                site_A = site_B
                site_B = temp
            else:
                # 正常情况：swap atomB 和 bin
                mps.swap_sites(site_B - 1)
                site_B -= 1

            # 应用发射门
            t_rel_B = self.time_grid.t[n - start_bin_B] if n >= start_bin_B else 0.0
            gamma_B = self.emit.get_gamma_B(t_rel_B) if n >= start_bin_B else 0.0
            should_emit_B = (n >= start_bin_B) and (gamma_B >= 1e-6)
            if should_emit_B:
                U_emit_B = emission_gate(
                    gamma=gamma_B,
                    dt=self.time_grid.dt * 1e9,
                    Alpha=self.emit.Alpha_B,
                    which_atom='B'
                )
                # 作用于 (仓 B_n, atomB)
                mps.apply_bond_op(site_B, U_emit_B)

                # 提取发射概率
                rho_B_n = mps.get_reduced_density([site_B])
                p_B_H = rho_B_n[IDX_BIN_780H_START:IDX_BIN_780H_END,
                               IDX_BIN_780H_START:IDX_BIN_780H_END].sum().real
                p_B_V = rho_B_n[IDX_BIN_780V_START:IDX_BIN_780V_END,
                               IDX_BIN_780H_START:IDX_BIN_780V_END].sum().real
                per_bin_prob_B[n] = p_B_H + p_B_V

            # 将atomB继续向左移（越过已处理的仓）
            if site_B > 0:
                if site_B - 1 == site_A:
                    # atomA 在左边，交换
                    mps.swap_sites(site_B - 1)
                    # swap 后 atomB 到 site_B-1，atomA 到 site_B
                    temp = site_A
                    site_A = site_B
                    site_B = temp
                else:
                    mps.swap_sites(site_B - 1)
                    site_B -= 1

            # 记录原子B状态
            rho_B_after = mps.get_reduced_density([site_B])
            col_idx_B = 2 * n
            atom_B_state_evolution[0, col_idx_B] = rho_B_after[0, 0].real
            atom_B_state_evolution[1, col_idx_B] = rho_B_after[1, 1].real
            atom_B_state_evolution[2, col_idx_B] = rho_B_after[2, 2].real

            # === 原子A发射 ===
            # 目标仓 A_n 在格点 2*n
            target_A = 2 * n

            # 策略：atomA 最终需要到达 site_A = target_A（在仓左侧）
            # 但在此之前，需要确保 atomB 不在 target_A 或 target_A + 1

            # 步骤1：如果 atomB 阻挡在 target_A 或 target_A+1，先将其移到 target_A - 1
            if site_B >= target_A:
                # atomB 在 target_A 或更右边，需要向左移
                while site_B > target_A - 1 and site_B > 0:
                    # 将 atomB 向左移动
                    if site_B - 1 == site_A:
                        # atomA 在左边，先交换
                        mps.swap_sites(site_B - 1)
                        # swap 后 atomA 到 site_B，atomB 到 site_B-1
                        site_B = site_B - 1  # atomB 新位置
                        # site_A 不变（实际上 atomA 移到了 site_B，但我们会更新）
                        # 实际上 atomA 移到了原 site_B，所以 site_A = site_B + 1
                        old_site_A = site_A
                        site_A = site_B + 1
                        # 现在 site_B = old_site_A - 1, site_A = old_site_A
                    else:
                        mps.swap_sites(site_B - 1)
                        site_B -= 1

            # 步骤2：现在 atomB 在 target_A - 1 或更左边，移动 atomA
            while site_A > target_A + 1:
                if site_A - 1 == site_B:
                    # atomB 在左边，交换
                    mps.swap_sites(site_A - 1)
                    # swap 后 atomA 到 site_A-1，atomB 到 site_A
                    site_B = site_A  # atomB 新位置
                    site_A = site_A - 1  # atomA 新位置
                else:
                    mps.swap_sites(site_A - 1)
                    site_A -= 1

            # 步骤3：现在 site_A = target_A + 1，需要 swap 使 atomA 在仓左侧
            if site_A - 1 == site_B:
                # atomB 在 target_A 位置，这是特殊情况
                # 交换 atomA 和 atomB
                mps.swap_sites(site_A - 1)
                temp = site_A
                site_A = site_B
                site_B = temp
            else:
                # 正常情况：swap atomA 和 bin
                mps.swap_sites(site_A - 1)
                site_A -= 1

            # 应用发射门
            t_rel_A = self.time_grid.t[n - start_bin_A] if n >= start_bin_A else 0.0
            gamma_A = self.emit.get_gamma_A(t_rel_A) if n >= start_bin_A else 0.0
            should_emit_A = (n >= start_bin_A) and (gamma_A >= 1e-6)
            if should_emit_A:
                U_emit_A = emission_gate(
                    gamma=gamma_A,
                    dt=self.time_grid.dt * 1e9,
                    Alpha=self.emit.Alpha_A,
                    which_atom='A'
                )
                # 作用于 (仓 A_n, atomA)
                mps.apply_bond_op(site_A, U_emit_A)

                # 提取发射概率
                rho_A_n = mps.get_reduced_density([site_A])
                p_A_H = rho_A_n[IDX_BIN_780H_START:IDX_BIN_780H_END,
                               IDX_BIN_780H_START:IDX_BIN_780H_END].sum().real
                p_A_V = rho_A_n[IDX_BIN_780V_START:IDX_BIN_780V_END,
                               IDX_BIN_780H_START:IDX_BIN_780V_END].sum().real
                per_bin_prob_A[n] = p_A_H + p_A_V

            # 将atomA继续向左移（越过已处理的仓）
            if site_A > 0:
                if site_A - 1 == site_B:
                    # atomB 在左边，交换
                    mps.swap_sites(site_A - 1)
                    # swap 后 atomA 到 site_A-1，atomB 到 site_A
                    temp = site_B
                    site_B = site_A
                    site_A = temp
                else:
                    mps.swap_sites(site_A - 1)
                    site_A -= 1

            # 记录原子A状态
            rho_A_after = mps.get_reduced_density([site_A])
            col_idx_A = 2 * n + 1
            atom_A_state_evolution[0, col_idx_A] = rho_A_after[0, 0].real
            atom_A_state_evolution[1, col_idx_A] = rho_A_after[1, 1].real
            atom_A_state_evolution[2, col_idx_A] = rho_A_after[2, 2].real

            if verbose and (self.time_grid.N - n) % 50 == 0:
                processed = self.time_grid.N - n
                print(f"  Processed {processed}/{self.time_grid.N} bins... "
                      f"(atomA@{site_A}, atomB@{site_B})")

        # 获取最终原子态
        rho_atom_A = mps.get_reduced_density([0])
        rho_atom_B = mps.get_reduced_density([1])

        atom_states = {'A': rho_atom_A, 'B': rho_atom_B}

        if verbose:
            print(f"\n  Complete!")
            print(f"  Final: atomA@0, atomB@1")
            print(f"  Final chi: {mps.get_bond_dimensions()}")
            print(f"  Norm: {mps.norm():.6f}")
            print(f"\nFinal atomic states:")
            print(f"  Atom A: P(|e>)={rho_atom_A[2,2].real:.4f}")
            print(f"  Atom B: P(|e>)={rho_atom_B[2,2].real:.4f}")
            print(f"\nEmission statistics:")
            print(f"  Arm A total: {per_bin_prob_A.sum():.4f}, peak: {per_bin_prob_A.max():.4f}")
            print(f"  Arm B total: {per_bin_prob_B.sum():.4f}, peak: {per_bin_prob_B.max():.4f}")

        return EmissionResult(
            mps=mps,
            time_grid=self.time_grid,
            per_bin_prob_A=per_bin_prob_A,
            per_bin_prob_B=per_bin_prob_B,
            atom_states=atom_states,
            atom_A_state_evolution=atom_A_state_evolution,
            atom_B_state_evolution=atom_B_state_evolution,
        )


def run_single_trajectory(
    time_grid: TimeGrid,
    emit_params: EmitParams,
    qfc_params: QFCParams,
    fiber_params: FiberParams,
    det_params: DetParams,
    chi_max: int = 100,
    seed: Optional[int] = None,
) -> TrajectoryResult:
    """
    运行单次轨迹的便捷函数。

    Parameters
    ----------
    time_grid : TimeGrid
        时间离散化
    emit_params : EmitParams
        发射参数
    qfc_params : QFCParams
        QFC参数
    fiber_params : FiberParams
        光纤/琼斯/PMD参数
    det_params : DetParams
        探测参数
    chi_max : int
        最大键维度
    seed : int, optional
        随机种子

    Returns
    -------
    TrajectoryResult
        轨迹的结果
    """
    runner = TrajectoryRunner(
        time_grid=time_grid,
        emit_params=emit_params,
        qfc_params=qfc_params,
        fiber_params=fiber_params,
        det_params=det_params,
        chi_max=chi_max,
        seed=seed,
    )
    return runner.run()


def run_emission_only(
    time_grid: TimeGrid,
    emit_params: EmitParams,
    qfc_params: Optional[QFCParams] = None,
    fiber_params: Optional[FiberParams] = None,
    det_params: Optional[DetParams] = None,
    chi_max: int = 100,
    verbose: bool = True,
    delay_bins_B: int = 0,
) -> EmissionResult:
    """
    使用SWAP传送带协议运行仅发射仿真。

    这是总仿真第一阶段的便捷函数：
    - 两个原子（A和B）处于激发态
    - 发射到时间仓（仅780nm，暂无QFC）
    - 最终态准备好进入下一门（基站BSM）

    Parameters
    ----------
    time_grid : TimeGrid
        时间离散化
    emit_params : EmitParams
        发射参数（gamma_A, gamma_B, Alpha_A, Alpha_B）
    qfc_params : QFCParams, optional
        仅发射阶段不使用，但保留以保持接口一致
    fiber_params : FiberParams, optional
        仅发射阶段不使用，但保留以保持接口一致
    det_params : DetParams, optional
        仅发射阶段不使用，但保留以保持接口一致
    chi_max : int
        最大键维度
    verbose : bool
        是否打印进度信息
    delay_bins_B : int
        原子B发射延迟的仓数（用于时间偏移）

    Returns
    -------
    EmissionResult
        发射仿真结果的容器
    """
    # 如果未提供则创建默认参数
    if qfc_params is None:
        from ..config import QFCParams as _QFCParams
        qfc_params = _QFCParams()
    if fiber_params is None:
        from ..config import FiberParams as _FiberParams
        fiber_params = _FiberParams()
    if det_params is None:
        from ..config import DetParams as _DetParams
        det_params = _DetParams()

    runner = TrajectoryRunner(
        time_grid=time_grid,
        emit_params=emit_params,
        qfc_params=qfc_params,
        fiber_params=fiber_params,
        det_params=det_params,
        chi_max=chi_max,
    )
    return runner.run_emission(verbose=verbose, delay_bins_B=delay_bins_B)


# ============================================================================
# 统一处理函数（apply_* 模式）
# 所有函数遵循相同接口：
#   - 输入：mps (MPSState), params, verbose (bool)
#   - 输出：mps (MPSState)
#   - 打印格式：所有函数保持一致
# ============================================================================

def apply_qfc(
    mps: MPSState,
    n_bins: int,
    theta_H: float = np.pi/4,
    theta_V: float = np.pi/4,
    verbose: bool = True,
) -> MPSState:
    """
    对所有仓应用QFC门。

    Parameters
    ----------
    mps : MPSState
        MPS态（布局：A1, B1, A2, B2, ..., AN, BN, atomA, atomB）
    n_bins : int
        时间仓数量
    theta_H : float
        H偏振的QFC角度（sin² = 转换概率）
    theta_V : float
        V偏振的QFC角度
    verbose : bool
        是否打印进度

    Returns
    -------
    MPSState
        应用了QFC的MPS态（原地修改）
    """
    from ..physics.gates import qfc_gate

    _print_header("QFC", verbose)
    if verbose:
        print(f"  theta_H = {theta_H:.4f} (sin² = {np.sin(theta_H)**2:.3f})")
        print(f"  theta_V = {theta_V:.4f} (sin² = {np.sin(theta_V)**2:.3f})")

    # 获取QFC门（18x18，作用于单个仓）
    U_qfc = qfc_gate(theta_H=theta_H, theta_V=theta_V)

    if verbose:
        print(f"  U_qfc shape: {U_qfc.shape}")
        print(f"  n_bins={n_bins}, MPS L={mps.L}")
        print(f"  MPS d[:5]={mps.d[:5]}, d[-5:]={mps.d[-5:]}")

    # 对每个仓应用QFC
    # 链布局：atomA(0), atomB(1), A1(2), B1(3), A2(4), B2(5), ...
    for n in range(n_bins):
        site_A = 2 + 2 * n
        site_B = 2 + 2 * n + 1

        mps.apply_one_site_gate(site_A, U_qfc)
        mps.apply_one_site_gate(site_B, U_qfc)

        _print_progress(n + 1, n_bins, verbose)

    _print_footer(mps, verbose, stage="QFC")
    return mps


def apply_780_filter(
    mps: MPSState,
    n_bins: int,
    verbose: bool = True,
) -> MPSState:
    """
    应用100%损耗滤波器从所有仓中移除780nm光子。

    QFC之后，任何剩余的780nm光子（|H,vac>, |V,vac>）无法在光纤中
    传播，必须被滤除。这应用投影：
        P_filter = |vac><vac|_780 ⊗ I_1517

    Parameters
    ----------
    mps : MPSState
        MPS态（布局：atomA, atomB, A1, B1, A2, B2, ..., AN, BN）
    n_bins : int
        时间仓数量
    verbose : bool
        是否打印进度

    Returns
    -------
    MPSState
        移除了780nm光子的MPS态（原地修改）
    """
    from ..physics.gates import filter_780_gate
    from tenpy.linalg.np_conserved import Array

    _print_header("780nm Filter", verbose)

    # 获取780nm滤波器投影（18x18）
    P_filter = filter_780_gate()

    if verbose:
        print(f"  P_filter shape: {P_filter.shape}")
        print(f"  This projects 780nm photon states to vacuum")

    # 转换为带适当标签的TeNPy Array
    P_arr = Array.from_ndarray_trivial(P_filter, labels=['p', 'p*'])

    # 应用到所有仓，每个之后不归一化
    # 链布局：atomA(0), atomB(1), A1(2), B1(3), A2(4), B2(5), ...
    for n in range(n_bins):
        site_A = 2 + 2 * n
        site_B = 2 + 2 * n + 1

        # 应用���影（非酉，暂不重新归一化）
        mps._mps.apply_local_op(site_A, P_arr, unitary=False, renormalize=False)
        mps._mps.apply_local_op(site_B, P_arr, unitary=False, renormalize=False)

        _print_progress(n + 1, n_bins, verbose)

    # 最后的单一归一化
    mps._mps.canonical_form_finite(renormalize=True)

    _print_footer(mps, verbose, stage="780nm Filter")
    return mps


def apply_jones(
    mps: MPSState,
    n_bins: int,
    Jones_A: np.ndarray,
    Jones_B: np.ndarray,
    verbose: bool = True,
) -> MPSState:
    """
    对所有仓应用琼斯偏振旋转。

    Parameters
    ----------
    mps : MPSState
        MPS态（布局：atomA, atomB, A1, B1, A2, B2, ..., AN, BN）
    n_bins : int
        时间仓数量
    Jones_A : np.ndarray
        A臂的2x2琼斯矩阵
    Jones_B : np.ndarray
        B臂的2x2琼斯矩阵
    verbose : bool
        是否打印进度

    Returns
    -------
    MPSState
        应用了琼斯旋转的MPS态（原地修改）
    """
    from ..physics.gates import jones_gate_from_array

    _print_header("Jones", verbose)
    if verbose:
        print(f"  Jones_A: {Jones_A}")
        print(f"  Jones_B: {Jones_B}")

    # 获取琼斯门（18x18，已嵌入）
    U_J_A = jones_gate_from_array(Jones_A)
    U_J_B = jones_gate_from_array(Jones_B)

    # 链布局：atomA(0), atomB(1), A1(2), B1(3), A2(4), B2(5), ...
    for n in range(n_bins):
        site_A = 2 + 2 * n
        site_B = 2 + 2 * n + 1
        mps.apply_one_site_gate(site_A, U_J_A)
        mps.apply_one_site_gate(site_B, U_J_B)

        _print_progress(n + 1, n_bins, verbose)

    _print_footer(mps, verbose, stage="Jones")
    return mps


def apply_loss(
    mps: MPSState,
    n_bins: int,
    eta_H_A: float,
    eta_V_A: float,
    eta_H_B: float,
    eta_V_B: float,
    rng: np.random.Generator,
    verbose: bool = True,
) -> MPSState:
    """
    对所有仓应用损耗通道。

    Parameters
    ----------
    mps : MPSState
        MPS态（布局：atomA, atomB, A1, B1, A2, B2, ..., AN, BN）
    n_bins : int
        时间仓数量
    eta_H_A, eta_V_A : float
        A臂的透过率（H, V偏振）
    eta_H_B, eta_V_B : float
        B臂的透过率（H, V偏振）
    rng : np.random.Generator
        用于Kraus采样的随机数生成器
    verbose : bool
        是否打印进度

    Returns
    -------
    MPSState
        应用了损耗的MPS态（原地修改）
    """
    _print_header("Loss", verbose)
    if verbose:
        print(f"  Arm A: eta_H={eta_H_A:.3f}, eta_V={eta_V_A:.3f}")
        print(f"  Arm B: eta_H={eta_H_B:.3f}, eta_V={eta_V_B:.3f}")

    # 获取损耗Kraus算符（18x18，已嵌入）
    K_loss_A = loss_channel_1517(eta_H_A, eta_V_A)
    K_loss_B = loss_channel_1517(eta_H_B, eta_V_B)

    # 链布局：atomA(0), atomB(1), A1(2), B1(3), A2(4), B2(5), ...
    for n in range(n_bins):
        site_A = 2 + 2 * n
        site_B = 2 + 2 * n + 1
        mps.apply_kraus_one_site(site_A, K_loss_A, rng)
        mps.apply_kraus_one_site(site_B, K_loss_B, rng)

        _print_progress(n + 1, n_bins, verbose)

    _print_footer(mps, verbose, stage="Loss")
    return mps


def apply_loss_combined(
    mps: MPSState,
    n_bins: int,
    eta_780: float,
    eta_H_1517: float,
    eta_V_1517: float,
    rng: np.random.Generator,
    verbose: bool = True,
) -> MPSState:
    """
    对所有仓应用组合损耗通道（780和1517子空间）。

    对于QFC应用：通常 eta_780=0（100%滤波），
    eta_1517=0.5~0.8（正常传输损耗）。

    Parameters
    ----------
    mps : MPSState
        MPS态（布局：atomA, atomB, A1, B1, A2, B2, ..., AN, BN）
    n_bins : int
        时间仓数量
    eta_780 : float
        780nm子空间的透过率（0 = 100%损耗/滤波）
    eta_H_1517 : float
        1517nm H偏振的透过率
    eta_V_1517 : float
        1517nm V偏振的透过率
    rng : np.random.Generator
        用于Kraus采样的随机数生成器
    verbose : bool
        是否打印进度

    Returns
    -------
    MPSState
        应用了损耗的MPS态（原地修改）
    """
    _print_header("Loss", verbose)
    if verbose:
        print(f"  780nm: eta={eta_780:.3f} ({'100% filtered' if eta_780==0 else 'partial loss'})")
        print(f"  1517nm: eta_H={eta_H_1517:.3f}, eta_V={eta_V_1517:.3f}")

    # 获取组合Kraus算符（18x18，两个子空间）
    K_list = loss_channel_both_subspaces(eta_780, eta_H_1517, eta_V_1517)

    # 链布局：atomA(0), atomB(1), A1(2), B1(3), A2(4), B2(5), ...
    for n in range(n_bins):
        site_A = 2 + 2 * n
        site_B = 2 + 2 * n + 1
        mps.apply_kraus_one_site(site_A, K_list, rng)
        mps.apply_kraus_one_site(site_B, K_list, rng)

        _print_progress(n + 1, n_bins, verbose)

    _print_footer(mps, verbose, stage="Loss")
    return mps


def apply_bs(
    mps: MPSState,
    n_bins: int,
    verbose: bool = True,
) -> MPSState:
    """
    对每个 A_n, B_n 对应用分束器。

    Parameters
    ----------
    mps : MPSState
        MPS态（布局：atomA, atomB, A1, B1, A2, B2, ..., AN, BN）
    n_bins : int
        时间仓数量
    verbose : bool
        是否打印进度

    Returns
    -------
    MPSState
        应用了BS的MPS态（原地修改）
    """
    from ..physics.gates import bs_gate_bin18

    _print_header("BS", verbose)

    # 获取BS门（324x324，作用于 bin_A × bin_B = 18 × 18）
    U_bs = bs_gate_bin18()

    if verbose:
        print(f"  U_bs shape: {U_bs.shape}")

    # 对每个 A_n, B_n 对应用BS
    # 链布局：atomA(0), atomB(1), A1(2), B1(3), A2(4), B2(5), ...
    for n in range(n_bins):
        site_A = 2 + 2 * n
        site_B = 2 + 2 * n + 1
        mps.apply_bond_op(site_A, U_bs)

        _print_progress(n + 1, n_bins, verbose)

    _print_footer(mps, verbose, stage="BS")
    return mps


# 一致打印格式的辅助函数
def _print_header(stage: str, verbose: bool):
    """以一致格式打印阶段标题。"""
    if verbose:
        print(f"\n{'='*60}")
        print(f"{stage:>56} <<<")
        print(f"{'='*60}")

def _print_progress(current: int, total: int, verbose: bool):
    """以一致格式打印进度。"""
    if verbose and (current % 50 == 0 or current == total):
        print(f"  Processed {current}/{total} bins...")

def _print_footer(mps: MPSState, verbose: bool, stage: str = ""):
    """以一致格式打印阶段尾部。"""
    if verbose:
        print(f"  Final chi: {mps.get_bond_dimensions()}")
        print(f"{stage} complete.")


def apply_fiber_channel(
    mps: MPSState,
    n_bins: int,
    fiber_params,
    rng: np.random.Generator,
    verbose: bool = True,
) -> tuple:
    """
    应用光纤信道效应：琼斯旋转 + 损耗（含随机采样）。

    这结合了 apply_jones 和 apply_loss_combined，但从
    FiberChannelParams 为每次轨迹采样参数（模拟光纤漂移）。

    Parameters
    ----------
    mps : MPSState
        MPS态（布局：A1, B1, A2, B2, ..., AN, BN, atomA, atomB）
    n_bins : int
        时间仓数量
    fiber_params : FiberChannelParams
        光纤信道参数（将采样新的琼斯矩阵和eta）
    rng : np.random.Generator
        随机数生成器
    verbose : bool
        是否打印进度

    Returns
    -------
    tuple
        (mps, sampled_params) 其中 sampled_params = (U_A, U_B, eta, phase)
    """
    from ..physics.channels import FiberChannelParams

    _print_header("Fiber Channel", verbose)

    # 为本次轨迹采样参数
    U_A, U_B, eta, phase = fiber_params.sample_all(rng)

    if verbose:
        print(f"  Sampled Jones_A:\n{U_A}")
        print(f"  Sampled Jones_B:\n{U_B}")
        print(f"  Phase drift: {phase:.4f} rad")
        print(f"  Sampled eta: {eta:.4f}")

    # 应用琼斯旋转
    from ..physics.gates import jones_gate_from_array
    U_J_A = jones_gate_from_array(U_A)
    U_J_B = jones_gate_from_array(U_B)

    for n in range(n_bins):
        site_A = 2 * n
        site_B = 2 * n + 1
        mps.apply_one_site_gate(site_A, U_J_A)
        mps.apply_one_site_gate(site_B, U_J_B)

        _print_progress(n + 1, n_bins, verbose)

    # 应用损耗（780滤波，1517使用采样的eta）
    from ..physics.channels import loss_channel_both_subspaces
    K_list = loss_channel_both_subspaces(eta_780=0.0, eta_H_1517=eta, eta_V_1517=eta)

    for n in range(n_bins):
        site_A = 2 * n
        site_B = 2 * n + 1
        mps.apply_kraus_one_site(site_A, K_list, rng)
        mps.apply_kraus_one_site(site_B, K_list, rng)

        _print_progress(n + 1, n_bins, verbose)

    _print_footer(mps, verbose, stage="Fiber Channel")

    return mps, (U_A, U_B, eta, phase)


def apply_detection(
    mps: MPSState,
    n_bins: int,
    eta_det: float = 1.0,
    p_dark: float = 0.0,
    rng: np.random.Generator = None,
    verbose: bool = True,
) -> Tuple[MPSState, List[Tuple[int, int, int, int]]]:
    """
    对分束器后的所有仓对应用探测POVM。

    这测量每个 (A_n, B_n) 对的光子并返回点击模式。
    每个格点有H和V探测器，每对4个结果（共16种组合）。

    Parameters
    ----------
    mps : MPSState
        MPS态（布局：A1, B1, A2, B2, ..., AN, BN, atomA, atomB）
    n_bins : int
        时间仓数量
    eta_det : float
        探测效率（0 <= eta_det <= 1）
    p_dark : float
        每个探测器的暗计数概率
    rng : np.random.Generator
        用于测量采样的随机数生成器
    verbose : bool
        是否打印进度

    Returns
    -------
    Tuple[MPSState, List[Tuple[int, int, int, int]]]
        (mps, outcomes) 其中 outcomes[n] = (dA_H, dA_V, dB_H, dB_V)
        对于仓n，d=0表示无点击，d=1表示有点击。

    Notes
    -----
    对于BSM（贝尔态测量），成功模式为：
        - (1,0,0,1) 或 (0,1,1,0)：Psi+ 通告（不同端口各有一个H和一个V）
        - (0,1,0,1) 或 (1,0,1,0)：Psi- 通告（不同端口相同偏振）

    测量是破坏性的：探测后，光子态塌缩。
    """
    if rng is None:
        rng = np.random.default_rng()

    _print_header("Detection", verbose)
    if verbose:
        print(f"  eta_det = {eta_det:.3f}, p_dark = {p_dark:.6f}")

    # 获取单格点探测POVM（每个格点4个结果）
    M_single, outcomes_single = detection_povm_single_site(eta_det, p_dark)
    # M_single[i] 是18x18，outcomes_single[i] 是 (d_H, d_V)

    all_outcomes = []

    for n in range(n_bins):
        site_A = 2 * n
        site_B = 2 * n + 1

        # 对格点A应用探测，获取结果索引
        mu_A = mps.apply_kraus_one_site(site_A, M_single, rng)
        dA_H, dA_V = outcomes_single[mu_A]

        # 对格点B应用探测，获取结果索引
        mu_B = mps.apply_kraus_one_site(site_B, M_single, rng)
        dB_H, dB_V = outcomes_single[mu_B]

        outcome = (dA_H, dA_V, dB_H, dB_V)
        all_outcomes.append(outcome)

        _print_progress(n + 1, n_bins, verbose)

    if verbose:
        # 统计成功模式
        psi_plus = [(1,0,0,1), (0,1,1,0)]  # 不同端口H-V或V-H
        psi_minus = [(0,1,0,1), (1,0,1,0)]  # 不同端口相同偏振（带相位）

        n_psi_plus = sum(1 for o in all_outcomes if o in psi_plus)
        n_psi_minus = sum(1 for o in all_outcomes if o in psi_minus)
        n_double_click = sum(1 for o in all_outcomes if sum(o) >= 2)

        print(f"  Results summary:")
        print(f"    Psi+ heralding: {n_psi_plus} bins")
        print(f"    Psi- heralding: {n_psi_minus} bins")
        print(f"    Multi-click: {n_double_click} bins")

    _print_footer(mps, verbose, stage="Detection")

    return mps, all_outcomes


def find_bsm_success(
    outcomes: List[Tuple[int, int, int, int]]
) -> Tuple[bool, int, str]:
    """
    检查是否有任何仓具有BSM成功模式。

    Parameters
    ----------
    outcomes : List[Tuple[int, int, int, int]]
        所有仓的探测结果，每个是 (dA_H, dA_V, dB_H, dB_V)

    Returns
    -------
    Tuple[bool, int, str]
        (success, bin_index, bell_state) 其中：
        - success：如果找到BSM通告则为True
        - bin_index：哪个仓（从0开始索引），无成功则为-1
        - bell_state："Psi+" 或 "Psi-" 或 ""
    """
    # BSM成功模式（每臂单个光子，不同探测器）
    psi_plus_patterns = [(1,0,0,1), (0,1,1,0)]  # H_A V_B 或 V_A H_B
    psi_minus_patterns = [(1,0,1,0), (0,1,0,1)]  # H_A H_B 或 V_A V_B（带相位）

    for n, outcome in enumerate(outcomes):
        if outcome in psi_plus_patterns:
            return True, n, "Psi+"
        if outcome in psi_minus_patterns:
            return True, n, "Psi-"

    return False, -1, ""
