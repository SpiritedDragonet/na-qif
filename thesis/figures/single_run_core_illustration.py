# -*- coding: utf-8 -*-
"""
NeurIPS-style "big figure" for an end-to-end time-bin MPS simulation pipeline.

Generates:
  - pipeline_neurips_style.pdf
  - pipeline_neurips_style.png

This script is self-contained (numpy + matplotlib only).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

# -----------------------------
# Illustrative operators (dimensions match your codebase)
# -----------------------------
def qfc_gate(theta_H=np.pi/4, theta_V=np.pi/4, phi_H=0.0, phi_V=0.0):
    """5D bin QFC gate: basis [vac, H780, V780, H1517, V1517]."""
    U = np.eye(5, dtype=complex)
    cH, sH = np.cos(theta_H), np.sin(theta_H)
    cV, sV = np.cos(theta_V), np.sin(theta_V)
    phase_H = np.exp(1j * phi_H)
    phase_V = np.exp(1j * phi_V)
    # H: (1,3)
    U[1, 1] = cH
    U[1, 3] = -np.conj(phase_H) * sH
    U[3, 1] = phase_H * sH
    U[3, 3] = cH
    # V: (2,4)
    U[2, 2] = cV
    U[2, 4] = -np.conj(phase_V) * sV
    U[4, 2] = phase_V * sV
    U[4, 4] = cV
    return U


def loss_channel_both_subspaces(eta_780=0.0, eta_H_1517=0.5, eta_V_1517=0.5):
    """5D loss channel Kraus set (single-photon truncation)."""
    K = []
    K0 = np.zeros((5, 5), dtype=complex)
    K0[0, 0] = 1.0
    K0[1, 1] = np.sqrt(eta_780)
    K0[2, 2] = np.sqrt(eta_780)
    K0[3, 3] = np.sqrt(eta_H_1517)
    K0[4, 4] = np.sqrt(eta_V_1517)
    K.append(K0)

    Kh780 = np.zeros((5, 5), dtype=complex)
    Kh780[0, 1] = np.sqrt(max(0.0, 1.0 - eta_780))
    K.append(Kh780)

    Kv780 = np.zeros((5, 5), dtype=complex)
    Kv780[0, 2] = np.sqrt(max(0.0, 1.0 - eta_780))
    K.append(Kv780)

    Kh1517 = np.zeros((5, 5), dtype=complex)
    Kh1517[0, 3] = np.sqrt(max(0.0, 1.0 - eta_H_1517))
    K.append(Kh1517)

    Kv1517 = np.zeros((5, 5), dtype=complex)
    Kv1517[0, 4] = np.sqrt(max(0.0, 1.0 - eta_V_1517))
    K.append(Kv1517)

    return [Ki for Ki in K if np.any(np.abs(Ki) > 0)]


def filter_cavity_rt(fwhm_hz=50e6, dt_s=1e-9, detuning_hz=0.0):
    """Single-pole filter discretization -> (r,t)."""
    r = np.exp(-(np.pi * fwhm_hz + 1j * 2 * np.pi * detuning_hz) * dt_s)
    t = float(np.sqrt(max(0.0, 1.0 - abs(r) ** 2)))
    return r, t


def filter_cavity_step_unitary_5d3d(r, t):
    """(bin_5d ⊗ mem_3d) memory step unitary (15x15)."""
    U = np.eye(15, dtype=complex)

    def idx(bin_level, mem_level):
        return int(bin_level * 3 + mem_level)

    h_bin_mem = idx(3, 0)  # |H_1517, vac_mem>
    h_vac_mem = idx(0, 1)  # |vac_bin, H_mem>
    v_bin_mem = idx(4, 0)  # |V_1517, vac_mem>
    v_vac_mem = idx(0, 2)  # |vac_bin, V_mem>

    # H block
    U[h_bin_mem, h_bin_mem] = r
    U[h_bin_mem, h_vac_mem] = -t
    U[h_vac_mem, h_bin_mem] = t
    U[h_vac_mem, h_vac_mem] = np.conj(r)
    # V block
    U[v_bin_mem, v_bin_mem] = r
    U[v_bin_mem, v_vac_mem] = -t
    U[v_vac_mem, v_bin_mem] = t
    U[v_vac_mem, v_vac_mem] = np.conj(r)
    return U


def random_su2(rng):
    """Sample SU(2) via normalized quaternion."""
    x = rng.standard_normal(4)
    x = x / np.linalg.norm(x)
    a, b, c, d = x
    U = np.array([[a + 1j * b, c + 1j * d], [-c + 1j * d, a - 1j * b]], dtype=complex)
    det = np.linalg.det(U)
    U = U / np.sqrt(det)
    return U


def beamsplitter_unitary_9(theta=np.pi / 4):
    """Illustrative 9x9 BS unitary on a simple two-port basis.
    Only mixes the one-photon sectors (H: 1<->3, V: 2<->6); leaves other basis states identity.
    """
    U = np.eye(9, dtype=complex)
    c, s = np.cos(theta), np.sin(theta)
    # H block
    U[[1, 1, 3, 3], [1, 3, 1, 3]] = [c, s, -s, c]
    # V block
    U[[2, 2, 6, 6], [2, 6, 2, 6]] = [c, s, -s, c]
    return U


def bell_state_rho(p_mix=0.25, bell="Psi-"):
    bell_vecs = {
        "Phi+": np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2),
        "Phi-": np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2),
        "Psi+": np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2),
        "Psi-": np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2),
    }
    psi = bell_vecs[bell]
    rho_pure = np.outer(psi, np.conj(psi))
    I = np.eye(4, dtype=complex) / 4.0
    return (1 - p_mix) * rho_pure + p_mix * I


# -----------------------------
# Drawing primitives
# -----------------------------
def add_stage_box(ax, x, y, w, h, title, fc=(0.95, 0.95, 0.95), ec="black",
                  lw=1.2, ls=(0, (5, 3)), alpha=0.55):
    rect = Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec,
                     linewidth=lw, linestyle=ls, alpha=alpha, zorder=0)
    ax.add_patch(rect)
    ax.text(x + 0.01 * w, y + h - 0.04 * h, title, ha="left", va="top",
            fontsize=11, weight="bold")
    return rect


def add_arrow(ax, xy_from, xy_to, lw=1.3, style="-|>", mutation_scale=12):
    arr = FancyArrowPatch(
        xy_from, xy_to,
        arrowstyle=style, linewidth=lw, color="black",
        mutation_scale=mutation_scale, zorder=5
    )
    ax.add_patch(arr)
    return arr


def add_heatmap(fig, pos, data, title, shape_text=None, cmap="viridis",
                grid=True, grid_step=1, fontsize=8):
    ax = fig.add_axes(pos)
    ax.imshow(np.abs(data), cmap=cmap, interpolation="nearest", aspect="auto")
    ax.set_title(title, fontsize=fontsize, pad=2)
    ax.set_xticks([])
    ax.set_yticks([])
    if shape_text:
        ax.text(0.5, -0.14, shape_text, transform=ax.transAxes,
                ha="center", va="top", fontsize=fontsize - 1)
    if grid:
        n, m = data.shape
        step = int(max(1, grid_step))
        ax.set_xticks(np.arange(-0.5, m, step), minor=True)
        ax.set_yticks(np.arange(-0.5, n, step), minor=True)
        ax.grid(which="minor", linewidth=0.25, alpha=0.35)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    return ax


def add_stack(fig, base_pos, mats, titles, shape_texts=None,
              dx=0.012, dy=0.012, cmap="viridis", grid_step=1):
    left, bottom, width, height = base_pos
    axes = []
    for k, mat in enumerate(mats):
        pos = [left + dx * k, bottom + dy * k, width, height]
        st = None if shape_texts is None else shape_texts[k]
        ax = add_heatmap(
            fig, pos, mat, titles[k],
            shape_text=st, cmap=cmap,
            grid=True, grid_step=grid_step, fontsize=7
        )
        ax.set_zorder(2 + k)
        ax.patch.set_alpha(0.95 if k == len(mats) - 1 else 0.75)
        axes.append(ax)
    return axes


def add_tensor_chain(ax, x0, y0, n=7, box_w=0.018, box_h=0.03, gap=0.008, label_prefix="A"):
    xs = []
    for i in range(n):
        x = x0 + i * (box_w + gap)
        xs.append(x)
        ax.add_patch(Rectangle((x, y0), box_w, box_h,
                               facecolor="white", edgecolor="black", linewidth=0.9))
        if i == 0:
            ax.text(x + box_w / 2, y0 + box_h / 2, f"{label_prefix}₀",
                    ha="center", va="center", fontsize=7)
        elif i == n - 1:
            ax.text(x + box_w / 2, y0 + box_h / 2, f"{label_prefix}ₙ",
                    ha="center", va="center", fontsize=7)
        elif i == 2:
            ax.text(x + box_w / 2, y0 + box_h / 2, "…",
                    ha="center", va="center", fontsize=10)

    for i in range(n - 1):
        x1 = xs[i] + box_w
        x2 = xs[i + 1]
        ax.plot([x1, x2], [y0 + box_h / 2, y0 + box_h / 2],
                color="black", linewidth=0.9)


def main():
    rng = np.random.default_rng(7)

    # Operators / tensors
    U_qfc = qfc_gate(theta_H=np.pi / 3, theta_V=np.pi / 3, phi_H=0.2, phi_V=-0.15)
    K_loss = loss_channel_both_subspaces(eta_780=0.0, eta_H_1517=0.55, eta_V_1517=0.50)
    r, t = filter_cavity_rt(fwhm_hz=45e6, dt_s=1e-9, detuning_hz=6e6)
    U_filter = filter_cavity_step_unitary_5d3d(r, t)

    X = rng.normal(size=(60, 60)) + 1j * rng.normal(size=(60, 60))
    Q, _ = np.linalg.qr(X)
    U_em = Q  # illustrative 60x60 unitary-like matrix

    chi = 6
    A_tensor = rng.normal(size=(chi, 5, chi)) + 1j * rng.normal(size=(chi, 5, chi))
    A_slices = [A_tensor[:, k, :] for k in [0, 1, 3]]  # vac, H_780, H_1517

    U_jones = random_su2(rng)
    PDL = np.diag([np.sqrt(0.92), np.sqrt(0.80)])
    phase_profile = np.exp(1j * (0.35 * np.linspace(-1, 1, 16) + rng.normal(scale=0.08, size=16)))

    U_bs = beamsplitter_unitary_9(theta=np.pi / 4)
    Y = rng.normal(size=(9, 9)) + 1j * rng.normal(size=(9, 9))
    E_click = (Y.conj().T @ Y)
    E_click = E_click / np.max(np.abs(E_click))

    rho_out = bell_state_rho(p_mix=0.28, bell="Psi-")

    # Matplotlib rcParams: TrueType fonts are publication-friendly
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 10,
    })

    fig = plt.figure(figsize=(20, 11))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.02, 0.975,
        "End-to-end time-bin MPS simulation:  Rb cavity emission → QFC/filter → fiber noise (pushed to POVM) → BSM → heralded atoms",
        ha="left", va="top", fontsize=14, weight="bold"
    )
    ax.text(0.02, 0.93, "Forward (Schrödinger): state / MPS evolution",
            ha="left", va="center", fontsize=11)
    ax.text(0.02, 0.51, "Backward (Heisenberg): effects / POVM  E  ←  Φ†(E)",
            ha="left", va="center", fontsize=11)
    ax.plot([0.02, 0.98], [0.52, 0.52], color="black", linewidth=1.0, alpha=0.6)

    # Stage layout
    left_margin, right_margin, gap, n_stage = 0.02, 0.02, 0.01, 5
    stage_w = (1 - left_margin - right_margin - gap * (n_stage - 1)) / n_stage
    stage_y, stage_h = 0.06, 0.86
    stage_xs = [left_margin + i * (stage_w + gap) for i in range(n_stage)]

    titles = [
        "1) Emission\n(MPS + TEBD + trajectories)",
        "2) QFC + filter cavity\n(non-Markov memory)",
        "3) Fiber channel\n(noise gates)",
        "4) Central BSM + detectors\n(POVM construction)",
        "5) Heralded atoms\n(metrics)",
    ]
    bg_colors = [
        (0.90, 0.95, 1.00),
        (0.92, 1.00, 0.92),
        (1.00, 0.97, 0.90),
        (1.00, 0.92, 0.92),
        (0.95, 0.95, 0.95),
    ]
    for x, title, fc in zip(stage_xs, titles, bg_colors):
        add_stage_box(ax, x, stage_y, stage_w, stage_h, title, fc=fc, alpha=0.45)

    # Connecting arrows
    y_top = 0.78
    for i in range(n_stage - 1):
        x_from = stage_xs[i] + stage_w
        x_to = stage_xs[i + 1]
        add_arrow(ax, (x_from, y_top), (x_to, y_top), lw=1.2, mutation_scale=13)

    y_bot = 0.26
    for i in range(n_stage - 1, 0, -1):
        x_from = stage_xs[i]
        x_to = stage_xs[i - 1] + stage_w
        add_arrow(ax, (x_from, y_bot), (x_to, y_bot), lw=1.2, mutation_scale=13)

    # Central contraction callout
    x4 = stage_xs[3]
    ax.text(
        x4 + 0.01, 0.54,
        "Contraction & heralding:\n" + r"$p(\mathrm{record})=\langle\Psi|\;E_{\mathrm{record}}\;|\Psi\rangle$",
        ha="left", va="bottom", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=0.9)
    )

    # Stage 1
    x1 = stage_xs[0]
    ax.text(x1 + 0.01, 0.86, "Arm A", fontsize=10, weight="bold")
    add_tensor_chain(ax, x1 + 0.06, 0.84, n=7, label_prefix="A")
    ax.text(x1 + 0.01, 0.80, "Arm B", fontsize=10, weight="bold")
    add_tensor_chain(ax, x1 + 0.06, 0.78, n=7, label_prefix="B")
    ax.text(x1 + 0.06, 0.815, "time bins (5D)", fontsize=8, ha="left", va="center")

    add_heatmap(fig, [x1 + 0.11, 0.60, 0.09, 0.17], U_em, "U_em (two-site)", "60×60",
                grid=False, cmap="magma")
    add_stack(
        fig, [x1 + 0.03, 0.60, 0.07, 0.07],
        mats=A_slices,
        titles=["A[n] slice: vac", "A[n] slice: H780", "A[n] slice: H1517"],
        shape_texts=["6×6", "6×6", "6×6"],
        dx=0.012, dy=0.010, cmap="viridis", grid_step=1
    )
    ax.text(x1 + 0.03, 0.56, r"Rank-3 MPS tensor  $A^{[n]}_{\alpha\,s\,\beta}$  (χ×d×χ)", fontsize=9)

    # Stage 2
    x2 = stage_xs[1]
    ax.text(x2 + 0.01, 0.86, "Per bin (both arms):", fontsize=10, weight="bold")
    ax.text(x2 + 0.02, 0.83, "U_qfc  →  Loss(Kraus)  →  (bin ⊗ mem) coupling", fontsize=9)
    add_heatmap(fig, [x2 + 0.02, 0.62, 0.07, 0.12], U_qfc, "U_qfc", "5×5",
                grid=True, grid_step=1, cmap="magma")

    loss_stack = K_loss[:3] if len(K_loss) >= 3 else K_loss
    add_stack(
        fig, [x2 + 0.10, 0.62, 0.06, 0.10],
        mats=loss_stack,
        titles=[f"K{idx}" for idx in range(len(loss_stack))],
        shape_texts=["5×5"] * len(loss_stack),
        dx=0.010, dy=0.010, cmap="viridis", grid_step=1
    )
    ax.text(x2 + 0.10, 0.58, "Loss channel\n(Kraus set)", fontsize=8, ha="left", va="top")

    add_heatmap(fig, [x2 + 0.02, 0.30, 0.14, 0.18], U_filter, "U_filter (memory step)", "15×15",
                grid=True, grid_step=3, cmap="magma")
    ax.text(x2 + 0.02, 0.25, "mem mode (3D) carries inter-bin correlations", fontsize=9)

    # Stage 3
    x3 = stage_xs[2]
    ax.text(x3 + 0.01, 0.86, "Physical fiber (16 km / arm)", fontsize=10, weight="bold")
    ax.text(x3 + 0.01, 0.83, "Modelled noise (sampled per trajectory):", fontsize=9)
    ax.text(x3 + 0.02, 0.80, "η loss, Jones SU(2), PDL, phase drift+slope+jitter, …", fontsize=9)
    add_heatmap(fig, [x3 + 0.02, 0.62, 0.055, 0.10], U_jones, "Jones", "2×2",
                grid=True, grid_step=1, cmap="magma")
    add_heatmap(fig, [x3 + 0.085, 0.62, 0.055, 0.10], PDL, "PDL", "2×2",
                grid=True, grid_step=1, cmap="magma")
    add_heatmap(fig, [x3 + 0.02, 0.32, 0.12, 0.08], phase_profile.reshape(1, -1), "phase vs bin", "1×16",
                grid=False, cmap="viridis")
    ax.text(x3 + 0.02, 0.28, r"Computational trick: push fiber to POVM  $E'=\Phi^\dagger(E)$", fontsize=9)

    # Stage 4
    x4 = stage_xs[3]
    ax.text(x4 + 0.01, 0.86, "Interfere two arms at BS", fontsize=10, weight="bold")
    ax.text(x4 + 0.01, 0.83, "Then 4 detectors → click-pattern POVM", fontsize=9)
    add_heatmap(fig, [x4 + 0.02, 0.62, 0.07, 0.12], U_bs, "U_BS", "9×9",
                grid=True, grid_step=1, cmap="magma")
    add_heatmap(fig, [x4 + 0.11, 0.62, 0.07, 0.12], E_click, "E_record", "9×9",
                grid=True, grid_step=1, cmap="viridis")
    ax.text(x4 + 0.02, 0.58,
            r"Mode mismatch:  $E = v_\mathrm{res}E_\mathrm{ind}+(1-v_\mathrm{res})E_\mathrm{dist}$",
            fontsize=9)
    det_y = 0.33
    for k, name in enumerate(["H1", "V1", "H2", "V2"]):
        ax.add_patch(Rectangle((x4 + 0.03 + 0.035 * k, det_y), 0.028, 0.035,
                               facecolor="white", edgecolor="black", linewidth=0.9))
        ax.text(x4 + 0.044 + 0.035 * k, det_y + 0.0175, name, ha="center", va="center", fontsize=8)
    ax.text(x4 + 0.02, 0.29, "η_det, p_dark, p_bg  →  POVM effects per bin", fontsize=9)

    # Stage 5
    x5 = stage_xs[4]
    add_heatmap(fig, [x5 + 0.02, 0.62, 0.12, 0.18], rho_out, r"$\rho_{AB}$ (heralded)", "4×4",
                grid=True, grid_step=1, cmap="viridis")
    ax.text(x5 + 0.02, 0.56, "Outputs (per run):", fontsize=10, weight="bold")
    ax.text(
        x5 + 0.03, 0.52,
        "p_success,  F_declared / F_true,\nfalse fraction,  ⟨σi⊗σj⟩,\nCHSH S_max,  …",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="black", alpha=0.9)
    )

    ax.text(
        0.78, 0.02,
        "Heatmaps show |matrix elements|; stacked heatmaps indicate Kraus sets / tensor slices.",
        ha="right", va="bottom", fontsize=9
    )

    fig.savefig("pipeline_neurips_style.pdf", bbox_inches="tight")
    fig.savefig("pipeline_neurips_style.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
