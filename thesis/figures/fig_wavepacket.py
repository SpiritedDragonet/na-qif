import pathlib

import matplotlib.pyplot as plt
import numpy as np


def _gaussian_pulse(t_ns: np.ndarray, center_ns: float, sigma_ns: float) -> np.ndarray:
    return np.exp(-0.5 * ((t_ns - center_ns) / sigma_ns) ** 2)


def _exp_tail(t_ns: np.ndarray, tau_ns: float) -> np.ndarray:
    out = np.exp(-np.maximum(t_ns, 0.0) / tau_ns)
    out[t_ns < 0.0] = 0.0
    return out


def _build_wavepacket(
    t_ns: np.ndarray,
    center_ns: float,
    sigma_drive_ns: float,
    tau_emit_ns: float,
    scale: float = 1.0,
) -> np.ndarray:
    dt = float(t_ns[1] - t_ns[0])
    drive = _gaussian_pulse(t_ns, center_ns, sigma_drive_ns)
    tail = _exp_tail(t_ns - center_ns, tau_emit_ns)
    conv = np.convolve(drive, tail, mode="full")[: t_ns.size] * dt
    conv = np.maximum(conv, 0.0)
    conv /= max(conv.max(), 1e-12)
    return scale * conv


def _fwhm_ns(t_ns: np.ndarray, y: np.ndarray) -> float:
    half = 0.5 * float(np.max(y))
    above = np.where(y >= half)[0]
    if above.size < 2:
        return 0.0
    return float(t_ns[above[-1]] - t_ns[above[0]])


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
        }
    )

    t_ns = np.linspace(0.0, 140.0, 1401)
    w_a = _build_wavepacket(
        t_ns=t_ns,
        center_ns=28.0,
        sigma_drive_ns=9.5,
        tau_emit_ns=26.2,
        scale=1.0,
    )
    w_b = _build_wavepacket(
        t_ns=t_ns,
        center_ns=29.8,
        sigma_drive_ns=9.0,
        tau_emit_ns=25.5,
        scale=0.96,
    )

    energy_a = np.cumsum(w_a)
    energy_b = np.cumsum(w_b)
    energy_a /= max(float(energy_a[-1]), 1e-12)
    energy_b /= max(float(energy_b[-1]), 1e-12)

    fwhm_a = _fwhm_ns(t_ns, w_a)
    fwhm_b = _fwhm_ns(t_ns, w_b)

    window_left, window_right = 18.0, 88.0
    fig = plt.figure(figsize=(9.2, 4.3), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.22)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    c_a = "#1f77b4"
    c_b = "#d62728"
    c_w = "#f2c14e"

    ax0.axvspan(window_left, window_right, color=c_w, alpha=0.16, label="acceptance window")
    ax0.plot(t_ns, w_a, lw=2.2, color=c_a, label="arm A")
    ax0.plot(t_ns, w_b, lw=2.2, color=c_b, label="arm B")
    ax0.set_xlabel("Time (ns)")
    ax0.set_ylabel("Normalized intensity")
    ax0.set_xlim(0.0, 130.0)
    ax0.set_ylim(0.0, 1.06 * max(float(w_a.max()), float(w_b.max())))
    ax0.set_title("Telecom wavepacket envelopes")
    ax0.legend(frameon=False, loc="upper right", fontsize=9)

    txt = (
        f"FWHM(A) = {fwhm_a:.1f} ns\n"
        f"FWHM(B) = {fwhm_b:.1f} ns\n"
        "tail scale ~ 26 ns"
    )
    ax0.text(
        0.03,
        0.96,
        txt,
        transform=ax0.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#cccccc", "boxstyle": "round,pad=0.3"},
    )

    ax1.plot(t_ns, energy_a, lw=2.2, color=c_a, label="A cumulative")
    ax1.plot(t_ns, energy_b, lw=2.2, color=c_b, label="B cumulative")
    ax1.axhline(0.65, ls="--", lw=1.4, color="#4d4d4d", label="65% level")
    ax1.axvline(window_right, ls="--", lw=1.3, color=c_w, alpha=0.9, label="70 ns window edge")
    ax1.set_xlim(0.0, 130.0)
    ax1.set_ylim(0.0, 1.02)
    ax1.set_xlabel("Time (ns)")
    ax1.set_ylabel("Cumulative energy")
    ax1.set_title("Energy capture profile")
    ax1.legend(frameon=False, loc="lower right", fontsize=8.5)

    fig.suptitle("Emission wavepacket calibration view", y=1.02, fontsize=12.5, fontweight="bold")
    out_path = pathlib.Path(__file__).with_suffix(".png")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
