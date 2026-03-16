import pathlib
import subprocess
import sys


def main() -> None:
    figures_dir = pathlib.Path(__file__).resolve().parent
    scripts = [
        "fig_sim_pipeline.py",
        "fig_calibration_pipeline.py",
        "fig_wavepacket.py",
        "fig_accept_hist.py",
        "fig_window_tradeoff.py",
        "fig_hom.py",
        "fig_bsm_patterns.py",
        "fig_bell_density_3d.py",
        "fig_chsh.py",
        "fig_distance.py",
        "fig_qfc_noise.py",
        "fig_detector_eff.py",
        "fig_error_budget.py",
        "fig_cost_scaling.py",
        "fig_tikz_arch_to_interface.py",
        "fig_tikz_interface_tasks.py",
        "fig_tikz_single_vs_two_photon.py",
        "fig_tikz_platform_tree.py",
        "fig_tikz_ch2_duallane.py",
        "fig_tikz_ch3_effective_dofs.py",
        "fig_tikz_ch3_effect_pushback.py",
    ]
    for script_name in scripts:
        script = figures_dir / script_name
        if not script.exists():
            raise FileNotFoundError(f"figure script not found: {script}")
        subprocess.run([sys.executable, str(script)], check=True)


if __name__ == "__main__":
    main()
