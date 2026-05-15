import pathlib
import subprocess
import sys


def convert_tikz_pdf_to_png(figures_dir: pathlib.Path, script_name: str) -> None:
    stem = pathlib.Path(script_name).stem
    tex_path = figures_dir / f"{stem}.tex"
    pdf_path = figures_dir / f"{stem}.pdf"
    if not tex_path.exists() or not pdf_path.exists():
        return

    convert_pdf_to_png(figures_dir, stem)


def convert_pdf_to_png(figures_dir: pathlib.Path, stem: str) -> None:
    pdf_path = figures_dir / f"{stem}.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(f"figure pdf not found: {pdf_path}")

    subprocess.run(
        [
            "pdftoppm",
            "-png",
            "-singlefile",
            str(pdf_path),
            str(figures_dir / stem),
        ],
        check=True,
    )


def main() -> None:
    figures_dir = pathlib.Path(__file__).resolve().parent
    scripts = [
        "fig_calibration_pipeline.py",
        "fig_wavepacket.py",
        "fig_accept_hist.py",
        "fig_hom.py",
        "fig_bsm_patterns.py",
        "fig_bell_density_3d.py",
        "fig_chsh.py",
        "fig_distance.py",
        "fig_qfc_noise.py",
        "fig_error_budget.py",
        "fig_cost_scaling.py",
    ]
    static_pdf_figures = [
        "fig_ch2_timebin_mps_tebd",
        "fig_ch1_system_pressure",
        "fig_ch1_modular_network",
        "fig_ch3_effect_pushback",
    ]
    for script_name in scripts:
        script = figures_dir / script_name
        if not script.exists():
            raise FileNotFoundError(f"figure script not found: {script}")
        subprocess.run([sys.executable, str(script)], check=True)
        convert_tikz_pdf_to_png(figures_dir, script_name)
    for stem in static_pdf_figures:
        convert_pdf_to_png(figures_dir, stem)


if __name__ == "__main__":
    main()

