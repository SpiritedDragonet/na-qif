import pathlib
import shutil
import subprocess


def main() -> None:
    figures_dir = pathlib.Path(__file__).resolve().parent
    name = pathlib.Path(__file__).stem
    tex_path = figures_dir / f"{name}.tex"
    if not tex_path.exists():
        raise FileNotFoundError(f"TikZ source not found: {tex_path}")

    build_dir = figures_dir / f"_{name}_build"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            "xelatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-output-directory",
            str(build_dir),
            str(tex_path),
        ],
        check=True,
    )

    pdf_src = build_dir / f"{name}.pdf"
    pdf_dst = figures_dir / f"{name}.pdf"
    if not pdf_src.exists():
        raise FileNotFoundError(f"Expected PDF not found: {pdf_src}")
    shutil.copy2(pdf_src, pdf_dst)

    subprocess.run(
        [
            "pdftoppm",
            "-png",
            "-singlefile",
            str(pdf_dst),
            str(figures_dir / name),
        ],
        check=True,
    )
    shutil.rmtree(build_dir)


if __name__ == "__main__":
    main()
