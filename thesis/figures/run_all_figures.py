import pathlib
import subprocess
import sys


def main() -> None:
    figures_dir = pathlib.Path(__file__).resolve().parent
    scripts = sorted(figures_dir.glob("fig_*.py"))
    for script in scripts:
        subprocess.run([sys.executable, str(script)], check=True)


if __name__ == "__main__":
    main()
