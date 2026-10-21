"""Run the shared analytic three-port reachable-region generator."""

from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(
        str(
            Path(__file__).resolve().parents[1]
            / "figures/main/plot_reachable_region.py"
        ),
        run_name="__main__",
    )
