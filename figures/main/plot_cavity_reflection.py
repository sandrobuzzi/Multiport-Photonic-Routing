#!/usr/bin/env python3
"""Plot the saved Fig. 1g cavity-reflection data."""

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from tfln_paths import data_path, output_path

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def main() -> None:
    data = np.genfromtxt(
        data_path("fig1/fig1g/cavity_reflection_K1_state1.csv"),
        delimiter=",",
        names=True,
    )

    fig, amplitude_axis = plt.subplots(figsize=(6.0, 3.5))
    phase_axis = amplitude_axis.twinx()
    amplitude_axis.plot(
        data["detuning_pm"], data["reflection_amplitude"], color="black"
    )
    phase_axis.plot(data["detuning_pm"], data["phase_rad"], color="tab:blue")

    amplitude_axis.set_xlabel("Wavelength detuning (pm)")
    amplitude_axis.set_ylabel("Reflection amplitude")
    phase_axis.set_ylabel("Reflection phase (rad)", color="tab:blue")
    amplitude_axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path("figures/main/fig1g_cavity_reflection.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
