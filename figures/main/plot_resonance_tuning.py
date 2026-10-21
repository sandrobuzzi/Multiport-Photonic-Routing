#!/usr/bin/env python3
"""Plot the saved Fig. 3b resonance-tuning data."""

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
        data_path("fig3/fig3b/fig3b_effective_tuning_source_data.csv"),
        delimiter=",",
        names=True,
    )

    voltage = data["voltage_v"]
    measured = data["measured_frequency_detuning_ghz"]
    empirical = data["empirical_fit_frequency_detuning_ghz"]
    linear = data["linear_fit_frequency_detuning_ghz"]

    order = np.argsort(voltage)
    fig, axis = plt.subplots(figsize=(5.0, 4.0))
    axis.scatter(voltage, measured, s=28, color="black", label="Measured")
    axis.plot(voltage[order], empirical[order], color="tab:blue", label="Empirical fit")
    axis.plot(
        voltage[order],
        linear[order],
        color="tab:orange",
        linestyle="--",
        label="Linear fit",
    )
    axis.set_xlabel("Voltage (V)")
    axis.set_ylabel("Resonance shift (GHz)")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path("figures/main/fig3b_resonance_tuning.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
