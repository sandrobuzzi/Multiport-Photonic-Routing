"""Plot the final, processed source-data coordinates for the supplementary reference-cavity figure.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from tfln_paths import ROOT, DATA_ROOT, output_path

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator

_sys.path.insert(0, str(ROOT / "figures/main"))
import plot_reflection_transmission as fig3a

HERE = Path(__file__).resolve().parent
OUTPUT_STEM = output_path("figures/supplementary/Full_Reference_Cavity_Spectrum")
MM = 1 / 25.4


def main() -> None:
    with np.load(DATA_ROOT / "fig3/fig3a/plot_full_reference_spectrum_plotted.npz") as data:
        wavelengths = data["wavelengths"]
        transmission_relative_db = data["transmission_relative_db"]
        reflection_relative_db = data["reflection_relative_db"]
        total_dbm = data["total_dbm"]
        envelope_dbm = data["envelope_dbm"]
        grating_wavelengths = data["grating_wavelengths"]
        grating_aligned_dbm = data["grating_aligned_dbm"]
    stride = 1
    x = wavelengths

    fig3a.configure_matplotlib()
    plt.rcParams.update(
        {
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.2,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(178 * MM, 72 * MM), sharex=True)

    axes[0].plot(
        x,
        transmission_relative_db[::stride],
        color="#000000",
        linewidth=0.55,
        label="Transmission",
        zorder=3,
    )
    axes[0].plot(
        x,
        reflection_relative_db[::stride],
        color="#aaaaaa",
        linewidth=0.55,
        label="Reflection",
        zorder=2,
    )
    axes[0].set_ylabel("Displayed power relative to common reference (dB)")
    axes[0].set_ylim(-40, 5)
    axes[0].set_yticks((-40, -30, -20, -10, 0))
    axes[0].legend(
        loc="lower center",
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.88,
        ncol=2,
    )

    axes[1].plot(
        x,
        total_dbm[::stride],
        color="#8a8a8a",
        linewidth=0.55,
        label="Reflection + transmission",
    )
    axes[1].plot(
        x,
        envelope_dbm[::stride],
        color="#000000",
        linewidth=1.0,
        label="90th-percentile envelope",
    )
    grating_stride = 1
    axes[1].plot(
        grating_wavelengths[::grating_stride],
        grating_aligned_dbm[::grating_stride],
        color="#555555",
        linestyle=(0, (3.0, 1.8)),
        linewidth=0.9,
        label="Plain-grating test (aligned)",
    )
    axes[1].set_ylabel("Combined detected power (dBm)")
    axes[1].set_ylim(-48, -10)
    axes[1].set_yticks((-45, -35, -25, -15))
    axes[1].legend(
        loc="lower center",
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.88,
        handlelength=3.0,
    )

    for index, axis in enumerate(axes):
        axis.axvspan(*fig3a.X_LIM, color="#e9e9e9", alpha=0.55, linewidth=0, zorder=-5)
        axis.text(
            sum(fig3a.X_LIM) / 2,
            0.97,
            "Fig. 3a range",
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=6.2,
            color="#555555",
        )
        axis.set_xlabel("Wavelength (nm)")
        axis.set_xlim(1480, 1640)
        axis.set_xticks((1480, 1520, 1560, 1600, 1640))
        axis.xaxis.set_minor_locator(AutoMinorLocator(2))
        axis.yaxis.set_minor_locator(AutoMinorLocator(2))
        axis.text(
            -0.12,
            1.01,
            "ab"[index],
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.5,
            fontweight="bold",
        )
        for spine in axis.spines.values():
            spine.set_linewidth(0.5)

    figure.subplots_adjust(left=0.075, right=0.975, top=0.97, bottom=0.18, wspace=0.30)
    OUTPUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_STEM.with_suffix(".pdf"), metadata={"CreationDate": None})
    figure.savefig(OUTPUT_STEM.with_suffix(".svg"), metadata={"Date": None})
    figure.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=400)
    plt.close(figure)

    print(f"Saved -> {OUTPUT_STEM.with_suffix('.pdf')}")
    print(f"Saved -> {OUTPUT_STEM.with_suffix('.svg')}")
    print(f"Saved -> {OUTPUT_STEM.with_suffix('.png')}")


if __name__ == "__main__":
    main()
