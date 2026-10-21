"""Plot the final, processed source-data coordinates for the supplementary reference-cavity figure.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from tfln_paths import ROOT, DATA_ROOT, output_path


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator

_sys.path.insert(0, str(ROOT / "figures/main"))
import plot_reflection_transmission as fig3a

OUTPUT_STEM = output_path("figures/supplementary/Reference_Cavity_Lorentzian_Fit")
MM = 1 / 25.4


def coefficient_of_determination(data: np.ndarray, model: np.ndarray) -> float:
    residual_sum = float(np.sum((data - model) ** 2))
    total_sum = float(np.sum((data - np.mean(data)) ** 2))
    return 1.0 - residual_sum / total_sum


def main() -> None:
    with np.load(DATA_ROOT / "fig3/fig3a/plot_reference_lorentzian_plotted.npz") as data:
        detuning_pm = data["detuning_pm"]
        transmission_data = data["transmission_data"]
        reflection_data = data["reflection_data"]
        transmission_model = data["transmission_model"]
        reflection_model = data["reflection_model"]
        transmission_residual = data["transmission_residual"]
        reflection_residual = data["reflection_residual"]
        center = float(data["center"])
        fwhm = float(data["fwhm"])
        q_factor = float(data["q_factor"])
        transmission_r2 = float(data["transmission_r2"])
        reflection_r2 = float(data["reflection_r2"])

    fig3a.configure_matplotlib()
    plt.rcParams.update(
        {
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.2,
        }
    )
    figure, (axis, residual_axis) = plt.subplots(
        2,
        1,
        figsize=(88 * MM, 72 * MM),
        sharex=True,
        gridspec_kw={"height_ratios": (3.2, 1.0), "hspace": 0.08},
    )

    axis.plot(
        detuning_pm,
        transmission_data,
        color="#000000",
        linewidth=0.8,
        label="Transmission data",
        zorder=2,
    )
    axis.plot(
        detuning_pm,
        transmission_model,
        color="#000000",
        linestyle=(0, (3.0, 1.8)),
        linewidth=1.15,
        label="Transmission fit",
        zorder=4,
    )
    axis.plot(
        detuning_pm,
        reflection_data,
        color="#b8b8b8",
        linewidth=0.8,
        label="Reflection data",
        zorder=1,
    )
    axis.plot(
        detuning_pm,
        reflection_model,
        color="#666666",
        linestyle=(0, (3.0, 1.8)),
        linewidth=1.15,
        label="Reflection fit",
        zorder=3,
    )
    axis.axvline(0.0, color="#d0d0d0", linestyle=":", linewidth=0.7, zorder=0)
    axis.set_ylabel("Normalized intensity (a.u.)")
    axis.set_ylim(fig3a.INSET_Y_LIM)
    axis.yaxis.set_minor_locator(AutoMinorLocator(2))
    axis.legend(
        loc="upper right",
        frameon=False,
        ncol=2,
        columnspacing=0.9,
        handlelength=2.3,
        handletextpad=0.45,
    )
    axis.text(
        0.035,
        0.96,
        rf"$\lambda_0={center:.3f}$ nm"
        + "\n"
        + rf"FWHM $={fwhm * 1000.0:.1f}$ pm"
        + "\n"
        + rf"$Q={q_factor / 1e4:.2f}\times10^4$",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=6.5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.0},
    )
    axis.text(
        -0.13,
        1.01,
        "a",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.5,
        fontweight="bold",
    )

    residual_axis.plot(
        detuning_pm,
        transmission_residual,
        color="#000000",
        linewidth=0.75,
        label="Transmission",
    )
    residual_axis.plot(
        detuning_pm,
        reflection_residual,
        color="#8a8a8a",
        linewidth=0.75,
        label="Reflection",
    )
    residual_axis.axhline(0.0, color="#b0b0b0", linestyle=":", linewidth=0.7)
    residual_axis.set_xlabel(r"Wavelength detuning, $\Delta\lambda$ (pm)")
    residual_axis.set_ylabel("Residual")
    residual_axis.set_xlim(-150.0, 150.0)
    residual_axis.set_ylim(-0.075, 0.095)
    residual_axis.set_xticks((-150, -100, -50, 0, 50, 100, 150))
    residual_axis.set_yticks((-0.05, 0.0, 0.05))
    residual_axis.xaxis.set_minor_locator(AutoMinorLocator(2))
    residual_axis.yaxis.set_minor_locator(AutoMinorLocator(2))
    residual_axis.text(
        0.025,
        0.92,
        rf"$R_T^2={transmission_r2:.4f}$, $R_R^2={reflection_r2:.4f}$",
        transform=residual_axis.transAxes,
        ha="left",
        va="top",
        fontsize=6.2,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 0.8},
    )
    residual_axis.text(
        -0.13,
        1.01,
        "b",
        transform=residual_axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.5,
        fontweight="bold",
    )

    for current_axis in (axis, residual_axis):
        current_axis.spines["top"].set_linewidth(0.5)
        current_axis.spines["right"].set_linewidth(0.5)
        current_axis.spines["bottom"].set_linewidth(0.5)
        current_axis.spines["left"].set_linewidth(0.5)

    figure.subplots_adjust(left=0.15, right=0.985, top=0.96, bottom=0.17)
    OUTPUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_STEM.with_suffix(".pdf"), metadata={"CreationDate": None})
    figure.savefig(OUTPUT_STEM.with_suffix(".svg"), metadata={"Date": None})
    figure.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=400)
    plt.close(figure)

    print(f"Saved -> {OUTPUT_STEM.with_suffix('.pdf')}")
    print(f"Saved -> {OUTPUT_STEM.with_suffix('.svg')}")
    print(f"Saved -> {OUTPUT_STEM.with_suffix('.png')}")
    print(f"Center: {center:.6f} nm")
    print(f"FWHM: {fwhm * 1000.0:.3f} pm")
    print(f"Loaded Q: {q_factor:.1f}")
    print(f"Transmission R^2: {transmission_r2:.6f}")
    print(f"Reflection R^2: {reflection_r2:.6f}")


if __name__ == "__main__":
    main()
