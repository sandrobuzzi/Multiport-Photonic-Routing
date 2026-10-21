"""Plot the final Fig. 3a source-data coordinates directly from CSV.

The JSON supplies the previously fitted resonance metadata.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from tfln_paths import DATA_ROOT, output_path

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator

HERE = Path(__file__).resolve().parent
PLOTTED_DATA = DATA_ROOT / "fig3/fig3a/fig3a_plotted_source_data.csv"
OUTPUT_STEM = output_path("figures/main/Fig3a_reflection_transmission")

# ---------------------------------------------------------------------------
# EDIT THESE SETTINGS TO TUNE THE PANEL IN AN IDE OR NOTEBOOK-LIKE WORKFLOW.
# Command-line arguments override them, so the source file can stay unchanged.
# ---------------------------------------------------------------------------
X_LIM = (1490.0, 1564.0)
Y_LIM = (-0.05, 1.40)
X_TICKS = (1500.0, 1520.0, 1540.0, 1560.0)
Y_TICKS = (0.0, 0.5, 1.0)

COLOR_TRANSMISSION = "#000000"
# Exact light gray sampled from the guide strokes in the assembled Fig. 3.
COLOR_REFLECTION = "#c7c7c7"
REFLECTION_ALPHA = 0.78
TRANSMISSION_LINEWIDTH = 0.6
REFLECTION_LINEWIDTH = 0.6

SHOW_LEGEND = True
LEGEND_LOC = "upper right"
LEGEND_ANCHOR = (0.985, 0.985)

SHOW_INSET = True
INSET_BOUNDS = (0.03, 0.40, 0.47, 0.58)
INSET_X_LIM_PM = (-200.0, 200.0)
INSET_Y_LIM = (-0.05, 1.40)


MM = 1 / 25.4
FIG_SIZE_MM = 56
FONT_SIZE = 7


def pair(values: list[float] | tuple[float, float]) -> tuple[float, float]:
    """Return a two-number argument as a tuple."""
    return float(values[0]), float(values[1])


def validate_limits(name: str, limits: tuple[float, float]) -> None:
    if not np.all(np.isfinite(limits)) or limits[0] >= limits[1]:
        raise ValueError(f"{name} must contain finite values in increasing order")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot final Fig. 3a source-data coordinates directly.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--xlim", nargs=2, type=float, default=X_LIM, metavar=("MIN", "MAX"))
    parser.add_argument("--ylim", nargs=2, type=float, default=Y_LIM, metavar=("MIN", "MAX"))
    parser.add_argument("--xticks", nargs="+", type=float, default=X_TICKS)
    parser.add_argument("--yticks", nargs="+", type=float, default=Y_TICKS)
    parser.add_argument("--reflection-alpha", type=float, default=REFLECTION_ALPHA)
    parser.add_argument("--reflection-color", default=COLOR_REFLECTION)
    parser.add_argument("--transmission-linewidth", type=float, default=TRANSMISSION_LINEWIDTH)
    parser.add_argument("--reflection-linewidth", type=float, default=REFLECTION_LINEWIDTH)
    parser.add_argument("--legend-loc", default=LEGEND_LOC)
    parser.add_argument(
        "--legend-anchor", nargs=2, type=float, default=LEGEND_ANCHOR, metavar=("X", "Y")
    )
    parser.add_argument(
        "--inset-bounds",
        nargs=4,
        type=float,
        default=INSET_BOUNDS,
        metavar=("LEFT", "BOTTOM", "WIDTH", "HEIGHT"),
    )
    parser.add_argument(
        "--inset-xlim", nargs=2, type=float, default=INSET_X_LIM_PM, metavar=("MIN", "MAX")
    )
    parser.add_argument(
        "--inset-ylim", nargs=2, type=float, default=INSET_Y_LIM, metavar=("MIN", "MAX")
    )
    parser.add_argument("--no-inset", action="store_true", help="Hide the resonance-fit inset.")
    parser.add_argument("--no-legend", action="store_true", help="Hide the trace legend.")
    parser.add_argument("--show", action="store_true", help="Open an interactive Matplotlib window.")
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=OUTPUT_STEM,
        help="Output path without an extension; both SVG and PNG are written.",
    )
    return parser.parse_args()


def clipped_trace(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), 0.0, None)






def configure_matplotlib() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": 6,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.minor.width": 0.4,
            "ytick.minor.width": 0.4,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.minor.size": 1.5,
            "ytick.minor.size": 1.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "svg.fonttype": "none",
            "svg.hashsalt": "fig3a-reflection-transmission",
        }
    )


def make_figure(args: argparse.Namespace) -> tuple[plt.Figure, float, float, float, float]:
    xlim = pair(args.xlim)
    ylim = pair(args.ylim)
    inset_xlim = pair(args.inset_xlim)
    inset_ylim = pair(args.inset_ylim)
    for name, limits in (
        ("--xlim", xlim),
        ("--ylim", ylim),
        ("--inset-xlim", inset_xlim),
        ("--inset-ylim", inset_ylim),
    ):
        validate_limits(name, limits)
    if not 0.0 <= args.reflection_alpha <= 1.0:
        raise ValueError("--reflection-alpha must be between 0 and 1")
    data = np.genfromtxt(PLOTTED_DATA, delimiter=",", names=True)
    metadata = json.loads(PLOTTED_DATA.with_suffix(".json").read_text())
    wavelengths = data["wavelength_nm"]
    transmission = data["transmission_plotted_au"]
    reflection = data["reflection_plotted_au"]
    plot_mask = (wavelengths >= xlim[0]) & (wavelengths <= xlim[1])
    if not np.any(plot_mask):
        raise ValueError("--xlim does not overlap the provided plotted data")
    center = metadata["resonance_center_nm"]
    fwhm = metadata["fwhm_pm"] / 1000.0
    q_factor = metadata["loaded_q"]
    q_standard_error = metadata["loaded_q_standard_error"]

    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(FIG_SIZE_MM * MM, FIG_SIZE_MM * MM))
    ax.plot(
        wavelengths[plot_mask],
        transmission[plot_mask],
        color=COLOR_TRANSMISSION,
        linewidth=args.transmission_linewidth,
        label="Transmission",
        zorder=3,
    )
    ax.plot(
        wavelengths[plot_mask],
        reflection[plot_mask],
        color=args.reflection_color,
        alpha=args.reflection_alpha,
        linewidth=args.reflection_linewidth,
        label="Reflection",
        zorder=2,
    )
    ax.axvline(
        center,
        color="#9a9a9a",
        linestyle="--",
        linewidth=0.55,
        alpha=0.85,
        ymax=0.84,
        zorder=1,
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Wavelength (nm)", labelpad=2)
    ax.set_ylabel("Intensity (a.u.)", labelpad=2)
    ax.set_xticks(args.xticks)
    ax.set_yticks(args.yticks)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(False)

    if SHOW_LEGEND and not args.no_legend:
        ax.legend(
            loc=args.legend_loc,
            bbox_to_anchor=pair(args.legend_anchor),
            frameon=False,
            borderpad=0.25,
            labelspacing=0.25,
            handlelength=1.5,
            handletextpad=0.45,
        )

    if SHOW_INSET and not args.no_inset:
        inset = ax.inset_axes(args.inset_bounds)
        inset_mask = (
            (wavelengths >= center + inset_xlim[0] / 1000.0)
            & (wavelengths <= center + inset_xlim[1] / 1000.0)
        )
        offset_pm = (wavelengths[inset_mask] - center) * 1000.0
        inset.plot(
            offset_pm,
            transmission[inset_mask],
            color=COLOR_TRANSMISSION,
            linewidth=0.9,
        )
        inset.plot(
            offset_pm,
            reflection[inset_mask],
            color=args.reflection_color,
            alpha=args.reflection_alpha,
            linewidth=0.9,
        )
        inset.axvline(0.0, color=args.reflection_color, linestyle="--", linewidth=0.6, zorder=1)
        inset.set_xlim(*inset_xlim)
        inset.set_ylim(*inset_ylim)
        inset.set_xticks([0.75 * inset_xlim[0], 0.0, 0.75 * inset_xlim[1]])
        inset.set_yticks(
            [tick for tick in (0.0, 0.5, 1.0) if inset_ylim[0] <= tick <= inset_ylim[1]]
        )
        inset.yaxis.tick_right()
        inset.xaxis.set_minor_locator(AutoMinorLocator(2))
        inset.yaxis.set_minor_locator(AutoMinorLocator(2))
        inset.set_xlabel(r"$\Delta\lambda$ (pm)", labelpad=1)
        inset.text(
            0.04,
            0.96,
            rf"$Q = {q_factor / 1e4:.1f}\,\times\,10^4$"
            + "\n"
            + rf"$\lambda_0 = {center:.2f}$ nm",
            transform=inset.transAxes,
            ha="left",
            va="top",
            fontsize=FONT_SIZE,
            bbox=dict(
                boxstyle="square,pad=0.12", facecolor="white", edgecolor="none", alpha=0.88
            ),
        )

        ax.indicate_inset_zoom(inset, edgecolor="#999999", linewidth=0.5, alpha=0.6)

    fig.tight_layout(pad=0.0)
    return fig, center, fwhm, q_factor, q_standard_error


def main() -> None:
    args = parse_args()
    output_stem = args.output_stem.expanduser().resolve()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig, center, fwhm, q_factor, q_standard_error = make_figure(args)
    fig.savefig(
        output_stem.with_suffix(".svg"),
        bbox_inches="tight",
        pad_inches=0.0,
        metadata={"Date": None},
    )
    fig.savefig(
        output_stem.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.0,
        metadata={"Software": "Matplotlib"},
    )

    print(f"Saved -> {output_stem.with_suffix('.svg')}")
    print(f"Saved -> {output_stem.with_suffix('.png')}")
    print(f"Main limits:             x={tuple(args.xlim)}, y={tuple(args.ylim)}")
    print("Source: final plotted CSV coordinates")
    print(f"Shared resonance center: {center:.5f} nm")
    print(f"Shared resonance FWHM:   {fwhm * 1000:.2f} pm")
    print(f"Loaded Q:                {q_factor:.0f} +/- {q_standard_error:.0f}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
