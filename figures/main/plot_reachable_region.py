"""Generate the analytic three-port reachable-power-region figure."""

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from tfln_paths import OUTPUT_ROOT


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

OUTPUT_PDF = OUTPUT_ROOT / "figures" / "supplementary" / "FIG_Reachable_Region_3x3.pdf"
OUTPUT_PNG = OUTPUT_ROOT / "figures" / "supplementary" / "FIG_Reachable_Region_3x3.png"
FIGURE_WIDTH_IN = 180 / 25.4
FIGURE_HEIGHT_IN = 82 / 25.4
EXPORT_DPI = 508  # 200 pixels/mm, giving exactly 3600 pixels across 180 mm.

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

BLUE = "#3378A8"
BLUE_LIGHT = "#9CC8E3"
ORANGE = "#D97722"
GRAY = "#777777"


def boundary_power(t: np.ndarray) -> np.ndarray:
    """Power boundary in the (P1,P2,P3) coordinates used in the text."""
    return (
        np.column_stack(
            (
                (1 + 2 * np.cos(t)) ** 2,
                (1 + 2 * np.cos(t - 2 * np.pi / 3)) ** 2,
                (1 + 2 * np.cos(t + 2 * np.pi / 3)) ** 2,
            )
        )
        / 9
    )


def barycentric_xy(power: np.ndarray) -> np.ndarray:
    """Map barycentric powers to an equilateral-triangle Cartesian plot."""
    return np.column_stack(
        (power[:, 1] + 0.5 * power[:, 2], np.sqrt(3) * power[:, 2] / 2)
    )


def main() -> None:
    t = np.linspace(0, 2 * np.pi, 901)
    boundary = boundary_power(t)
    boundary_xy = barycentric_xy(boundary)

    simplex = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    simplex_closed = np.vstack((simplex, simplex[0]))
    simplex_xy = barycentric_xy(simplex)
    simplex_xy_closed = np.vstack((simplex_xy, simplex_xy[0]))

    balanced = np.array(
        [
            [4 / 9, 4 / 9, 1 / 9],
            [1 / 9, 4 / 9, 4 / 9],
            [4 / 9, 1 / 9, 4 / 9],
        ]
    )
    balanced_xy = barycentric_xy(balanced)

    fig = plt.figure(
        figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN),
        dpi=EXPORT_DPI,
        constrained_layout=False,
    )
    grid = fig.add_gridspec(1, 2, width_ratios=[1.08, 1.0], wspace=0.23)

    # (a) The conservation plane, physical simplex, and reachable subset.
    ax3 = fig.add_subplot(grid[0, 0], projection="3d")
    ax3.add_collection3d(
        Poly3DCollection(
            [simplex],
            facecolor="#F3F3F3",
            edgecolor=GRAY,
            linewidth=1.2,
            alpha=0.76,
        )
    )
    ax3.add_collection3d(
        Poly3DCollection(
            [boundary],
            facecolor=BLUE_LIGHT,
            edgecolor=BLUE,
            linewidth=1.35,
            alpha=0.88,
        )
    )
    ax3.plot(
        simplex_closed[:, 0],
        simplex_closed[:, 1],
        simplex_closed[:, 2],
        color=GRAY,
        lw=1.2,
    )
    ax3.plot(boundary[:, 0], boundary[:, 1], boundary[:, 2], color=BLUE, lw=1.45)
    ax3.scatter(
        balanced[:, 0],
        balanced[:, 1],
        balanced[:, 2],
        marker="D",
        s=17,
        color=ORANGE,
        edgecolor="white",
        linewidth=0.45,
        depthshade=False,
        zorder=10,
    )
    ax3.set(xlim=(-0.08, 1.06), ylim=(-0.08, 1.06), zlim=(-0.08, 1.06))
    ax3.set_xticks([0, 0.5, 1])
    ax3.set_yticks([0, 0.5, 1])
    ax3.set_zticks([0, 0.5, 1])
    ax3.set_xlabel(r"$P_1$", labelpad=-1)
    ax3.set_ylabel(r"$P_2$", labelpad=-1)
    # Draw the vertical-axis label in axes coordinates so mplot3d cannot
    # override its orientation.  The 270-degree rotation makes it read
    # top-to-bottom, matching the other labels in the manuscript.
    ax3.set_zlabel("")
    ax3.text2D(
        0.055,
        0.43,
        r"$P_3$",
        transform=ax3.transAxes,
        rotation=270,
        ha="center",
        va="center",
    )
    ax3.view_init(elev=25, azim=10)
    ax3.set_box_aspect((1, 1, 0.88))
    ax3.grid(False)
    for axis in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor((0.7, 0.7, 0.7, 0.65))
    ax3.text2D(
        0.01,
        0.98,
        "a)",
        transform=ax3.transAxes,
        va="top",
        fontsize=8,
        fontfamily="Helvetica",
    )
    ax3.text2D(
        0.52,
        0.91,
        r"$P_1+P_2+P_3=1$",
        transform=ax3.transAxes,
        color=GRAY,
        fontsize=7,
    )

    # (b) Barycentric view of the same simplex.
    ax2 = fig.add_subplot(grid[0, 1])
    ax2.fill(
        simplex_xy_closed[:, 0], simplex_xy_closed[:, 1], color="#F3F3F3", zorder=0
    )
    ax2.plot(
        simplex_xy_closed[:, 0], simplex_xy_closed[:, 1], color=GRAY, lw=1.25, zorder=2
    )
    ax2.fill(
        boundary_xy[:, 0], boundary_xy[:, 1], color=BLUE_LIGHT, alpha=0.9, zorder=1
    )
    ax2.plot(boundary_xy[:, 0], boundary_xy[:, 1], color=BLUE, lw=1.5, zorder=3)
    ax2.scatter(
        balanced_xy[:, 0],
        balanced_xy[:, 1],
        marker="D",
        s=25,
        color=ORANGE,
        edgecolor="white",
        linewidth=0.55,
        zorder=5,
    )
    ax2.annotate(
        "balanced limit\n" + r"$(4/9,4/9,1/9)$",
        xy=balanced_xy[0],
        xytext=(0.50, 0.20),
        ha="center",
        va="center",
        color=ORANGE,
        fontsize=7,
        arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.8, shrinkA=2, shrinkB=3),
    )
    ax2.text(
        0.50,
        0.38,
        "reachable\nregion",
        ha="center",
        va="center",
        color="#225B80",
        fontsize=7,
    )
    ax2.text(-0.045, -0.035, r"$P_1=1$", ha="center", va="top")
    ax2.text(1.045, -0.035, r"$P_2=1$", ha="center", va="top")
    ax2.text(0.50, np.sqrt(3) / 2 + 0.045, r"$P_3=1$", ha="center", va="bottom")
    ax2.set_xlim(-0.11, 1.11)
    ax2.set_ylim(-0.09, 0.98)
    ax2.set_aspect("equal")
    ax2.axis("off")
    ax2.text(
        0.01,
        0.98,
        "b)",
        transform=ax2.transAxes,
        va="top",
        fontsize=8,
        fontfamily="Helvetica",
    )

    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.035, right=0.985, bottom=0.06, top=0.98)
    fig.savefig(OUTPUT_PDF)
    fig.savefig(OUTPUT_PNG, dpi=EXPORT_DPI)
    plt.close(fig)


if __name__ == "__main__":
    main()
