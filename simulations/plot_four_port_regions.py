"""Publication-size Matplotlib sections of three ideal four-port routers.

Both figures are exactly 180 mm wide. PDF assets are vector graphics; PNG
copies are exported at 508 dpi. All labels are 7 pt Helvetica, with 8 pt bold
panel letters. Walsh and DFT sections are analytic. Intermediate sections use
a converged ellipse envelope, never a convex hull of phase samples.
"""

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from tfln_paths import OUTPUT_ROOT
import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.font_manager import FontProperties
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

OUT = OUTPUT_ROOT / "figures/supplementary"
WIDTH = 180 / 25.4
DPI = 508
BLUE, LIGHT, GRAY = "#3378A8", "#9CC8E3", "#777777"
ALPHAS = [0, np.pi / 4, np.pi / 2]
NAMES = ["Walsh: 0°", "Intermediate: 45°", "DFT: 90°"]
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "mathtext.fontset": "custom",
        "mathtext.rm": "DejaVu Sans",
        "mathtext.it": "DejaVu Sans:italic",
        "mathtext.bf": "DejaVu Sans:bold",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.6,
    }
)

PANEL_FONT = FontProperties(family="DejaVu Sans", weight="bold", size=8)


def section(s, alpha, resolution=1800):
    if alpha == 0:
        if abs(s - 0.5) < 1e-12:
            return [
                np.array([[-0.5, -0.5], [0.5, 0.5]]),
                np.array([[-0.5, 0.5], [0.5, -0.5]]),
            ]
        flip = s > 0.5
        t = 1 - s if flip else s
        x = np.linspace(-t, t, 601)
        lo, hi = (1 - t) / t * np.abs(x), np.sqrt(x * x + 1 - 2 * t)
        upper = np.column_stack((np.r_[x, x[::-1]], np.r_[hi, lo[::-1]]))
        polys = [upper, upper * np.array([1, -1])]
        return [p[:, ::-1] if flip else p for p in polys]
    theta = np.linspace(0, 2 * np.pi, 1001)
    cx, sy = np.cos(theta), np.sin(theta)
    if abs(alpha - np.pi / 2) < 1e-12:
        radius = 1 / np.sqrt(cx * cx / s**2 + sy * sy / (1 - s) ** 2)
    elif abs(s - 0.5) < 1e-12:
        radius = (
            0.5 * np.sin(alpha) / np.sqrt(1 - np.cos(alpha) * np.abs(np.sin(2 * theta)))
        )
    else:
        q = 1 - 2 * s
        low, high = max(0, -q), s * s
        f = np.unique(
            np.r_[np.linspace(1e-10, 1, resolution), np.geomspace(1e-14, 1, resolution)]
        )[:, None]
        k = low + (high - low) * f
        ell = q + k if q >= 0 else (high - low) * f
        quad = (
            ell * cx**2
            + k * sy**2
            - 2 * np.cos(alpha) * np.sqrt(k * ell) * np.abs(cx * sy)
        )
        radius = np.sqrt(np.max(k * ell * np.sin(alpha) ** 2 / quad, axis=0))
        if q > 0:
            radius[np.abs(cx) < 1e-12] = np.maximum(
                radius[np.abs(cx) < 1e-12], np.sqrt(q)
            )
        else:
            radius[np.abs(sy) < 1e-12] = np.maximum(
                radius[np.abs(sy) < 1e-12], np.sqrt(-q)
            )
    return [np.column_stack((radius * cx, radius * sy))]


def export(fig, stem):
    # Do not use bbox_inches='tight': it would change the physical dimensions.
    fig.savefig(OUT / f"{stem}.pdf", facecolor="white")
    fig.savefig(OUT / f"{stem}.png", dpi=DPI, facecolor="white")
    plt.close(fig)


def plot_sections():
    fig, axes = plt.subplots(
        3, 3, figsize=(WIDTH, 142 / 25.4), sharex=True, sharey=True
    )
    fig.subplots_adjust(
        left=0.15, right=0.985, bottom=0.13, top=0.91, wspace=0.20, hspace=0.24
    )
    for row, alpha in enumerate(ALPHAS):
        for col, s in enumerate([0.25, 0.5, 0.75]):
            ax = axes[row, col]
            ax.add_patch(
                Rectangle(
                    (-s, -(1 - s)),
                    2 * s,
                    2 * (1 - s),
                    facecolor="#F2F2F2",
                    edgecolor=GRAY,
                    lw=0.65,
                )
            )
            for p in section(s, alpha):
                if len(p) > 2:
                    ax.fill(*p.T, color=LIGHT, alpha=0.8, edgecolor="none")
                ax.plot(*p.T, color=BLUE, lw=0.85)
            ax.axhline(0, color="#CCCCCC", lw=0.35, zorder=0)
            ax.axvline(0, color="#CCCCCC", lw=0.35, zorder=0)
            ax.set_aspect("equal")
            ax.set_xlim(-0.81, 0.81)
            ax.set_ylim(-0.81, 0.81)
            ax.set_xticks([-0.75, 0, 0.75])
            ax.set_yticks([-0.75, 0, 0.75])
            ax.tick_params(width=0.5, length=2, pad=2)
            ax.spines[["top", "right"]].set_visible(False)
            ax.text(
                -0.08,
                1.025,
                chr(97 + 3 * row + col),
                transform=ax.transAxes,
                fontproperties=PANEL_FONT,
                va="bottom",
            )
            if row == 0:
                ax.set_title(f"$s={s:.2f}$", pad=12)
            if col == 0:
                ax.set_ylabel(NAMES[row] + "\n" + r"$y=P_2-P_4$", labelpad=6)
            if row == 2:
                ax.set_xlabel(r"$x=P_1-P_3$", labelpad=4)
    fig.legend(
        handles=[
            Patch(facecolor=LIGHT, edgecolor=BLUE, label="Reachable section"),
            Patch(facecolor="#F2F2F2", edgecolor=GRAY, label="Full simplex section"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.55, 0.025),
        ncol=2,
        frameon=False,
    )
    export(fig, "FIG_Reachable_Regions_4x4_Sections")


def plot_sliced_3d():
    fig = plt.figure(figsize=(WIDTH, 86 / 25.4))
    vertices = np.array([[1, 0, 1], [0, 1, 0], [-1, 0, 1], [0, -1, 0]])
    for col, alpha in enumerate(ALPHAS):
        ax = fig.add_subplot(1, 3, col + 1, projection="3d")
        for i in range(4):
            for j in range(i + 1, 4):
                ax.plot(*vertices[[i, j]].T, color=GRAY, lw=0.55)
        for s in [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]:
            rectangle = np.array(
                [
                    [-s, -(1 - s), s],
                    [s, -(1 - s), s],
                    [s, 1 - s, s],
                    [-s, 1 - s, s],
                    [-s, -(1 - s), s],
                ]
            )
            if s in [0.25, 0.5, 0.75]:
                ax.plot(*rectangle.T, color="#AAAAAA", lw=0.45)
            for p in section(s, alpha):
                coords = np.column_stack((p, np.full(len(p), s)))
                ax.plot(*coords.T, color=BLUE, lw=0.8 if s == 0.5 else 0.5)
                if s in [0.25, 0.5, 0.75] and len(p) > 2:
                    ax.add_collection3d(
                        Poly3DCollection(
                            [coords], facecolor=LIGHT, alpha=0.5, edgecolor="none"
                        )
                    )
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(0, 1)
        # The x=1 and y=-1 ticks share the foreground corner. Omit the latter
        # label so the two numbers cannot collide in a publication-size panel.
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([0, 1])
        ax.set_zticks([0, 0.5, 1])
        ax.tick_params(pad=-2, labelsize=7)
        ax.set_xlabel(r"$x=P_1-P_3$", labelpad=-5)
        ax.set_ylabel(r"$y=P_2-P_4$", labelpad=-5)
        ax.set_box_aspect((2, 2, 1.7), zoom=0.87)
        ax.set_proj_type("ortho")
        ax.view_init(elev=22, azim=-52)
        ax.set_title(NAMES[col], pad=1)
        ax.text2D(
            0.02, 0.96, chr(97 + col), transform=ax.transAxes, fontproperties=PANEL_FONT
        )
        ax.grid(False)
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.pane.fill = False
            axis.pane.set_edgecolor("white")
    fig.text(0.5, 0.91, r"Vertical coordinate: $s=P_1+P_3$", ha="center")
    fig.subplots_adjust(left=0.005, right=0.965, top=0.85, bottom=0.19, wspace=0.1)
    fig.legend(
        handles=[
            Patch(facecolor=LIGHT, edgecolor=BLUE, label="Reachable sections"),
            Patch(facecolor="none", edgecolor=GRAY, label="Full power simplex"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=2,
        frameon=False,
    )
    export(fig, "FIG_Reachable_Regions_4x4_Sliced3D")


def verify():
    rng = np.random.default_rng(41)
    phi, psi, chi = rng.uniform(0, 2 * np.pi, (3, 10000))
    u = np.column_stack(
        (
            np.exp(1j * phi),
            np.exp(1j * (chi + psi)),
            np.exp(-1j * phi),
            np.exp(1j * (chi - psi)),
        )
    )
    s = (np.cos(phi) ** 2 + np.cos(psi) ** 2) / 2
    x = np.cos(phi) * np.cos(psi) * np.cos(chi)
    for alpha in ALPHAS:
        z = np.exp(1j * alpha)
        m = np.array([[1, 1, 1, 1], [1, z, -1, -z], [1, -1, 1, -1], [1, -z, -1, z]]) / 2
        y = np.sin(phi) * np.sin(psi) * np.cos(chi + alpha)
        p = np.column_stack(
            ((s + x) / 2, (1 - s + y) / 2, (s - x) / 2, (1 - s - y) / 2)
        )
        assert np.max(np.abs(p - np.abs(u @ m / 2) ** 2)) < 1e-12
    for s in [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]:
        coarse = section(s, np.pi / 4, 1800)[0]
        fine = section(s, np.pi / 4, 3600)[0]
        assert np.max(np.linalg.norm(coarse - fine, axis=1)) < 3e-5
    print("Power maps and intermediate-envelope convergence checks passed.")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    verify()
    plot_sections()
    plot_sliced_3d()
