#!/usr/bin/env python3
"""Fig. 4-style full voltage cube and optimized-state plane sections.

The cube rendering follows ``Good_Cubes_Plots.py`` and the plane construction
follows ``Triangle_Plot_v4.py`` from the Fig. 4 source directory. The optical
response is reevaluated with the optimized shared static phases used for the
minimum-drive calculation.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from tfln_paths import OUTPUT_ROOT


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.stats.qmc import Sobol

from analyze_fitted_model_variations import load_model
from analyze_phase_cavity_performance import output_powers

OUTPUT = OUTPUT_ROOT / "figures/supplementary" / "FIG_Joint_Drive_Geometry"
MM = 1 / 25.4
FS = 7
N_VOXELS = 15
HALF_SPAN = 5.0
GRID_N = 420

# Optimizer arm order is (K3,K2,K1); displayed Fig. 4 order is (V1,V2,V3)
# = (K1,K2,K3).
STATES_INTERNAL = np.array(
    [
        [-0.79705967, 0.74789970, -0.99445371],
        [-2.47548620, 1.87118073, -1.87733616],
        [-0.01671784, -0.58758762, -1.56846997],
    ]
)
STATES_REL = STATES_INTERNAL[:, ::-1]
PHASE_INTERNAL = np.array([0.0, 0.60531448, 0.92487868])
STATE_COLORS = ("#cc0000", "#009900", "#0044cc")


def quality_components(power: np.ndarray, low=None, span=None, q_max=None):
    """Fig. 4 quality-RGB rule, with P1/P2/P3 mapped to R/G/B."""
    power = np.asarray(power)
    if low is None:
        low = power.min(axis=0)
    if span is None:
        span = np.maximum(power.max(axis=0) - low, np.finfo(float).eps)
    normalized = np.clip((power - low) / span, 0, None)
    log_power = 10 * np.log10(normalized + 1e-6)
    competitor = np.maximum.reduce(
        (np.roll(log_power, 1, axis=1), np.roll(log_power, 2, axis=1))
    )
    quality = np.maximum(log_power - competitor, 0)
    if q_max is None:
        q_max = max(float(quality.max()), np.finfo(float).eps)
    return np.clip(quality / q_max, 0, 1), low, span, q_max


def voxelise(points: np.ndarray, colors: np.ndarray):
    """Port of the Fig. 4 cube voxelization and Sobol display sampling."""
    edges = np.linspace(-HALF_SPAN, HALF_SPAN, N_VOXELS + 1)
    indices = [
        np.clip(np.digitize(points[:, axis], edges) - 1, 0, N_VOXELS - 1)
        for axis in range(3)
    ]
    count = np.zeros((N_VOXELS,) * 3)
    voxel_color = np.zeros((N_VOXELS,) * 3 + (3,))
    np.add.at(count, tuple(indices), 1)
    for channel in range(3):
        np.add.at(voxel_color[..., channel], tuple(indices), colors[:, channel])
    filled = count > 0
    voxel_color[filled] /= count[filled, None]

    sobol = Sobol(d=3, scramble=True, seed=42).random(32768) * 2 * HALF_SPAN - HALF_SPAN
    sobol_indices = [
        np.clip(np.digitize(sobol[:, axis], edges) - 1, 0, N_VOXELS - 1)
        for axis in range(3)
    ]
    valid = filled[tuple(sobol_indices)]
    return sobol[valid], voxel_color[tuple(index[valid] for index in sobol_indices)]


def box_plane_polygon(low, high, normal, offset):
    """Ordered polygon formed by an axis-aligned box and a plane."""
    vertices = []
    for free_axis in range(3):
        fixed = [axis for axis in range(3) if axis != free_axis]
        for first in (low[fixed[0]], high[fixed[0]]):
            for second in (low[fixed[1]], high[fixed[1]]):
                point = np.zeros(3)
                point[fixed] = (first, second)
                point[free_axis] = (offset - normal[fixed] @ point[fixed]) / normal[
                    free_axis
                ]
                if low[free_axis] - 1e-9 <= point[free_axis] <= high[free_axis] + 1e-9:
                    vertices.append(point)
    vertices = np.unique(np.round(vertices, 11), axis=0)
    center = vertices.mean(axis=0)
    trial = np.cross(normal, [0.0, 0.0, 1.0])
    if np.linalg.norm(trial) < 1e-10:
        trial = np.cross(normal, [0.0, 1.0, 0.0])
    basis_1 = trial / np.linalg.norm(trial)
    basis_2 = np.cross(normal, basis_1)
    angle = np.arctan2((vertices - center) @ basis_2, (vertices - center) @ basis_1)
    return vertices[np.argsort(angle)]


def plane_basis(states_abs):
    """Use the basis convention of Fig. 4 Triangle_Plot_v4.py."""
    origin = states_abs.mean(axis=0)
    first = states_abs[1] - states_abs[0]
    first /= np.linalg.norm(first)
    second = states_abs[2] - states_abs[0]
    second -= np.dot(second, first) * first
    second /= np.linalg.norm(second)
    normal = np.cross(first, second)
    e1 = np.cross(normal, [0.0, 0.0, 1.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    if float((states_abs[0] - origin) @ e2) > 0:
        e1 *= -1
        e2 = np.cross(normal, e1)
    return origin, normal, e1, e2


def model_power(params, optical, physical_v):
    """Evaluate physical (V1,V2,V3)=(K1,K2,K3) voltages."""
    internal_v = np.asarray(physical_v)[..., ::-1].reshape(-1, 3)
    return output_powers(
        params,
        optical,
        internal_v,
        imbalance_scale=1.0,
        roundtrip_phase_offset_rad=PHASE_INTERNAL,
    )


def draw_wireframe(ax, low, high, back=True, color="#888888"):
    """Exact 3-back/9-front edge convention used by Fig. 4 cube plots."""
    xl, yl, zl = low
    xh, yh, zh = high
    back_edges = [
        ([xl, xh], [yl, yl], [zl, zl]),
        ([xl, xl], [yl, yh], [zl, zl]),
        ([xl, xl], [yl, yl], [zl, zh]),
    ]
    front_edges = [
        ([xl, xh], [yh, yh], [zl, zl]),
        ([xh, xh], [yl, yh], [zl, zl]),
        ([xl, xh], [yl, yl], [zh, zh]),
        ([xl, xh], [yh, yh], [zh, zh]),
        ([xl, xl], [yl, yh], [zh, zh]),
        ([xh, xh], [yl, yh], [zh, zh]),
        ([xh, xh], [yl, yl], [zl, zh]),
        ([xl, xl], [yh, yh], [zl, zh]),
        ([xh, xh], [yh, yh], [zl, zh]),
    ]
    if back:
        for xs, ys, zs in back_edges:
            ax.plot(xs, ys, zs, color=color, lw=0.4, ls="--", dashes=(3, 3), zorder=1)
    for xs, ys, zs in front_edges:
        ax.plot(xs, ys, zs, color=color, lw=0.4, zorder=8)


def style_plane_axis(ax):
    ax.set_xlabel(r"$e_1$ (V)")
    ax.set_ylabel(r"$e_2$ (V)")
    ax.set_aspect("equal")


def draw_cube_panel(
    fig,
    subplot_spec,
    display_abs,
    display_rgb,
    full_low,
    full_high,
    full_polygon,
    tight_low,
    tight_high,
    states_abs,
):
    """Draw one cube with the exact Fig. 4 projection and wireframe style."""
    ax = fig.add_subplot(subplot_spec, projection="3d")
    ax.set_proj_type("ortho")
    ax.set_box_aspect((1, 1, 1))
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("none")
    ax.grid(False)
    draw_wireframe(ax, full_low - 0.2, full_high + 0.2, back=True)
    ax.scatter(
        display_abs[:, 0],
        display_abs[:, 1],
        display_abs[:, 2],
        c=display_rgb,
        s=1.15,
        alpha=0.72,
        linewidths=0,
        depthshade=True,
        rasterized=True,
        zorder=2,
    )
    ax.add_collection3d(
        Poly3DCollection(
            [full_polygon],
            facecolor="#bdbdbd",
            edgecolor="#222222",
            alpha=0.18,
            linewidth=0.55,
            zorder=4,
        )
    )
    draw_wireframe(ax, tight_low, tight_high, back=False, color="#222222")
    for index, (state, color) in enumerate(zip(states_abs, STATE_COLORS), start=1):
        ax.scatter(
            *state,
            s=17,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            depthshade=False,
            zorder=9,
        )
        ax.text(*(state + 0.16), rf"$S_{index}$", color=color, fontsize=8, zorder=10)
    ax.set_xlim(full_low[0] - 0.2, full_high[0] + 0.2)
    ax.set_ylim(full_low[1] - 0.2, full_high[1] + 0.2)
    ax.set_zlim(full_low[2] - 0.2, full_high[2] + 0.2)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(mpl.ticker.MaxNLocator(nbins=2, integer=True))
    ax.tick_params(axis="x", pad=-5, colors="#888888", labelcolor="black")
    ax.tick_params(axis="y", pad=-5, colors="#888888", labelcolor="black")
    ax.tick_params(axis="z", pad=-2, colors="#888888", labelcolor="black")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.text2D(
        0.80,
        -0.05,
        r"$V_1$ (V)",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
    )
    ax.text2D(
        0.20,
        -0.05,
        r"$V_2$ (V)",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
    )
    ax.text2D(
        -0.01,
        0.74,
        r"$V_3$ (V)",
        transform=ax.transAxes,
        ha="center",
        va="center",
        rotation=90,
        fontsize=8,
    )
    ax.view_init(elev=np.degrees(np.arctan(1 / np.sqrt(2))), azim=45)
    return ax


def main():
    params, optical = load_model()
    baseline = optical.baseline_v[::-1]
    states_abs = baseline + STATES_REL
    full_low = baseline - HALF_SPAN
    full_high = baseline + HALF_SPAN
    tight_low = states_abs.min(axis=0)
    tight_high = states_abs.max(axis=0)
    tight_side = float(np.max(tight_high - tight_low))

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": FS,
            "axes.titlesize": FS,
            "axes.labelsize": 8,
            "xtick.labelsize": FS,
            "ytick.labelsize": FS,
            "legend.fontsize": FS,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.0,
            "ytick.major.size": 2.0,
            "lines.linewidth": 0.8,
            "patch.linewidth": 0.5,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )

    relative_axis = np.linspace(-HALF_SPAN, HALF_SPAN, 25)
    rr = np.stack(
        np.meshgrid(relative_axis, relative_axis, relative_axis, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    cube_abs = baseline + rr
    cube_power = model_power(params, optical, cube_abs)
    cube_rgb, power_low, power_span, q_max = quality_components(cube_power)
    intensity_rgb = np.clip((cube_power - power_low) / power_span, 0, 1)
    display_rel, display_intensity_rgb = voxelise(rr, intensity_rgb)
    _, display_quality_rgb = voxelise(rr, cube_rgb)
    display_abs = baseline + display_rel

    origin, normal, e1, e2 = plane_basis(states_abs)
    plane_offset = float(normal @ states_abs[0])
    full_polygon = box_plane_polygon(full_low, full_high, normal, plane_offset)
    tight_polygon = box_plane_polygon(tight_low, tight_high, normal, plane_offset)
    state_uv = np.column_stack(((states_abs - origin) @ e1, (states_abs - origin) @ e2))
    full_polygon_uv = np.column_stack(
        ((full_polygon - origin) @ e1, (full_polygon - origin) @ e2)
    )
    tight_polygon_uv = np.column_stack(
        ((tight_polygon - origin) @ e1, (tight_polygon - origin) @ e2)
    )

    full_half = max(abs(full_polygon_uv).max(), 0.5) + 0.45
    u_full = np.linspace(-full_half, full_half, GRID_N)
    v_full = np.linspace(-full_half, full_half, GRID_N)
    U, V = np.meshgrid(u_full, v_full)
    plane_abs = origin + U[..., None] * e1 + V[..., None] * e2
    plane_power = model_power(params, optical, plane_abs)
    plane_rgb, *_ = quality_components(plane_power, power_low, power_span, q_max)
    plane_rgb = plane_rgb.reshape(GRID_N, GRID_N, 3)
    in_full_cube = np.all(
        (plane_abs >= full_low - 1e-10) & (plane_abs <= full_high + 1e-10), axis=-1
    )

    # Keep the four panels in one row at the manuscript's 180-mm figure width.
    fig = plt.figure(figsize=(180 * MM, 65 * MM))
    gs = fig.add_gridspec(1, 4, width_ratios=(1.04, 1.04, 1, 1), wspace=0.20)

    ax_a = draw_cube_panel(
        fig,
        gs[0, 0],
        display_abs,
        display_intensity_rgb,
        full_low,
        full_high,
        full_polygon,
        tight_low,
        tight_high,
        states_abs,
    )
    ax_b = draw_cube_panel(
        fig,
        gs[0, 1],
        display_abs,
        display_quality_rgb,
        full_low,
        full_high,
        full_polygon,
        tight_low,
        tight_high,
        states_abs,
    )

    ax = fig.add_subplot(gs[0, 2])
    extent = (-full_half, full_half, -full_half, full_half)
    ax.imshow(
        plane_rgb,
        origin="lower",
        extent=extent,
        aspect="equal",
        interpolation="bilinear",
    )
    gray = np.zeros((GRID_N, GRID_N, 4))
    gray[~in_full_cube] = (0.6, 0.6, 0.6, 0.45)
    ax.imshow(gray, origin="lower", extent=extent, aspect="equal", interpolation="none")
    ax.contour(
        U, V, in_full_cube.astype(float), levels=[0.5], colors="black", linewidths=0.6
    )
    for index, (point, color) in enumerate(zip(state_uv, STATE_COLORS), start=1):
        ax.scatter(
            *point, s=18, color=color, edgecolor="white", linewidth=0.6, zorder=5
        )
        label_offset = (-0.25, -0.40) if index == 3 else (0.18, 0.18)
        ax.text(
            point[0] + label_offset[0],
            point[1] + label_offset[1],
            rf"$S_{index}$",
            color="white",
            fontsize=8,
            fontweight="bold",
            ha="right" if index == 3 else "left",
            zorder=6,
        )
    ax.set_xlim(-full_half, full_half)
    ax.set_ylim(-full_half, full_half)
    tick = np.arange(np.ceil(-full_half / 2) * 2, full_half + 1e-9, 2)
    ax.set_xticks(tick)
    ax.set_yticks(tick)
    style_plane_axis(ax)
    ax_c = ax

    margin = 0.65
    zoom_xlim = (state_uv[:, 0].min() - margin, state_uv[:, 0].max() + margin)
    zoom_ylim = (state_uv[:, 1].min() - margin, state_uv[:, 1].max() + margin)
    ax = fig.add_subplot(gs[0, 3])
    ax.imshow(
        plane_rgb,
        origin="lower",
        extent=extent,
        aspect="equal",
        interpolation="bilinear",
    )
    closed_tight = np.vstack((tight_polygon_uv, tight_polygon_uv[0]))
    ax.plot(
        closed_tight[:, 0], closed_tight[:, 1], color="white", linewidth=1.8, zorder=4
    )
    ax.plot(
        closed_tight[:, 0],
        closed_tight[:, 1],
        color="#222222",
        linewidth=0.65,
        zorder=5,
    )
    triangle = np.vstack((state_uv, state_uv[0]))
    ax.plot(triangle[:, 0], triangle[:, 1], color="white", linewidth=0.75, zorder=5)
    for index, (point, color) in enumerate(zip(state_uv, STATE_COLORS), start=1):
        ax.scatter(
            *point, s=22, color=color, edgecolor="white", linewidth=0.7, zorder=6
        )
        offsets = ((0.18, -0.25), (0.10, 0.12), (-0.43, -0.18))
        ax.text(
            point[0] + offsets[index - 1][0],
            point[1] + offsets[index - 1][1],
            rf"$S_{index}$",
            color="white",
            fontsize=8,
            fontweight="bold",
            zorder=7,
        )
    ax.set_xlim(zoom_xlim)
    ax.set_ylim(zoom_ylim)
    style_plane_axis(ax)
    ax_d = ax

    fig.subplots_adjust(left=0.035, right=0.99, bottom=0.15, top=0.865)
    # Use a common plot height for c and d. The narrower voltage range in d
    # otherwise stretches that panel vertically above the other three plots.
    fig.canvas.draw()
    c_box = ax_c.get_position()
    d_box = ax_d.get_position()
    d_cell = ax_d.get_subplotspec().get_position(fig)
    d_width = d_box.width * c_box.height / d_box.height
    ax_d.set_position(
        ((d_cell.x0 + d_cell.x1 - d_width) / 2, c_box.y0, d_width, c_box.height)
    )
    headings = (
        "Intensity (model)",
        "Extinction ratio",
        "Full plane section",
        "Optimized states",
    )
    for panel, heading, panel_ax in zip("abcd", headings, (ax_a, ax_b, ax_c, ax_d)):
        cell = panel_ax.get_subplotspec().get_position(fig)
        fig.text(cell.x0, 0.84, panel, fontweight="bold", fontsize=8)
        fig.text((cell.x0 + cell.x1) / 2, 0.84, heading, ha="center", fontweight="bold")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        kwargs = {"dpi": 400} if suffix == "png" else {}
        fig.savefig(OUTPUT.with_suffix(f".{suffix}"), facecolor="white", **kwargs)
    plt.close(fig)
    print(
        f"wrote {OUTPUT.with_suffix('.pdf')} (maximum inter-state span {tight_side:.2f} V)"
    )


if __name__ == "__main__":
    main()
