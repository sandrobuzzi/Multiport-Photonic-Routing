"""
Plot the routing plane used in Fig. 4d.

Single output:
  triangle_plane_filled.svg  — quality-RGB cut-plane, with linear extrapolation
                               outside the measured cube, gray overlay on the
                               extrapolated region, boundary contour, and
                               operating-point dots.

Plane definition
----------------
  ORIGIN  — centroid of the three operating points (centre of the triangle)
  normal  — computed from the R/G/B operating points (fixes the plane in space)

Basis within the plane
-----------------------
  e1 (x-axis) — direction of the K3=const cube-face edges on the cutting plane,
                 i.e. cross(normal, ẑ).  This makes those edges of the hexagonal
                 measured boundary run horizontally, giving the cross-section an
                 axis-aligned appearance (flat top and bottom).

  Sign convention: P_1 (red) and P_2 (green) operating points are placed on the
                   negative-v (lower) side so they appear at the bottom of the plot.
"""

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from tfln_paths import data_path, output_path


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
MM = 1 / 25.4
# Match the archived original SVG canvas, not automatic layout metrics.
FIG_SIZE = (157.68 / 72, 157.68 / 72)
FS = 7

mpl.rcParams.update(
    {
        "font.size": FS,
        "axes.labelsize": FS,
        "xtick.labelsize": FS,
        "ytick.labelsize": FS,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
    }
)

SAVE_KW = dict(format="svg", facecolor="white")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
STROBE_NPZ = data_path(
    "fig4/voltage_cube/inputs/uniform_cube_16_strobe_20260505_230858.npz"
)
FIT_NPZ = data_path("fig4/voltage_cube/inputs/full_cube_fit_reciprocal_MT.npz")
OUT_FILLED = output_path("figures/main/triangle_plane_filled_v4.svg")

GRID_N = 400  # 2-D interpolation resolution


# ---------------------------------------------------------------------------
# Quality colours: P_1→R, P_2→G, P_3→B, normalised by plane max
# ---------------------------------------------------------------------------
def compute_quality_colors(out1, out2, out3, eps=1e-6):
    def norm01(x):
        lo, hi = float(x.min()), float(x.max())
        return (x - lo) / (hi - lo) if hi > lo else np.full_like(x, 0.5)

    r = norm01(out1)
    g = norm01(out2)
    b = norm01(out3)
    lr = 10.0 * np.log10(r + eps)
    lg = 10.0 * np.log10(g + eps)
    lb = 10.0 * np.log10(b + eps)
    qr = np.maximum(lr - np.maximum(lg, lb), 0.0)
    qg = np.maximum(lg - np.maximum(lr, lb), 0.0)
    qb = np.maximum(lb - np.maximum(lr, lg), 0.0)
    q_max = max(float(qr.max()), float(qg.max()), float(qb.max()), 1e-30)
    return np.clip(np.stack([qr, qg, qb], axis=1) / q_max, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Linearly-extrapolating RegularGridInterpolator wrapper
# ---------------------------------------------------------------------------
def _make_linear_extrap(axes, values):
    rgi = RegularGridInterpolator(
        axes, values, method="linear", bounds_error=False, fill_value=None
    )
    grads = np.gradient(values, *axes)
    rgi_g = [
        RegularGridInterpolator(
            axes, g, method="linear", bounds_error=False, fill_value=None
        )
        for g in grads
    ]

    def _eval(pts):
        lo = np.array([a[0] for a in axes])
        hi = np.array([a[-1] for a in axes])
        pts_c = np.clip(pts, lo, hi)
        val = rgi(pts_c)
        delta = pts - pts_c
        if delta.any():
            g_vals = np.stack([rg(pts_c) for rg in rgi_g], axis=1)
            val += (g_vals * delta).sum(axis=1)
        return val

    return _eval


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
ds = np.load(STROBE_NPZ, allow_pickle=True)
fit = np.load(FIT_NPZ, allow_pickle=True)

c1, c2, c3 = fit["c1"], fit["c2"], fit["c3"]  # c1=K3, c2=K2, c3=K1 (physical V)
t_vol = fit[
    "t_cube"
]  # (20,20,20,3); [..0]=CH2/P1, [..1]=CH4/P3, [..2]=CH3/P2; idx [K3,K2,K1]

# Operating points in [K1, K2, K3] physical voltage space
R_pt = np.array(ds["strobe_red_abs_v"], dtype=float)  # D_1
G_pt = np.array(ds["strobe_green_abs_v"], dtype=float)  # D_2
B_pt = np.array(ds["strobe_blue_abs_v"], dtype=float)  # D_3

# Plane normal — defined by the three operating points, independent of basis choice
_t1 = G_pt - R_pt
_t1 /= np.linalg.norm(_t1)
_t2 = B_pt - R_pt
_t2 -= np.dot(_t2, _t1) * _t1
_t2 /= np.linalg.norm(_t2)
normal = np.cross(_t1, _t2)  # unit normal to the operating plane

# ---------------------------------------------------------------------------
# USER PARAMETER
#
#   ORIGIN  — anchor point of the plane; default = centroid of the triangle
# ---------------------------------------------------------------------------
ORIGIN = (R_pt + G_pt + B_pt) / 3.0

# ---------------------------------------------------------------------------
# Orthonormal basis
#
#   e1  —  K3=const face-edge direction: cross(normal, ẑ) = [n_y, -n_x, 0]
#           K3 edges of the hexagonal boundary run horizontally (axis-aligned).
#   Sign:  D1 (red) and P2 (green) at bottom (negative-v side).
# ---------------------------------------------------------------------------
_d3 = np.cross(normal, np.array([0.0, 0.0, 1.0]))  # [n_y, -n_x, 0]
e1 = _d3 / np.linalg.norm(_d3)
e2 = np.cross(normal, e1)
if float((R_pt - ORIGIN) @ e2) > 0:  # ensure D1 (R_pt) at bottom (negative v)
    e1 = -e1
    e2 = np.cross(normal, e1)

# ---------------------------------------------------------------------------
# Model grid (used for the plotting extent)
# ---------------------------------------------------------------------------
Cm1, Cm2, Cm3 = np.meshgrid(c1, c2, c3, indexing="ij")  # c1=K3, c2=K2, c3=K1
mod_abs = np.stack([Cm3.ravel(), Cm2.ravel(), Cm1.ravel()], axis=1)  # [K1, K2, K3]
near = np.abs((mod_abs - ORIGIN) @ normal) < 0.25

# Operating-point (u, v) coordinates relative to ORIGIN
R_u = float((R_pt - ORIGIN) @ e1)
R_v = float((R_pt - ORIGIN) @ e2)
G_u = float((G_pt - ORIGIN) @ e1)
G_v = float((G_pt - ORIGIN) @ e2)
B_u = float((B_pt - ORIGIN) @ e1)
B_v = float((B_pt - ORIGIN) @ e2)

# ---------------------------------------------------------------------------
# Build linearly-extrapolating model evaluators
# ---------------------------------------------------------------------------
axes = (c1, c2, c3)
_eval_d1 = _make_linear_extrap(axes, t_vol[:, :, :, 0])  # CH2/P1 RED
_eval_d2 = _make_linear_extrap(axes, t_vol[:, :, :, 2])  # CH3/P2 GREEN
_eval_d3 = _make_linear_extrap(axes, t_vol[:, :, :, 1])  # CH4/P3 BLUE


def eval_raw_uv(u_flat, v_flat):
    """D1, D2, D3 at (u,v) plane coords; linear extrapolation outside cube."""
    pts = ORIGIN + u_flat[:, None] * e1 + v_flat[:, None] * e2  # [K1, K2, K3]
    q = np.stack([pts[:, 2], pts[:, 1], pts[:, 0]], axis=1)  # [K3, K2, K1]
    return _eval_d1(q), _eval_d2(q), _eval_d3(q)


# ---------------------------------------------------------------------------
# Square 2-D grid with equal axes — same symmetric range on u and v
# ---------------------------------------------------------------------------
mp_u = (mod_abs[near] - ORIGIN) @ e1
mp_v = (mod_abs[near] - ORIGIN) @ e2

# Single half-range so both axes run from -_half to +_half (equal scales and limits)
_half = max(abs(mp_u).max(), abs(mp_v).max()) + 0.5
u_min = -_half
u_max = _half
v_min = -_half
v_max = _half

gu = np.linspace(u_min, u_max, GRID_N)
gv = np.linspace(v_min, v_max, GRID_N)
GU, GV = np.meshgrid(gu, gv)
extent = [u_min, u_max, v_min, v_max]

# ---------------------------------------------------------------------------
# Evaluate model on the 2-D grid
# ---------------------------------------------------------------------------
print("Evaluating model on 2-D plane grid...")
d1_grid, d2_grid, d3_grid = eval_raw_uv(GU.ravel(), GV.ravel())
img_filled = compute_quality_colors(d1_grid, d2_grid, d3_grid).reshape(
    GRID_N, GRID_N, 3
)

# valid_mask2d: inside the measured cube (interpolated) vs outside (extrapolated)
_pts_grid = ORIGIN + GU.ravel()[:, None] * e1 + GV.ravel()[:, None] * e2
valid_mask2d = (
    (
        (_pts_grid[:, 0] >= c3.min())
        & (_pts_grid[:, 0] <= c3.max())
        & (_pts_grid[:, 1] >= c2.min())
        & (_pts_grid[:, 1] <= c2.max())
        & (_pts_grid[:, 2] >= c1.min())
        & (_pts_grid[:, 2] <= c1.max())
    )
    .reshape(GRID_N, GRID_N)
    .astype(float)
)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=FIG_SIZE)

ax.imshow(
    img_filled, origin="lower", extent=extent, aspect="equal", interpolation="bilinear"
)

# Semi-transparent gray overlay on extrapolated region
gray_mask = np.zeros((GRID_N, GRID_N, 4))
gray_mask[~valid_mask2d.astype(bool)] = [0.6, 0.6, 0.6, 0.45]
ax.imshow(
    gray_mask, origin="lower", extent=extent, aspect="equal", interpolation="none"
)

# Boundary contour (interpolated ↔ extrapolated)
ax.contour(GU, GV, valid_mask2d, levels=[0.5], colors="black", linewidths=0.6)

# Operating-point dots
_dot_kw = dict(s=18, zorder=5, linewidths=0.6, edgecolors="white")
ax.scatter([R_u], [R_v], c="#cc0000", **_dot_kw)  # D_1
ax.scatter([G_u], [G_v], c="#009900", **_dot_kw)  # D_2
ax.scatter([B_u], [B_v], c="#0044cc", **_dot_kw)  # D_3

ax.set_xlabel("$e_1$ (V)")
ax.set_ylabel("$e_2$ (V)")
fig.tight_layout(pad=0)

# Identical ticks on both axes at 2 V spacing
_tick_step = 2
_ticks = np.arange(np.ceil(u_min / _tick_step) * _tick_step, u_max + 1e-9, _tick_step)
ax.set_xticks(_ticks)
ax.set_yticks(_ticks)

# Exact plotting rectangle from the original triangle_plane_filled_v4.svg:
# canvas 157.68 pt; SVG x=27.4838, y=1.5579, width=height=130.1962 pt.
# Explicit placement avoids font/backend-dependent tight_layout margins.
ax.set_position(
    [
        27.4838 / 157.68,
        (157.68 - 1.5579 - 130.1962) / 157.68,
        130.1962 / 157.68,
        130.1962 / 157.68,
    ]
)

fig.savefig(OUT_FILLED, **SAVE_KW)
print(f"Saved: {OUT_FILLED}")
plt.close(fig)
