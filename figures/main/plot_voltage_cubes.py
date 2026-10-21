"""Plot measured and fitted voltage cubes, routing contrast and optical traces.

The output convention is P1/CH2 red, P2/CH3 green, P3/CH4 blue.
"""

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from tfln_paths import data_path, output_path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ---------------------------------------------------------------------------
# Paper style
# ---------------------------------------------------------------------------
MM = 1 / 25.4
FIG_SIZE = (55.824 * MM, 55.824 * MM)
FS = 7

mpl.rcParams.update(
    {
        "font.size": FS,
        "axes.titlesize": FS,
        "axes.labelsize": FS,
        "xtick.labelsize": FS,
        "ytick.labelsize": FS,
        "legend.fontsize": FS,
        "figure.titlesize": FS,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
        "lines.linewidth": 0.8,
        "patch.linewidth": 0.5,
    }
)

SAVE_KW = dict(format="svg", facecolor="white")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
STROBE_NPZ = data_path(
    "fig4/voltage_cube/inputs/uniform_cube_16_strobe_20260505_230858.npz"
)
REC_NPZ = data_path("fig4/voltage_cube/inputs/reconstructed_exp_cube.npz")
FIT_NPZ = data_path("fig4/voltage_cube/inputs/full_cube_fit_reciprocal_MT.npz")

OUT1 = output_path("figures/main/uniform_cube_16_strobe_rgb_waveform.svg")
OUT2 = output_path("figures/main/uniform_cube_16_strobe_rgb_quality.svg")
OUT3 = output_path("figures/main/uniform_cube_16_strobe_rgb_cube_exp.svg")
OUT4 = output_path("figures/main/uniform_cube_16_strobe_rgb_cube_fit.svg")
OUT5 = output_path("figures/main/uniform_cube_16_strobe_rgb_cube_quality.svg")
OUT6 = output_path("figures/main/uniform_cube_16_strobe_rgb_slice2d.svg")
OUT7 = output_path("figures/main/uniform_cube_16_strobe_rgb_slice2d_interp.svg")
OUT8 = output_path("figures/main/uniform_cube_16_strobe_rgb_waveform_zoom.svg")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
AWG_OUTPUT_RANGE_V = 5.0
AWG_AMPLITUDE = 1
N_VOXELS = 15
VOXEL_RANGE = AWG_OUTPUT_RANGE_V


# ---------------------------------------------------------------------------
# Cube helpers
# ---------------------------------------------------------------------------
def voxelise(xs, ys, zs, colors, n_voxels=N_VOXELS, v_range=None):
    if v_range is None:
        v_range = VOXEL_RANGE
    edges = np.linspace(-v_range, v_range, n_voxels + 1)
    ix = np.clip(np.digitize(xs, edges) - 1, 0, n_voxels - 1)
    iy = np.clip(np.digitize(ys, edges) - 1, 0, n_voxels - 1)
    iz = np.clip(np.digitize(zs, edges) - 1, 0, n_voxels - 1)
    cnt = np.zeros((n_voxels, n_voxels, n_voxels), dtype=np.float64)
    col = np.zeros((n_voxels, n_voxels, n_voxels, 3), dtype=np.float64)
    np.add.at(cnt, (ix, iy, iz), 1)
    for c in range(3):
        np.add.at(col[:, :, :, c], (ix, iy, iz), colors[:, c])
    filled = cnt > 0
    for c in range(3):
        col[filled, c] /= cnt[filled]
    fi, fj, fk = np.nonzero(filled)
    mids = (edges[:-1] + edges[1:]) / 2.0
    cx, cy, cz = mids[fi], mids[fj], mids[fk]
    avg_col = col[fi, fj, fk]
    mx = avg_col.max(axis=1)
    chroma = mx - (avg_col.sum(axis=1) - mx) / 2.0
    return cx, cy, cz, avg_col, cnt[fi, fj, fk], chroma


def _sobol_display_points(vx, vy, vz, vcol, vchroma, n_voxels, v_range):
    from scipy.stats.qmc import Sobol as _Sobol

    edges = np.linspace(-v_range, v_range, n_voxels + 1)
    vox_col = np.full((n_voxels, n_voxels, n_voxels, 3), -1.0)
    vox_chr = np.full((n_voxels, n_voxels, n_voxels), -1.0)
    fi = np.clip(np.digitize(vx, edges) - 1, 0, n_voxels - 1)
    fj = np.clip(np.digitize(vy, edges) - 1, 0, n_voxels - 1)
    fk = np.clip(np.digitize(vz, edges) - 1, 0, n_voxels - 1)
    vox_col[fi, fj, fk] = vcol
    vox_chr[fi, fj, fk] = vchroma
    pts = _Sobol(d=3, scramble=True, seed=42).random(32768) * 2 * v_range - v_range
    si = np.clip(np.digitize(pts[:, 0], edges) - 1, 0, n_voxels - 1)
    sj = np.clip(np.digitize(pts[:, 1], edges) - 1, 0, n_voxels - 1)
    sk = np.clip(np.digitize(pts[:, 2], edges) - 1, 0, n_voxels - 1)
    s_col = vox_col[si, sj, sk]
    s_chr = vox_chr[si, sj, sk]
    valid = s_chr >= 0
    return pts[valid, 0], pts[valid, 1], pts[valid, 2], s_col[valid], s_chr[valid]


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
    q_global_max = max(float(qr.max()), float(qg.max()), float(qb.max()), 1e-30)
    # P_1→R, P_2→G, P_3→B
    return np.clip(np.stack([qr, qg, qb], axis=1) / q_global_max, 0.0, 1.0)


def _make_cube_fig(
    awg1,
    awg2,
    awg3,
    out1,
    out2,
    out3,
    k_biases,
    colors,
    chroma_or_q,
    save_path,
    best_points=None,
):
    """Paper-ready cube figure: 56×56 mm, 7 pt fonts, no slider."""
    print(f"  Voxelising {len(awg1)} pts into {N_VOXELS}^3 grid...")
    vx, vy, vz, vcol, _, vch = voxelise(awg1, awg2, awg3, colors)
    print(f"  Filled voxels: {len(vx)} / {N_VOXELS**3}")

    if chroma_or_q == "quality":
        eps = 1e-6

        def _n01(x):
            lo, hi = float(x.min()), float(x.max())
            return (x - lo) / (hi - lo) if hi > lo else np.full_like(x, 0.5)

        _r, _g, _b = _n01(out1), _n01(out2), _n01(out3)
        _lr = 10 * np.log10(_r + eps)
        _lg = 10 * np.log10(_g + eps)
        _lb = 10 * np.log10(_b + eps)
        _qmag = np.maximum.reduce(
            [
                np.maximum(_lr - np.maximum(_lg, _lb), 0.0),
                np.maximum(_lg - np.maximum(_lr, _lb), 0.0),
                np.maximum(_lb - np.maximum(_lr, _lg), 0.0),
            ]
        )
        edges = np.linspace(-VOXEL_RANGE, VOXEL_RANGE, N_VOXELS + 1)
        _ix = np.clip(np.digitize(awg1, edges) - 1, 0, N_VOXELS - 1)
        _iy = np.clip(np.digitize(awg2, edges) - 1, 0, N_VOXELS - 1)
        _iz = np.clip(np.digitize(awg3, edges) - 1, 0, N_VOXELS - 1)
        _vq = np.zeros((N_VOXELS,) * 3)
        _vc2 = np.zeros((N_VOXELS,) * 3)
        np.add.at(_vq, (_ix, _iy, _iz), _qmag)
        np.add.at(_vc2, (_ix, _iy, _iz), 1)
        _f = _vc2 > 0
        _vq[_f] /= _vc2[_f]
        _fi = np.clip(np.digitize(vx, edges) - 1, 0, N_VOXELS - 1)
        _fj = np.clip(np.digitize(vy, edges) - 1, 0, N_VOXELS - 1)
        _fk = np.clip(np.digitize(vz, edges) - 1, 0, N_VOXELS - 1)
        vch = _vq[_fi, _fj, _fk]

    sx, sy, sz, s_col, s_val = _sobol_display_points(
        vx, vy, vz, vcol, vch, N_VOXELS, VOXEL_RANGE
    )
    sx_d = sx + k_biases[0]
    sy_d = sy + k_biases[1]
    sz_d = sz + k_biases[2]

    fig = plt.figure(figsize=FIG_SIZE)
    ax = fig.add_axes([0, 0, 1, 1], projection="3d")

    ax.set_proj_type("ortho")
    ax.set_box_aspect([1, 1, 1])

    # Hide all default panes and grid lines
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("none")
    ax.grid(False)

    # Wireframe: 3 back edges (hidden vertex xl,yl,zl) drawn before scatter,
    # 9 front edges drawn after scatter — creates depth illusion.
    # Viewer is at +x+y+z (azim=45, elev≈35), so the hidden corner is (xl,yl,zl).
    xl, xh = k_biases[0] - 5.2, k_biases[0] + 5.2
    yl, yh = k_biases[1] - 5.2, k_biases[1] + 5.2
    zl, zh = k_biases[2] - 5.2, k_biases[2] + 5.2
    _ew, _ec = 0.4, "#888888"
    _back_edges = [
        ([xl, xh], [yl, yl], [zl, zl]),  # bottom-front edge (hidden corner along x)
        ([xl, xl], [yl, yh], [zl, zl]),  # bottom-left edge  (hidden corner along y)
        ([xl, xl], [yl, yl], [zl, zh]),  # left-front vertical (hidden corner along z)
    ]
    _front_edges = [
        # bottom face (remaining 3)
        ([xl, xh], [yh, yh], [zl, zl]),
        ([xh, xh], [yl, yh], [zl, zl]),
        # top face (all 4)
        ([xl, xh], [yl, yl], [zh, zh]),
        ([xl, xh], [yh, yh], [zh, zh]),
        ([xl, xl], [yl, yh], [zh, zh]),
        ([xh, xh], [yl, yh], [zh, zh]),
        # vertical edges (remaining 3)
        ([xh, xh], [yl, yl], [zl, zh]),
        ([xl, xl], [yh, yh], [zl, zh]),
        ([xh, xh], [yh, yh], [zl, zh]),
    ]
    for xs, ys, zs in _back_edges:
        ax.plot(xs, ys, zs, color=_ec, lw=_ew, zorder=1, linestyle="--", dashes=(3, 3))

    ax.scatter(
        sx_d,
        sy_d,
        sz_d,
        c=s_col,
        s=2,
        alpha=1.0,
        linewidths=0,
        depthshade=True,
        rasterized=True,
        zorder=2,
    )

    for xs, ys, zs in _front_edges:
        ax.plot(xs, ys, zs, color=_ec, lw=_ew, zorder=3)

    ax.set_xlim(k_biases[0] - 5.2, k_biases[0] + 5.2)
    ax.set_ylim(k_biases[1] - 5.2, k_biases[1] + 5.2)
    ax.set_zlim(k_biases[2] - 5.2, k_biases[2] + 5.2)
    # 2 ticks per axis prevents bottom-corner collisions between X and Y labels
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=2, integer=True))
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=2, integer=True))
    ax.zaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=2, integer=True))

    # X/Y: pull labels up toward cube bottom; Z: push labels left into empty space
    ax.tick_params(
        axis="x",
        which="major",
        pad=-5,
        labelsize=FS,
        colors="#888888",
        labelcolor="black",
        width=_ew,
    )
    ax.tick_params(
        axis="y",
        which="major",
        pad=-5,
        labelsize=FS,
        colors="#888888",
        labelcolor="black",
        width=_ew,
    )
    ax.tick_params(
        axis="z",
        which="major",
        pad=-2,
        labelsize=FS,
        colors="#888888",
        labelcolor="black",
        width=_ew,
    )

    # Place axis labels as 2D annotations — avoids projection-dependent overlaps
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.text2D(
        0.80,
        0.06,
        "V$_1$ (V)",
        transform=ax.transAxes,
        fontsize=FS,
        ha="center",
        va="top",
        color="black",
    )
    ax.text2D(
        0.20,
        0.06,
        "V$_2$ (V)",
        transform=ax.transAxes,
        fontsize=FS,
        ha="center",
        va="top",
        color="black",
    )
    ax.text2D(
        0.01,
        0.50,
        "V$_3$ (V)",
        transform=ax.transAxes,
        fontsize=FS,
        ha="center",
        va="center",
        rotation=90,
        color="black",
    )

    ax.view_init(elev=np.degrees(np.arctan(1 / np.sqrt(2))), azim=45)
    ax.set_title("")

    if save_path:
        fig.savefig(save_path, dpi=900, **SAVE_KW)
        print(f"  Saved: {save_path}")
    return fig


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
ds = np.load(STROBE_NPZ, allow_pickle=True)
rec = np.load(REC_NPZ, allow_pickle=True)
fit = np.load(FIT_NPZ, allow_pickle=True)

k_biases = np.array(
    [float(ds["k1_voltage"]), float(ds["k2_voltage"]), float(ds["k3_voltage"])],
    dtype=np.float64,
)

# Experimental cube: physical → AWG-relative
V1r = rec["V1_raw"] - k_biases[0]
V2r = rec["V2_raw"] - k_biases[1]
V3r = rec["V3_raw"] - k_biases[2]
O1r = rec["out1_raw"]
O2r = rec["out2_raw"]
O3r = rec["out3_raw"]

# Reciprocal M^T fit: c1=K3, c2=K2, c3=K1; t_cube detector
# order is [0]=CH2/P1, [1]=CH4/P3, [2]=CH3/P2.
c1, c2, c3 = fit["c1"], fit["c2"], fit["c3"]
Cm1, Cm2, Cm3 = np.meshgrid(c1, c2, c3, indexing="ij")
Cf_K3 = Cm1.ravel() - k_biases[2]
Cf_K2 = Cm2.ravel() - k_biases[1]
Cf_K1 = Cm3.ravel() - k_biases[0]
t = fit["t_cube"]
Pf1 = t[..., 0].ravel()  # CH2; displayed red
Pf2 = t[..., 2].ravel()  # CH3/P2; displayed green
Pf3 = t[..., 1].ravel()  # CH4/P3; displayed blue

best_points = {
    "red": {"abs_v": ds["strobe_red_abs_v"], "Q_dB": float(ds["strobe_red_q_db"])},
    "green": {
        "abs_v": ds["strobe_green_abs_v"],
        "Q_dB": float(ds["strobe_green_q_db"]),
    },
    "blue": {"abs_v": ds["strobe_blue_abs_v"], "Q_dB": float(ds["strobe_blue_q_db"])},
}

# Shared color normalization across experimental and model
_ch1_lo = min(float(O1r.min()), float(Pf1.min()))
_ch1_hi = max(float(O1r.max()), float(Pf1.max()))
_ch2_lo = min(float(O2r.min()), float(Pf2.min()))
_ch2_hi = max(float(O2r.max()), float(Pf2.max()))
_ch3_lo = min(float(O3r.min()), float(Pf3.min()))
_ch3_hi = max(float(O3r.max()), float(Pf3.max()))


def shared_norm(o1, o2, o3):
    r = np.clip((o1 - _ch1_lo) / (_ch1_hi - _ch1_lo), 0, 1)  # D_1 → R
    g = np.clip((o2 - _ch2_lo) / (_ch2_hi - _ch2_lo), 0, 1)  # P_2 → G
    b = np.clip((o3 - _ch3_lo) / (_ch3_hi - _ch3_lo), 0, 1)  # P_3 → B
    return np.stack([r, g, b], axis=1)


# ---------------------------------------------------------------------------
# Waveform data preparation
# ---------------------------------------------------------------------------
t_s = ds["scope_ch1_time"]
ch2 = ds["scope_ch2_voltage"]
ch3 = ds["scope_ch3_voltage"]
ch4 = ds["scope_ch4_voltage"]
t_us = t_s * 1e6
dt = float(t_s[1] - t_s[0])

delays_smp = ds["chirp_seed_d_out_s"] / dt


def fractional_shift(sig, delay_smp):
    n = len(sig)
    src = (np.arange(n) + delay_smp) % n
    i0 = np.floor(src).astype(int) % n
    i1 = (i0 + 1) % n
    f = src - np.floor(src)
    return sig[i0] * (1 - f) + sig[i1] * f


ch2c = fractional_shift(ch2, delays_smp[0])
ch3c = fractional_shift(ch3, delays_smp[1])
ch4c = fractional_shift(ch4, delays_smp[2])

T_cycle_ns = 100.0
slot_w_ns = 20.0
phase_ns = 10.1
slot_colors = ["#4488ff", "#44cc44", "#888888", "#888888", "#ff4444"]
slot_names = ["D_3", "D_2", "idle", "idle", "D_1"]


def shade_slots(ax, xlim=(0, 500), offset=0.0):
    for c in range(-1, int((xlim[1] + offset) / T_cycle_ns) + 2):
        for s in range(len(slot_names)):
            x0 = c * T_cycle_ns + s * slot_w_ns + phase_ns - offset
            x1 = x0 + slot_w_ns
            x0c = max(x0, xlim[0])
            x1c = min(x1, xlim[1])
            if x1c > x0c:
                ax.axvspan(x0c, x1c, alpha=0.18, color=slot_colors[s], lw=0)


mask_zoom = (t_us >= 0) & (t_us <= 0.5)
t_z_ns = t_s[mask_zoom] * 1e9
_n = lambda x: (x - x.min()) / (x.max() - x.min())
ch2z_n = _n(ch2c[mask_zoom])
ch3z_n = _n(ch3c[mask_zoom])
ch4z_n = _n(ch4c[mask_zoom])

mask_full = (t_us >= 0) & (t_us <= 25)
t_us_full = t_us[mask_full]
d1_f = ch2c[mask_full]
d2_f = ch3c[mask_full]
d3_f = ch4c[mask_full]
eps = 1e-6
d1n = np.clip((d1_f - d1_f.min()) / (d1_f.max() - d1_f.min()), 0, None)
d2n = np.clip((d2_f - d2_f.min()) / (d2_f.max() - d2_f.min()), 0, None)
d3n = np.clip((d3_f - d3_f.min()) / (d3_f.max() - d3_f.min()), 0, None)
Q_D1 = 10 * np.log10(d1n + eps) - np.maximum(
    10 * np.log10(d2n + eps), 10 * np.log10(d3n + eps)
)
Q_D2 = 10 * np.log10(d2n + eps) - np.maximum(
    10 * np.log10(d1n + eps), 10 * np.log10(d3n + eps)
)
Q_D3 = 10 * np.log10(d3n + eps) - np.maximum(
    10 * np.log10(d1n + eps), 10 * np.log10(d2n + eps)
)
mask_zq = (t_us_full >= 0) & (t_us_full <= 0.5)
t_zq_ns = t_us_full[mask_zq] * 1000


# ---------------------------------------------------------------------------
# Fig 1 — strobe waveform 500 ns zoom
# ---------------------------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=FIG_SIZE)
ax1.plot(t_z_ns, ch2z_n, color="#cc0000", label="$P_1$")
ax1.plot(t_z_ns, ch3z_n, color="#007700", label="$P_2$")
ax1.plot(t_z_ns, ch4z_n, color="#0055cc", label="$P_3$")
shade_slots(ax1)
ax1.set_xlabel("Time (ns)")
ax1.set_ylabel("Normalized power (a.u.)")
ax1.legend(loc="upper right")
ax1.set_xlim(0, 500)
ax1.set_ylim(-0.05, 1.15)
ax1.grid(color="#cccccc", lw=0.3)
fig1.tight_layout(pad=0)
fig1.savefig(OUT1, **SAVE_KW)
print(f"Saved: {OUT1}")

# ---------------------------------------------------------------------------
# Fig 8 — single-cycle zoom
# ---------------------------------------------------------------------------
_t_shift = 80.0
fig8, ax8 = plt.subplots(figsize=FIG_SIZE)
ax8.plot(t_z_ns - _t_shift, ch2z_n, color="#cc0000", label="$P_1$")
ax8.plot(t_z_ns - _t_shift, ch3z_n, color="#007700", label="$P_2$")
ax8.plot(t_z_ns - _t_shift, ch4z_n, color="#0055cc", label="$P_3$")
shade_slots(ax8, xlim=(0, 80), offset=_t_shift)
ax8.set_xlabel("Time (ns)")
ax8.set_ylabel("Normalized power (a.u.)")
ax8.legend(loc="upper right")
ax8.set_xlim(0, 80)
ax8.set_ylim(-0.05, 1.15)
ax8.grid(color="#cccccc", lw=0.3)
fig8.tight_layout(pad=0)
fig8.savefig(OUT8, **SAVE_KW)
print(f"Saved: {OUT8}")

# ---------------------------------------------------------------------------
# Fig 2 — switching quality vs time
# ---------------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=FIG_SIZE)
shade_slots(ax2)
ax2.plot(t_zq_ns, Q_D1[mask_zq], color="#cc0000", label="$P_1$")
ax2.plot(t_zq_ns, Q_D2[mask_zq], color="#007700", label="$P_2$")
ax2.plot(t_zq_ns, Q_D3[mask_zq], color="#0055cc", label="$P_3$")
ax2.axhline(0, color="#aaaaaa", lw=0.5, ls="--")
ax2.set_xlabel("Time (ns)")
ax2.set_ylabel("Isolation (dB)")
ax2.legend(loc="upper right")
ax2.set_xlim(0, 500)
ax2.grid(color="#cccccc", lw=0.3)
fig2.tight_layout(pad=0)
fig2.savefig(OUT2, **SAVE_KW)
print(f"Saved: {OUT2}")

# ---------------------------------------------------------------------------
# Fig 3 — Experimental cube
# ---------------------------------------------------------------------------
col_exp = shared_norm(O1r, O2r, O3r)
fig3 = _make_cube_fig(
    V1r,
    V2r,
    V3r,
    O1r,
    O2r,
    O3r,
    k_biases,
    col_exp,
    "chroma",
    OUT3,
    best_points=best_points,
)

# ---------------------------------------------------------------------------
# Fig 4 — Fitted model cube
# ---------------------------------------------------------------------------
col_fit = shared_norm(Pf1, Pf2, Pf3)
fig4 = _make_cube_fig(
    Cf_K1,
    Cf_K2,
    Cf_K3,
    Pf1,
    Pf2,
    Pf3,
    k_biases,
    col_fit,
    "chroma",
    OUT4,
    best_points=best_points,
)

# ---------------------------------------------------------------------------
# Fig 5 — Quality cube
# ---------------------------------------------------------------------------
col_q = compute_quality_colors(O1r, O2r, O3r)
fig5 = _make_cube_fig(
    V1r,
    V2r,
    V3r,
    O1r,
    O2r,
    O3r,
    k_biases,
    col_q,
    "quality",
    OUT5,
    best_points=best_points,
)

plt.close(fig3)
plt.close(fig4)
plt.close(fig5)

# ---------------------------------------------------------------------------
# 2D plane geometry shared by Figs 6 & 7
# ---------------------------------------------------------------------------
print("Building Fig 6 - 2D plane slice...")

R_pt = np.array(best_points["red"]["abs_v"], dtype=np.float64)
G_pt = np.array(best_points["green"]["abs_v"], dtype=np.float64)
B_pt = np.array(best_points["blue"]["abs_v"], dtype=np.float64)

e1 = G_pt - R_pt
e1 = e1 / np.linalg.norm(e1)
v2 = B_pt - R_pt
v2 = v2 - np.dot(v2, e1) * e1
e2 = v2 / np.linalg.norm(v2)
normal = np.cross(e1, e2)


def proj2d(pts_3d):
    d = pts_3d - R_pt
    return np.dot(d, e1), np.dot(d, e2)


R_u, R_v = proj2d(R_pt)
G_u, G_v = proj2d(G_pt)
B_u, B_v = proj2d(B_pt)

PLANE_TOL_V = 0.5

exp_phys = np.stack([V1r + k_biases[0], V2r + k_biases[1], V3r + k_biases[2]], axis=1)
dist_exp = np.abs((exp_phys - R_pt) @ normal)
mask_ep = dist_exp < PLANE_TOL_V
print(f"  Experimental pts within {PLANE_TOL_V} V: {mask_ep.sum()}")
ep_u = (exp_phys[mask_ep] - R_pt) @ e1
ep_v = (exp_phys[mask_ep] - R_pt) @ e2
col_ep = shared_norm(O1r[mask_ep], O2r[mask_ep], O3r[mask_ep])

mod_phys = np.stack(
    [Cf_K1 + k_biases[0], Cf_K2 + k_biases[1], Cf_K3 + k_biases[2]], axis=1
)
dist_mod = np.abs((mod_phys - R_pt) @ normal)
mask_mp = dist_mod < PLANE_TOL_V
print(f"  Model pts within {PLANE_TOL_V} V: {mask_mp.sum()}")
mp_u = (mod_phys[mask_mp] - R_pt) @ e1
mp_v = (mod_phys[mask_mp] - R_pt) @ e2
col_mp = shared_norm(Pf1[mask_mp], Pf2[mask_mp], Pf3[mask_mp])

_star_kw = dict(marker="*", s=20, zorder=10, linewidths=0.4)

# ---------------------------------------------------------------------------
# Fig 6 — 2D scatter slice
# ---------------------------------------------------------------------------
fig6, axes6 = plt.subplots(1, 2, figsize=FIG_SIZE)
for ax, u, v, col, ttl in [
    (axes6[0], ep_u, ep_v, col_ep, "Experiment"),
    (axes6[1], mp_u, mp_v, col_mp, "Model"),
]:
    ax.scatter(u, v, c=col, s=2, alpha=0.6, linewidths=0)
    ax.scatter([R_u], [R_v], c="red", edgecolors="k", **_star_kw)
    ax.scatter([G_u], [G_v], c="lime", edgecolors="k", **_star_kw)
    ax.scatter([B_u], [B_v], c="blue", edgecolors="k", **_star_kw)
    ax.set_xlabel("$e_1$ (V)")
    ax.set_ylabel("$e_2$ (V)")
    ax.set_title(ttl)
    ax.set_aspect("equal")
    ax.grid(color="#cccccc", lw=0.3)
fig6.tight_layout(pad=0)
fig6.savefig(OUT6, **SAVE_KW)
print(f"Saved: {OUT6}")

# ---------------------------------------------------------------------------
# Fig 7 — 2D interpolated color maps
# ---------------------------------------------------------------------------
print("Building Fig 7 - 2D interpolated color maps...")
from scipy.interpolate import griddata

GRID_N = 300
u_min = min(ep_u.min(), mp_u.min()) - 0.2
u_max = max(ep_u.max(), mp_u.max()) + 0.2
v_min = min(ep_v.min(), mp_v.min()) - 0.2
v_max = max(ep_v.max(), mp_v.max()) + 0.2
gu, gv = np.linspace(u_min, u_max, GRID_N), np.linspace(v_min, v_max, GRID_N)
GU, GV = np.meshgrid(gu, gv)
grid_pts = np.stack([GU.ravel(), GV.ravel()], axis=1)


def interp_rgb(u, v, col, method="linear"):
    img = np.full((GRID_N * GRID_N, 3), np.nan)
    for ch in range(3):
        img[:, ch] = griddata(
            np.stack([u, v], axis=1), col[:, ch], grid_pts, method=method
        )
    return np.clip(img.reshape(GRID_N, GRID_N, 3), 0, 1)


def nan_to_white(img):
    out = img.copy()
    out[np.any(np.isnan(img), axis=2)] = 1.0
    return out


img_exp_w = nan_to_white(interp_rgb(ep_u, ep_v, col_ep))
img_mod_w = nan_to_white(interp_rgb(mp_u, mp_v, col_mp))
extent = [u_min, u_max, v_min, v_max]

fig7, axes7 = plt.subplots(1, 2, figsize=FIG_SIZE)
for ax, img, ttl in [
    (axes7[0], img_exp_w, "Experiment"),
    (axes7[1], img_mod_w, "Model"),
]:
    ax.imshow(
        img, origin="lower", extent=extent, aspect="equal", interpolation="nearest"
    )
    ax.scatter([R_u], [R_v], c="red", edgecolors="k", **_star_kw)
    ax.scatter([G_u], [G_v], c="lime", edgecolors="k", **_star_kw)
    ax.scatter([B_u], [B_v], c="blue", edgecolors="k", **_star_kw)
    ax.set_xlabel("$e_1$ (V)")
    ax.set_ylabel("$e_2$ (V)")
    ax.set_title(ttl)
fig7.tight_layout(pad=0)
fig7.savefig(OUT7, **SAVE_KW)
print(f"Saved: {OUT7}")
