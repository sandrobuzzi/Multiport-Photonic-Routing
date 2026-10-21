"""Plot programmed electrode voltages and measured optical switching traces.

The two panels share a time axis and use the manuscript output-port convention.
"""

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from tfln_paths import data_path, output_path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paper style  (118 × 56 mm, 7 pt)
# ---------------------------------------------------------------------------
MM = 1 / 25.4
FIG_W = 118 * MM
FIG_H = 56 * MM
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

SAVE_KW = dict(facecolor="white", bbox_inches="tight", pad_inches=0)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
STROBE_NPZ = data_path(
    "fig4/voltage_cube/inputs/uniform_cube_16_strobe_20260505_230858.npz"
)
OUT = output_path("figures/main/switching_demo_118x56.svg")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
ds = np.load(STROBE_NPZ, allow_pickle=True)
t_s = ds["scope_ch1_time"]
ch2 = ds["scope_ch2_voltage"]
ch3 = ds["scope_ch3_voltage"]
ch4 = ds["scope_ch4_voltage"]
t_us = t_s * 1e6
dt = float(t_s[1] - t_s[0])

# ---------------------------------------------------------------------------
# Delay-correct optical outputs
# ---------------------------------------------------------------------------
delays_smp = ds["chirp_seed_d_out_s"] / dt


def fractional_shift(sig, delay_smp):
    n = len(sig)
    src = (np.arange(n) + delay_smp) % n
    i0 = np.floor(src).astype(int) % n
    i1 = (i0 + 1) % n
    f = src - np.floor(src)
    return sig[i0] * (1 - f) + sig[i1] * f


ch2c = fractional_shift(ch2, delays_smp[0])  # D1
ch3c = fractional_shift(ch3, delays_smp[1])  # D2
ch4c = fractional_shift(ch4, delays_smp[2])  # D3

# ---------------------------------------------------------------------------
# Zoom mask  0–500 ns
# ---------------------------------------------------------------------------
mask = (t_us >= 0) & (t_us <= 0.5)
t_ns = t_s[mask] * 1e9

_n = lambda x: (x - x.min()) / (x.max() - x.min())
d1_norm = _n(ch2c[mask])
d2_norm = _n(ch3c[mask])
d3_norm = _n(ch4c[mask])

# ---------------------------------------------------------------------------
# AWG waveform parameters
# ---------------------------------------------------------------------------
T_cycle_ns = float(ds["strobe_t_cycle_ns"])  # 100 ns
n_slots = 5
slot_w_ns = T_cycle_ns / n_slots  # 20 ns

n_idle = int(ds["strobe_n_idle_slots"])
awg_d1 = np.array(ds["strobe_red_awg_v"], dtype=float)
awg_d2 = np.array(ds["strobe_green_awg_v"], dtype=float)
awg_d3 = np.array(ds["strobe_blue_awg_v"], dtype=float)
awg_idle = -(awg_d1 + awg_d2 + awg_d3) / n_idle

# Acquisition order is P1 -> P3 -> P2 -> idle -> idle.
# The displayed cycle starts at P3.
states_v = [awg_d3, awg_d2, awg_idle, awg_idle, awg_d1]
ALPHA = 0.7
n_flat_ns = (1.0 - ALPHA) * slot_w_ns  # 6 ns
n_trans_ns = ALPHA * slot_w_ns  # 14 ns

# ---------------------------------------------------------------------------
# phase_ns from 50 % rising-edge crossings (transition-midpoint method)
# ---------------------------------------------------------------------------
_mask100 = (t_ns >= 0) & (t_ns < T_cycle_ns)
_t100 = t_ns[_mask100]
_d1r = ch2c[mask][_mask100]
_d2r = ch3c[mask][_mask100]
_d3r = ch4c[mask][_mask100]
_slot_idx = [0, 1, 4]


def _half_rise(t, sig, lo, hi):
    win = (t >= lo) & (t <= hi)
    tw, sw = t[win], sig[win]
    thr = 0.5 * sw.max()
    for i in range(len(sw) - 1):
        if sw[i] < thr <= sw[i + 1]:
            f = (thr - sw[i]) / (sw[i + 1] - sw[i])
            return float(tw[i] + f * (tw[i + 1] - tw[i]))
    return float(tw[np.argmax(sw)])


_cross = [
    _half_rise(_t100, _d3r, 0, 20),
    _half_rise(_t100, _d2r, 20, 40),
    _half_rise(_t100, _d1r, 70, 100),
]
_amps = [_d3r.max(), _d2r.max(), _d1r.max()]
_ph_est = [_cross[i] - _slot_idx[i] * slot_w_ns + n_trans_ns / 2 for i in range(3)]
phase_ns = float(np.average(_ph_est, weights=_amps))


# ---------------------------------------------------------------------------
# Reconstruct AWG electrode waveforms
# ---------------------------------------------------------------------------
def build_awg_waveform(t_ns_arr):
    t_cyc = (t_ns_arr - phase_ns) % T_cycle_ns
    s_idx = np.clip(np.floor(t_cyc / slot_w_ns).astype(int), 0, n_slots - 1)
    t_slot = t_cyc - s_idx * slot_w_ns
    K = np.zeros((3, len(t_ns_arr)))
    for ch in range(3):
        v_curr = np.array([states_v[s][ch] for s in range(n_slots)])
        v_next = np.roll(v_curr, -1)
        trans_frac = np.clip((t_slot - n_flat_ns) / n_trans_ns, 0.0, 1.0)
        crossfade = 0.5 * (1.0 - np.cos(np.pi * trans_frac))
        K[ch] = v_curr[s_idx] + (v_next[s_idx] - v_curr[s_idx]) * crossfade
    return K


Kmat = build_awg_waveform(t_ns)
K1, K2, K3 = Kmat[0], Kmat[1], Kmat[2]

# ---------------------------------------------------------------------------
# Slot shading
# ---------------------------------------------------------------------------
slot_colors = ["#4488ff", "#44cc44", "#888888", "#888888", "#ff4444"]
slot_names = ["D_3", "D_2", "idle", "idle", "D_1"]
shade_phase_ns = (phase_ns - n_trans_ns / 2) % T_cycle_ns

BAND_W_NS = 8.0  # narrow band centered on the stable (flat) region of each slot


def shade_slots(ax, xlim, t_shift=0.0):
    for c in range(-1, int(xlim[1] / T_cycle_ns) + 2):
        for s in range(len(slot_names)):
            if slot_names[s] == "idle":
                continue
            center = c * T_cycle_ns + s * slot_w_ns + shade_phase_ns + 10.0 + t_shift
            x0 = center - BAND_W_NS / 2
            x1 = center + BAND_W_NS / 2
            x0c = max(x0, xlim[0])
            x1c = min(x1, xlim[1])
            if x1c > x0c:
                ax.axvspan(x0c, x1c, alpha=0.22, color=slot_colors[s], lw=0)


# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
COL_D1 = "#cc0000"
COL_D2 = "#007700"
COL_D3 = "#0055cc"

# x-axis window: active bands (D1→D3→D2) with a small idle margin each side
_d1_start = shade_phase_ns + 4 * slot_w_ns
_d2_end = shade_phase_ns + T_cycle_ns + 2 * slot_w_ns
_idle_margin = 0.25 * slot_w_ns  # ~5 ns of idle visible on each side
XLIM_abs = (_d1_start - _idle_margin, _d2_end + _idle_margin)

# Shift so the left edge becomes t = 0
t_offset = XLIM_abs[0]
XLIM = (0.0, XLIM_abs[1] - XLIM_abs[0])
shade_phase_ns = shade_phase_ns - t_offset  # in shifted coordinates
t_plot = t_ns - t_offset  # shifted time for plotting

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, (ax_top, ax_bot) = plt.subplots(
    2,
    1,
    figsize=(FIG_W, FIG_H),
    sharex=True,
)

import matplotlib.ticker as mticker


def draw_slot_boundaries(ax, xlim):
    """Grey vertical lines at each slot boundary within xlim."""
    for c in range(-1, int(xlim[1] / T_cycle_ns) + 2):
        for s in range(n_slots):
            x = c * T_cycle_ns + s * slot_w_ns + shade_phase_ns
            if xlim[0] <= x <= xlim[1]:
                ax.axvline(x, color="#bbbbbb", lw=0.5, zorder=0)


# --- Top panel: electrode inputs ---
# Grey boundary lines only — no colour fills
draw_slot_boundaries(ax_top, XLIM)
ax_top.plot(t_plot, K1, color="black", ls="-", label="$V_1$")
ax_top.plot(t_plot, K2, color="black", ls="--", label="$V_2$")
ax_top.plot(t_plot, K3, color="black", ls=":", label="$V_3$")
ax_top.set_ylabel("Voltage (V)")
ax_top.set_xlim(*XLIM)
ax_top.set_ylim(-5.5, 5.5)  # symmetric; data peaks ~±4.6/+3.3 V, top free for legend
ax_top.yaxis.set_major_locator(mticker.MultipleLocator(2))
ax_top.legend(
    loc="upper right",
    bbox_to_anchor=(1.005, 1.04),
    ncol=3,
    columnspacing=0.8,
    handlelength=1.2,
    borderpad=0.3,
    labelspacing=0.2,
    frameon=False,
)
ax_top.tick_params(bottom=False)

# --- Bottom panel: optical outputs ---
# Bands centred at slot midpoints (shade+10); curves shifted +1 ns to match physical peaks
T_CURVE_SHIFT = 1.0
shade_slots(ax_bot, XLIM)
ax_bot.plot(t_plot + T_CURVE_SHIFT, d1_norm, color=COL_D1, label="$P_1$")
ax_bot.plot(t_plot + T_CURVE_SHIFT, d2_norm, color=COL_D2, label="$P_2$")
ax_bot.plot(t_plot + T_CURVE_SHIFT, d3_norm, color=COL_D3, label="$P_3$")
ax_bot.set_xlabel("Time (ns)")
ax_bot.set_ylabel("Norm. power")
ax_bot.set_xlim(*XLIM)
ax_bot.set_ylim(-0.05, 1.28)  # data max = 1.0; top ~0.28 free for legend
ax_bot.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
ax_bot.xaxis.set_major_locator(mticker.MultipleLocator(10))
ax_bot.legend(
    loc="upper right",
    bbox_to_anchor=(1.005, 1.04),
    ncol=3,
    columnspacing=0.8,
    handlelength=1.0,
    borderpad=0.3,
    labelspacing=0.2,
    frameon=False,
)

fig.tight_layout(pad=0.0, h_pad=0.3)
fig.savefig(OUT, **SAVE_KW)
print(f"Saved: {OUT}")
