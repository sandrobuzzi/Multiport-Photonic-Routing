#!/usr/bin/env python3
"""Compare the grating reference with all A46 outputs on output-1 calibration."""

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from tfln_paths import data_path, output_path

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from scipy.ndimage import median_filter

HERE = Path(__file__).resolve().parent
REFERENCE_FILE = data_path("insertion_loss/IL_grating_ref_20260531_075632.npz")
DEVICE_FILE = data_path("insertion_loss/IL_A46_device_20260531_075316.npz")
COMBINED_OUTPUT_FILE = output_path(
    "figures/supplementary/a46_power_budget_combined.png"
)

# Physical-output mapping from the reciprocal switching analysis.  Output 1
# and the grating reference use detector 3.  Detector 1 is corrected by the
# established 11.34 - 5.73 = 5.61 dB inter-detector offset used in the A46
# analysis; detectors 2 and 3 used the same -20-dBm power-meter range.
REFERENCE_CHANNEL = "detector_3_watts"
OUTPUT_CHANNELS = {
    "Output 1": ("detector_3_watts", 0.0),
    "Output 2": ("detector_1_watts", 11.34 - 5.73),
    "Output 3": ("detector_2_watts", 0.0),
}
SWITCHING_WAVELENGTH_NM = 1541.66
MEDIAN_POINTS = 101  # 0.101 nm for the 1-pm wavelength grid
FIGURE_WIDTH_IN = 180.0 / 25.4
FIGURE_HEIGHT_IN = FIGURE_WIDTH_IN * 5.0 / 13.4


plt.rcParams.update(
    {
        "font.size": 7,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
    }
)


def dbm(power_w: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(power_w, 1e-15) / 1e-3)


with np.load(REFERENCE_FILE, allow_pickle=False) as reference, np.load(
    DEVICE_FILE, allow_pickle=False
) as device:
    wavelength_nm = reference["wavelengths"]
    if not np.array_equal(wavelength_nm, device["wavelengths"]):
        raise ValueError("Reference and device wavelength axes do not match")

    reference_w = np.maximum(reference[REFERENCE_CHANNEL], 0.0)
    outputs_w = {
        label: np.maximum(device[channel], 0.0) * 10.0 ** (offset_db / 10.0)
        for label, (channel, offset_db) in OUTPUT_CHANNELS.items()
    }

reference_smooth_w = median_filter(reference_w, MEDIAN_POINTS, mode="nearest")
outputs_smooth_w = {
    label: median_filter(power_w, MEDIAN_POINTS, mode="nearest")
    for label, power_w in outputs_w.items()
}
total_smooth_w = sum(outputs_smooth_w.values())
excess_losses_db = {
    label: -10.0
    * np.log10(np.maximum(power_w, 1e-15) / np.maximum(reference_smooth_w, 1e-15))
    for label, power_w in outputs_smooth_w.items()
}
total_excess_loss_db = -10.0 * np.log10(
    np.maximum(total_smooth_w, 1e-15) / np.maximum(reference_smooth_w, 1e-15)
)

operating_index = int(np.argmin(np.abs(wavelength_nm - SWITCHING_WAVELENGTH_NM)))
reference_operating_dbm = float(dbm(reference_smooth_w)[operating_index])
output_operating_dbm = {
    label: float(dbm(power_w)[operating_index])
    for label, power_w in outputs_smooth_w.items()
}
excess_operating_db = {
    label: float(loss_db[operating_index])
    for label, loss_db in excess_losses_db.items()
}
total_operating_dbm = float(dbm(total_smooth_w)[operating_index])
total_excess_operating_db = float(total_excess_loss_db[operating_index])

output_colors = {
    # Match the manuscript's P1/P2/P3 convention: red, green, blue.
    "Output 1": "#c44e52",
    "Output 2": "#55a868",
    "Output 3": "#4c72b0",
}
output_draw_order = ("Output 2", "Output 3", "Output 1")
output_zorder = {"Output 2": 1, "Output 3": 2, "Output 1": 3}


def plot_power_panel(ax: plt.Axes, mask: np.ndarray, show_sum: bool = True) -> None:
    ax.plot(
        wavelength_nm[mask],
        dbm(reference_smooth_w)[mask],
        color="black",
        linewidth=1.6,
        zorder=5,
        label="Grating reference",
    )
    # Draw output 2 first so that all three output traces remain
    # legible where the three outputs overlap.
    for label in output_draw_order:
        power_w = outputs_smooth_w[label]
        ax.plot(
            wavelength_nm[mask],
            dbm(power_w)[mask],
            color=output_colors[label],
            linewidth=1.15,
            zorder=output_zorder[label],
            label={"Output 1": r"$P_1$", "Output 2": r"$P_2$", "Output 3": r"$P_3$"}[
                label
            ],
        )
    if show_sum:
        ax.plot(
            wavelength_nm[mask],
            dbm(total_smooth_w)[mask],
            color="#666666",
            linewidth=1.35,
            linestyle="--",
            zorder=4,
            label="Output sum",
        )
    ax.axvline(
        SWITCHING_WAVELENGTH_NM,
        color="#d18f00",
        linestyle=":",
        linewidth=1.1,
        zorder=6,
    )
    ax.grid(alpha=0.18)


# Combine the useful-band overview and switching-region detail in one figure.
wide_mask = (wavelength_nm >= 1520.0) & (wavelength_nm <= 1560.0)
zoom_mask = (wavelength_nm >= 1541.25) & (wavelength_nm <= 1542.05)
figure, axes = plt.subplots(
    1,
    2,
    figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN),
    layout="constrained",
    gridspec_kw={"width_ratios": (1.22, 1.0)},
)

plot_power_panel(axes[0], wide_mask)
axes[0].set(
    xlabel="Wavelength (nm)",
    ylabel="Calibration-corrected power (dBm)",
    ylim=(-42.0, -10.0),
    title="Useful-band overview",
)
axes[0].text(
    -0.04,
    1.05,
    "a",
    transform=axes[0].transAxes,
    ha="right",
    va="bottom",
    fontweight="bold",
    fontsize=8,
    clip_on=False,
)

plot_power_panel(axes[1], zoom_mask)
axes[1].set(
    xlabel="Wavelength (nm)",
    ylabel="Calibration-corrected power (dBm)",
    ylim=(-48.0, -11.0),
    title="Switching-region detail",
)
axes[1].xaxis.set_major_locator(MultipleLocator(0.2))
axes[1].text(
    -0.04,
    1.05,
    "b",
    transform=axes[1].transAxes,
    ha="right",
    va="bottom",
    fontweight="bold",
    fontsize=8,
    clip_on=False,
)
for label in output_draw_order:
    axes[1].scatter(
        [SWITCHING_WAVELENGTH_NM],
        [output_operating_dbm[label]],
        color=output_colors[label],
        s=24,
        zorder=7,
    )
axes[1].scatter(
    [SWITCHING_WAVELENGTH_NM], [reference_operating_dbm], color="black", s=26, zorder=7
)
axes[1].text(
    0.97,
    0.04,
    "1541.66 nm\n"
    f"reference: {reference_operating_dbm:.2f} dBm\n"
    f"corrected sum: {total_operating_dbm:.2f} dBm\n"
    f"summed excess loss: {total_excess_operating_db:.2f} dB",
    transform=axes[1].transAxes,
    ha="right",
    va="bottom",
    fontsize=7,
    zorder=8,
    bbox={"facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 1.0, "pad": 4},
)

handles, labels = axes[0].get_legend_handles_labels()
legend_items = dict(zip(labels, handles))
labels = ["Grating reference", r"$P_1$", r"$P_2$", r"$P_3$", "Output sum"]
handles = [legend_items[label] for label in labels]
figure.legend(
    handles,
    labels,
    frameon=False,
    ncol=5,
    fontsize=7,
    loc="outside lower center",
)
figure.suptitle("A46 calibration-corrected power budget", fontsize=8)
figure.savefig(COMBINED_OUTPUT_FILE, dpi=300)
figure.savefig(COMBINED_OUTPUT_FILE.with_suffix(".svg"))

print(
    f"Reference at {wavelength_nm[operating_index]:.3f} nm: {reference_operating_dbm:.3f} dBm"
)
for label in outputs_smooth_w:
    print(
        f"A46 {label.lower()}: {output_operating_dbm[label]:.3f} dBm; "
        f"excess loss {excess_operating_db[label]:.3f} dB"
    )
print(f"Corrected output sum: {total_operating_dbm:.3f} dBm")
print(f"Summed-output excess loss: {total_excess_operating_db:.3f} dB")
print(f"Saved {COMBINED_OUTPUT_FILE}")
