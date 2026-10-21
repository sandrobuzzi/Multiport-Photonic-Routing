#!/usr/bin/env python3
"""Reconstruct Fano-Q-constrained cavity spectra at the A46 switching states."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from tfln_paths import DATA_ROOT, OUTPUT_ROOT

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

C_MPS = 299_792_458.0
FANO_LOADED_Q = 50_741.0
CAVITY_ORDER = ("K1", "K2", "K3")
CAVITY_COLORS = {
    "K1": "#c44e52",
    "K2": "#4c72b0",
    "K3": "#55a868",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the three fitted cavity Lorentzians at each switching state."
    )
    parser.add_argument(
        "--parameters",
        type=Path,
        default=DATA_ROOT / "derived/model_parameters.json",
        help="JSON file containing the fitted cavity and switching-state parameters.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_ROOT / "figures/supplementary",
        help="Directory for the PDF, SVG, PNG, and summary CSV.",
    )
    return parser.parse_args()


def load_parameters(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        params = json.load(stream)
    if tuple(params["cavities"]) != CAVITY_ORDER:
        raise ValueError(
            f"Expected cavity order {CAVITY_ORDER}; got {tuple(params['cavities'])}"
        )
    return params


def cavity_model(cavity: dict, laser_nm: float, baseline_v: float) -> dict:
    """Return the voltage-independent TCMT quantities for one cavity."""
    laser_m = laser_nm * 1e-9
    eta_hz_v = float(cavity["tuning_efficiency_hz_per_v"])
    dlambda_dv_m = -(laser_m**2 / C_MPS) * eta_hz_v
    lambda_res_dc_m = laser_m + float(cavity["delta_at_baseline_pm"]) * 1e-12
    lambda_res_0_m = lambda_res_dc_m - dlambda_dv_m * baseline_v
    omega_res_0 = 2.0 * np.pi * C_MPS / lambda_res_0_m
    kappa = omega_res_0 / float(cavity["loaded_q"])
    coupling_ratio = float(cavity["external_to_internal_coupling"])
    kappa_e = kappa * coupling_ratio / (1.0 + coupling_ratio)
    return {
        "eta_hz_v": eta_hz_v,
        "omega_res_0": omega_res_0,
        "kappa": kappa,
        "kappa_e": kappa_e,
    }


def constrain_to_fano_q(params: dict) -> dict:
    """Apply the measured K1 Fano Q while preserving the voltage-cube response.

    The fixed-wavelength cube determines detuning and tuning rate only in
    units of the loaded linewidth. For each cavity, this transformation sets
    Q_L to the independently measured K1 value and rescales both the resonance
    offset and eta so that r(V) at the experimental laser is unchanged.
    """
    constrained = json.loads(json.dumps(params))
    laser_nm = float(params["laser_wavelength_nm"])
    laser_omega = 2.0 * np.pi * C_MPS / (laser_nm * 1e-9)
    for name in CAVITY_ORDER:
        baseline_v = float(params["baseline_voltage_v"][name])
        original_cavity = params["cavities"][name]
        original_model = cavity_model(original_cavity, laser_nm, baseline_v)
        original_q = float(original_cavity["loaded_q"])

        # Enforce kappa_new = scale*kappa_old together with
        # omega_res,new(V)-omega_laser = scale*[omega_res,old(V)-omega_laser].
        denominator = (
            FANO_LOADED_Q * original_model["omega_res_0"] / original_q
            - original_model["omega_res_0"]
            + laser_omega
        )
        scale = laser_omega / denominator
        omega_res_0 = laser_omega + scale * (
            original_model["omega_res_0"] - laser_omega
        )
        eta_hz_v = scale * original_model["eta_hz_v"]
        lambda_res_0_m = 2.0 * np.pi * C_MPS / omega_res_0
        dlambda_dv_m = -((laser_nm * 1e-9) ** 2) / C_MPS * eta_hz_v
        lambda_at_baseline_nm = (lambda_res_0_m + dlambda_dv_m * baseline_v) * 1e9

        cavity = constrained["cavities"][name]
        cavity["loaded_q"] = FANO_LOADED_Q
        cavity["tuning_efficiency_hz_per_v"] = eta_hz_v
        cavity["delta_at_baseline_pm"] = (lambda_at_baseline_nm - laser_nm) * 1e3

        new_model = cavity_model(cavity, laser_nm, baseline_v)
        voltages = [baseline_v] + [
            float(state["voltage_v"][name]) for state in params["switching_states"]
        ]
        for voltage in voltages:
            original_r = evaluate_reflection(
                original_model, voltage, np.asarray([laser_nm])
            )[0]
            new_r = evaluate_reflection(new_model, voltage, np.asarray([laser_nm]))[0]
            np.testing.assert_allclose(new_r, original_r, rtol=2e-6, atol=2e-8)

    constrained["fano_q_constraint"] = {
        "loaded_q": FANO_LOADED_Q,
        "source": "A46 K1 single-sided complex-Fano spectroscopy",
        "scope": "common-Q reconstruction for the three nominally identical switch cavities",
    }
    return constrained


def evaluate_cavity(
    model: dict, voltage_v: float, wavelength_nm: np.ndarray
) -> tuple[np.ndarray, float]:
    reflection, resonance_nm = evaluate_reflection(model, voltage_v, wavelength_nm)
    return np.abs(reflection) ** 2, resonance_nm


def evaluate_reflection(
    model: dict, voltage_v: float, wavelength_nm: np.ndarray
) -> tuple[np.ndarray, float]:
    omega_res = model["omega_res_0"] + 2.0 * np.pi * model["eta_hz_v"] * voltage_v
    omega = 2.0 * np.pi * C_MPS / (wavelength_nm * 1e-9)
    delta = omega_res - omega
    reflection = 1.0 - 2.0 * model["kappa_e"] / (2.0j * delta + model["kappa"])
    resonance_nm = 2.0 * np.pi * C_MPS / omega_res * 1e9
    return reflection, resonance_nm


def recombined_signals(
    params: dict, models: dict, state: dict, wavelength_nm: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Coherent fitted output powers and gain/offset calibrated detector signals."""
    config = params["recombination_model"]
    a_diag, a_far = config["mmi_amplitude_parameters"]
    w = np.exp(2j * np.pi / 3)
    phase = np.array([[1, 1, 1], [1, w, w.conjugate()], [1, w.conjugate(), w]])
    mag = np.array([[a_diag, 1, a_far], [1, a_diag, 1], [a_far, 1, a_diag]])
    mmi = mag * phase
    mmi /= np.sqrt(np.mean(np.abs(mmi) ** 2) * 3)
    reflected = np.stack(
        [
            evaluate_reflection(models[n], state["voltage_v"][n], wavelength_nm)[0]
            for n in config["arm_order"]
        ]
    )
    dv = np.array(
        [
            state["voltage_v"][n] - params["baseline_voltage_v"][n]
            for n in config["arm_order"]
        ]
    )
    arm_phase = (
        2 * np.array(config["one_way_arm_phase_rad"])
        + config["voltage_phase_slope_rad_per_v"] * dv
    )
    incident = mmi[:, config["input_port_index_zero_based"]]
    # Reciprocity maps the forward MMI transfer matrix to its transpose on
    # return; it does not complex-conjugate the propagation phases.
    field = mmi.T @ (reflected * (incident * np.exp(1j * arm_phase))[:, None])
    background = np.array(config["coherent_background_amplitude"]) * np.exp(
        1j * np.array(config["coherent_background_phase_rad"])
    )
    field += background[:, None]
    detector_power = np.abs(field[config["source_detector_port_order_zero_based"]]) ** 2
    calibration = np.array(config["detector_gain_offset"])
    detector_signal = detector_power * calibration[:, :1] + calibration[:, 1:]
    order = config["figure_channel_source_detector_indices"]
    return detector_power[order], detector_signal[order]


def linewidth_nm(model: dict, voltage_v: float) -> tuple[float, float, float]:
    """Exact wavelength endpoints and width of the frequency-domain FWHM."""
    omega_res = model["omega_res_0"] + 2.0 * np.pi * model["eta_hz_v"] * voltage_v
    omega_lo = omega_res - model["kappa"] / 2.0
    omega_hi = omega_res + model["kappa"] / 2.0
    wavelength_hi_nm = 2.0 * np.pi * C_MPS / omega_lo * 1e9
    wavelength_lo_nm = 2.0 * np.pi * C_MPS / omega_hi * 1e9
    return wavelength_lo_nm, wavelength_hi_nm, wavelength_hi_nm - wavelength_lo_nm


def transmission_at_laser(model: dict, voltage_v: float, laser_nm: float) -> float:
    transmission, _ = evaluate_cavity(model, voltage_v, np.asarray([laser_nm]))
    return float(transmission[0])


def build_rows(params: dict) -> list[dict]:
    laser_nm = float(params["laser_wavelength_nm"])
    models = {
        name: cavity_model(
            params["cavities"][name],
            laser_nm,
            float(params["baseline_voltage_v"][name]),
        )
        for name in CAVITY_ORDER
    }
    rows = []
    laser_frequency_hz = C_MPS / (laser_nm * 1e-9)
    for state in params["switching_states"]:
        for name in CAVITY_ORDER:
            voltage_v = float(state["voltage_v"][name])
            _, resonance_nm = evaluate_cavity(
                models[name], voltage_v, np.asarray([laser_nm])
            )
            _, _, width_nm = linewidth_nm(models[name], voltage_v)
            resonance_frequency_hz = C_MPS / (resonance_nm * 1e-9)
            cavity = params["cavities"][name]
            rows.append(
                {
                    "state": state["label"],
                    "source_state": state["source_state"],
                    "cavity": name,
                    "voltage_v": voltage_v,
                    "resonance_nm": resonance_nm,
                    "linewidth_pm": width_nm * 1e3,
                    "detuning_from_laser_pm": (resonance_nm - laser_nm) * 1e3,
                    "detuning_frequency_ghz": (
                        resonance_frequency_hz - laser_frequency_hz
                    )
                    / 1e9,
                    "transmission_at_laser": transmission_at_laser(
                        models[name], voltage_v, laser_nm
                    ),
                    "loaded_q": float(cavity["loaded_q"]),
                    "external_to_internal_coupling": float(
                        cavity["external_to_internal_coupling"]
                    ),
                    "tuning_efficiency_ghz_per_v": float(
                        cavity["tuning_efficiency_hz_per_v"]
                    )
                    / 1e9,
                }
            )
    return rows


def write_summary(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_states(params: dict, rows: list[dict], output_dir: Path) -> None:
    laser_nm = float(params["laser_wavelength_nm"])
    models = {
        name: cavity_model(
            params["cavities"][name],
            laser_nm,
            float(params["baseline_voltage_v"][name]),
        )
        for name in CAVITY_ORDER
    }

    voltage_v = np.linspace(-25.0, 25.0, 1800)

    mm = 1.0 / 25.4
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "legend.fontsize": 6.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.linewidth": 0.55,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "lines.linewidth": 1.15,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(3, 3, figsize=(178 * mm, 146 * mm), sharex="row")
    panel_labels = ("a", "b", "c", "d", "e", "f", "g", "h", "i")

    legend_handles = []
    for name in CAVITY_ORDER:
        voltage_linewidth = next(
            row["voltage_linewidth_v"] for row in rows if row["cavity"] == name
        )
        handle = mpl.lines.Line2D(
            [],
            [],
            color=CAVITY_COLORS[name],
            label=rf"$C_{{{name[1]}}}$  ($V_{{\rm linewidth}}={voltage_linewidth:.2f}$ V)",
        )
        legend_handles.append(handle)

    output_colors = ("#c44e52", "#55a868", "#4c72b0")
    for panel_idx, state in enumerate(params["switching_states"]):
        ax = axes[0, panel_idx]
        voltage_tuple = tuple(float(state["voltage_v"][name]) for name in CAVITY_ORDER)

        for name in CAVITY_ORDER:
            reflection = np.asarray(
                [
                    evaluate_reflection(models[name], float(v), np.asarray([laser_nm]))[
                        0
                    ][0]
                    for v in voltage_v
                ]
            )
            reflectance = np.abs(reflection) ** 2
            phase = -np.unwrap(np.angle(reflection))
            phase -= 0.5 * (phase[0] + phase[-1])
            axes[1, panel_idx].plot(
                voltage_v,
                phase / np.pi,
                color=CAVITY_COLORS[name],
            )
            color = CAVITY_COLORS[name]
            ax.plot(voltage_v, reflectance, color=color, zorder=3)

            operating_v = float(state["voltage_v"][name])
            operating_reflection = evaluate_reflection(
                models[name], operating_v, np.asarray([laser_nm])
            )[0][0]
            operating_phase = float(np.interp(operating_v, voltage_v, phase / np.pi))
            ax.scatter(
                [operating_v],
                [np.abs(operating_reflection) ** 2],
                s=15,
                color=color,
                edgecolor="white",
                linewidth=0.45,
                zorder=5,
            )
            axes[1, panel_idx].scatter(
                [operating_v],
                [operating_phase],
                s=15,
                color=color,
                edgecolor="white",
                linewidth=0.45,
                zorder=5,
            )

        ax.set_title(
            f"{state['label']}\n"
            + rf"$(V_1,V_2,V_3)=({voltage_tuple[0]:.2f},{voltage_tuple[1]:.2f},{voltage_tuple[2]:.2f})$ V",
            pad=5,
        )
        ax.set_xlim(voltage_v[0], voltage_v[-1])
        ax.set_ylim(0.0, 1.04)
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        powers, _ = recombined_signals(params, models, state, np.asarray([laser_nm]))
        axes[2, panel_idx].bar(
            np.arange(3),
            powers[:, 0],
            color=output_colors,
            width=0.62,
            edgecolor="white",
            linewidth=0.4,
        )
        axes[2, panel_idx].set_xticks(np.arange(3), [r"$P_1$", r"$P_2$", r"$P_3$"])
        axes[1, panel_idx].set_ylim(-1, 1)
        axes[1, panel_idx].set_yticks([-1, 0, 1])
        axes[1, panel_idx].set_xlabel(r"Applied cavity voltage $V_j$ (V)")

    output_ymax = max(ax.get_ylim()[1] for ax in axes[2])
    for row_idx in range(3):
        for col_idx in range(3):
            ax = axes[row_idx, col_idx]
            if row_idx == 2:
                ax.set_ylim(0, output_ymax)
            label_y = 1.24 if row_idx == 0 else 1.03
            ax.text(
                0.0,
                label_y,
                panel_labels[row_idx * 3 + col_idx],
                transform=ax.transAxes,
                fontweight="bold",
                va="bottom",
            )
            ax.grid(axis="y", color="#d9d9d9", linewidth=0.35)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if col_idx:
                ax.tick_params(labelleft=False)
    axes[0, 0].set_ylabel(r"Reflectance at the laser  $|r_j|^2$")
    axes[1, 0].set_ylabel(r"Reflection phase  $\phi_j/\pi$")
    axes[2, 0].set_ylabel("Output power at the laser\n(model units)")
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=3,
        frameon=False,
        handlelength=2.1,
        columnspacing=1.4,
    )
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.94), w_pad=1.0, h_pad=1.3)

    stem = output_dir / "cavity_switching_states"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.015)
    svg_path = stem.with_suffix(".svg")
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.015)
    svg_lines = svg_path.read_text(encoding="utf-8").splitlines()
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_lines) + "\n",
        encoding="utf-8",
    )
    fig.savefig(
        stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.015
    )
    plt.close(fig)


def plot_wavelength_states(params: dict, rows: list[dict], output_dir: Path) -> None:
    """Plot the common-Q wavelength reconstruction used in the Supplement."""
    laser_nm = float(params["laser_wavelength_nm"])
    models = {
        name: cavity_model(
            params["cavities"][name],
            laser_nm,
            float(params["baseline_voltage_v"][name]),
        )
        for name in CAVITY_ORDER
    }

    resonance_values = np.asarray([row["resonance_nm"] for row in rows])
    linewidth_values_nm = np.asarray([row["linewidth_pm"] for row in rows]) * 1e-3
    x_min = min(laser_nm, float(resonance_values.min())) - 1.25 * float(
        linewidth_values_nm.max()
    )
    x_max = max(laser_nm, float(resonance_values.max())) + 1.25 * float(
        linewidth_values_nm.max()
    )
    wavelength_nm = np.linspace(x_min, x_max, 1800)

    mm = 1.0 / 25.4
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "legend.fontsize": 6.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.linewidth": 0.55,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "lines.linewidth": 1.15,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(3, 3, figsize=(178 * mm, 146 * mm), sharex=True)
    panel_labels = ("a", "b", "c", "d", "e", "f", "g", "h", "i")

    first_state_rows = {
        row["cavity"]: row
        for row in rows
        if row["state"] == params["switching_states"][0]["label"]
    }
    legend_handles = []
    for name in CAVITY_ORDER:
        linewidth = first_state_rows[name]["linewidth_pm"]
        legend_handles.append(
            mpl.lines.Line2D(
                [],
                [],
                color=CAVITY_COLORS[name],
                label=rf"$C_{{{name[1]}}}$  ({linewidth:.1f} pm)",
            )
        )
    legend_handles.append(
        mpl.lines.Line2D(
            [],
            [],
            color="black",
            linestyle=(0, (3, 2)),
            linewidth=0.9,
            label=f"Laser  ({laser_nm:.3f} nm)",
        )
    )

    output_colors = ("#c44e52", "#55a868", "#4c72b0")
    for panel_idx, state in enumerate(params["switching_states"]):
        ax = axes[0, panel_idx]
        voltage_tuple = tuple(float(state["voltage_v"][name]) for name in CAVITY_ORDER)
        ax.axvline(
            laser_nm, color="black", linestyle=(0, (3, 2)), linewidth=0.9, zorder=1
        )

        for name in CAVITY_ORDER:
            voltage_v = float(state["voltage_v"][name])
            reflectance, resonance_nm = evaluate_cavity(
                models[name], voltage_v, wavelength_nm
            )
            reflection, _ = evaluate_reflection(models[name], voltage_v, wavelength_nm)
            phase = np.unwrap(np.angle(reflection))
            if phase[0] < 0:
                phase += 2 * np.pi
            centered_phase = phase - np.pi
            axes[1, panel_idx].plot(
                wavelength_nm, centered_phase / np.pi, color=CAVITY_COLORS[name]
            )
            wl_lo, wl_hi, _ = linewidth_nm(models[name], voltage_v)
            color = CAVITY_COLORS[name]
            ax.plot(wavelength_nm, reflectance, color=color, zorder=3)

            center_r = float(
                evaluate_cavity(models[name], voltage_v, np.asarray([resonance_nm]))[0][
                    0
                ]
            )
            half_depth_r = 0.5 * (1.0 + center_r)
            ax.plot(
                [wl_lo, wl_hi],
                [half_depth_r, half_depth_r],
                color=color,
                linewidth=1.7,
                marker="|",
                markersize=4,
                markeredgewidth=0.8,
                zorder=4,
            )
            ax.scatter(
                [resonance_nm],
                [center_r],
                s=9,
                color=color,
                edgecolor="white",
                linewidth=0.35,
                zorder=5,
            )
            ax.scatter(
                [resonance_nm],
                [1.018],
                marker="v",
                s=8,
                color=color,
                linewidth=0,
                clip_on=False,
                zorder=5,
            )

        ax.set_title(
            f"{state['label']}\n"
            + rf"$(V_1,V_2,V_3)=({voltage_tuple[0]:.2f},{voltage_tuple[1]:.2f},{voltage_tuple[2]:.2f})$ V",
            pad=5,
        )
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0.0, 1.04)
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.05))
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        powers, _ = recombined_signals(params, models, state, wavelength_nm)
        for output_idx, color in enumerate(output_colors):
            axes[2, panel_idx].plot(
                wavelength_nm,
                powers[output_idx],
                color=color,
                label=rf"$P_{{{output_idx + 1}}}$",
            )
        axes[2, panel_idx].plot(
            wavelength_nm,
            powers.sum(axis=0),
            color="black",
            linestyle=(0, (4, 2)),
            label=r"$\sum_m P_m$",
        )
        axes[1, panel_idx].set_ylim(-1, 1)
        axes[1, panel_idx].set_yticks([-1, 0, 1])
        axes[2, panel_idx].set_xlabel("Wavelength (nm)")

    output_ymax = max(ax.get_ylim()[1] for ax in axes[2])
    for row_idx in range(3):
        for col_idx in range(3):
            ax = axes[row_idx, col_idx]
            if row_idx > 0:
                ax.axvline(
                    laser_nm,
                    color="black",
                    linestyle=(0, (3, 2)),
                    linewidth=0.9,
                    zorder=1,
                )
            if row_idx == 2:
                ax.set_ylim(0, output_ymax)
            label_y = 1.24 if row_idx == 0 else 1.03
            ax.text(
                0.0,
                label_y,
                panel_labels[row_idx * 3 + col_idx],
                transform=ax.transAxes,
                fontweight="bold",
                va="bottom",
            )
            ax.ticklabel_format(axis="x", style="plain", useOffset=False)
            ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.05))
            ax.grid(axis="y", color="#d9d9d9", linewidth=0.35)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if col_idx:
                ax.tick_params(labelleft=False)
    axes[0, 0].set_ylabel(r"Cavity reflectance  $|r_j|^2$")
    axes[1, 0].set_ylabel(r"Reflection phase  $\phi_j/\pi$")
    axes[2, 0].set_ylabel("Recombined power (model units)")
    axes[2, 1].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.30),
        ncol=4,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.0,
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=4,
        frameon=False,
        handlelength=2.1,
        columnspacing=1.4,
    )
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.94), w_pad=1.0, h_pad=1.3)

    stem = output_dir / "cavity_switching_states"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.015)
    svg_path = stem.with_suffix(".svg")
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.015)
    svg_lines = svg_path.read_text(encoding="utf-8").splitlines()
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_lines) + "\n", encoding="utf-8"
    )
    fig.savefig(
        stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.015
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    params = constrain_to_fano_q(load_parameters(args.parameters))
    rows = build_rows(params)
    write_summary(rows, args.output_dir / "cavity_state_summary.csv")
    plot_wavelength_states(params, rows, args.output_dir)

    print(f"Laser wavelength: {params['laser_wavelength_nm']:.5f} nm")
    for row in rows:
        print(
            f"{row['state']:11s}  {row['cavity']}  V={row['voltage_v']:+8.3f} V  "
            f"lambda={row['resonance_nm']:.6f} nm  FWHM={row['linewidth_pm']:.2f} pm  "
            f"R(laser)={row['transmission_at_laser']:.3f}"
        )


if __name__ == "__main__":
    main()
