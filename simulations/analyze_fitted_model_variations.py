#!/usr/bin/env python3
"""Explore reachable states and robustness of the calibrated three-port model.

This is deliberately a sensitivity study, not an uncertainty propagation: the
saved fit contains point estimates but no parameter covariance.  The optical
model uses the physically reciprocal return convention M.T and omits detector
calibration and empirical coherent-background terms.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from tfln_paths import DATA_ROOT, OUTPUT_ROOT

import csv
import json
from dataclasses import dataclass, replace

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

C_MPS = 299_792_458.0
PARAMETERS = DATA_ROOT / "derived/model_parameters.json"
OUTPUT_DIR = OUTPUT_ROOT / "simulations"
OUTPUT_PNG = OUTPUT_DIR / "fitted_model_reachable_sensitivity.png"
OUTPUT_SVG = OUTPUT_DIR / "fitted_model_reachable_sensitivity.svg"
OUTPUT_PDF = OUTPUT_DIR / "fitted_model_reachable_sensitivity.pdf"
OUTPUT_CSV = OUTPUT_DIR / "fitted_model_reachable_sensitivity.csv"

BLUE = "#3378A8"
ORANGE = "#D97722"
GREEN = "#4C956C"
GRAY = "#777777"
LIGHT = "#DCEAF3"

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.7,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


@dataclass(frozen=True)
class OpticalParameters:
    """Arrays use the fitted model's arm order (K3, K2, K1)."""

    laser_omega: float
    baseline_v: np.ndarray
    omega_res_0: np.ndarray
    eta_hz_v: np.ndarray
    kappa_e: np.ndarray
    kappa_i: np.ndarray
    coupling_ratio: np.ndarray
    loaded_q: np.ndarray
    detuning_pm: np.ndarray


def load_model() -> tuple[dict, OpticalParameters]:
    params = json.loads(PARAMETERS.read_text(encoding="utf-8"))
    config = params["recombination_model"]
    arm_order = config["arm_order"]
    laser_m = float(params["laser_wavelength_nm"]) * 1e-9
    laser_omega = 2 * np.pi * C_MPS / laser_m

    baseline = []
    omega_0 = []
    eta = []
    kappa_e = []
    kappa_i = []
    coupling = []
    loaded_q = []
    detuning_pm = []
    for name in arm_order:
        cavity = params["cavities"][name]
        base = float(params["baseline_voltage_v"][name])
        eta_j = float(cavity["tuning_efficiency_hz_per_v"])
        detuning_j = float(cavity["delta_at_baseline_pm"])
        dlambda_dv = -(laser_m**2 / C_MPS) * eta_j
        lambda_res_dc = laser_m + detuning_j * 1e-12
        lambda_res_0 = lambda_res_dc - dlambda_dv * base
        omega_j = 2 * np.pi * C_MPS / lambda_res_0
        q_j = float(cavity["loaded_q"])
        x_j = float(cavity["external_to_internal_coupling"])
        kappa = omega_j / q_j

        baseline.append(base)
        omega_0.append(omega_j)
        eta.append(eta_j)
        kappa_e.append(kappa * x_j / (1 + x_j))
        kappa_i.append(kappa / (1 + x_j))
        coupling.append(x_j)
        loaded_q.append(q_j)
        detuning_pm.append(detuning_j)

    optical = OpticalParameters(
        laser_omega=laser_omega,
        baseline_v=np.asarray(baseline),
        omega_res_0=np.asarray(omega_0),
        eta_hz_v=np.asarray(eta),
        kappa_e=np.asarray(kappa_e),
        kappa_i=np.asarray(kappa_i),
        coupling_ratio=np.asarray(coupling),
        loaded_q=np.asarray(loaded_q),
        detuning_pm=np.asarray(detuning_pm),
    )
    return params, optical


def rebuild_cavities(
    nominal: OpticalParameters,
    q_scale: np.ndarray,
    coupling_scale: np.ndarray,
    eta_scale: np.ndarray,
    detuning_shift_pm: np.ndarray,
) -> OpticalParameters:
    """Perturb fitted cavity quantities and reconstruct TCMT rates consistently."""
    laser_m = 2 * np.pi * C_MPS / nominal.laser_omega
    q = nominal.loaded_q * q_scale
    coupling = nominal.coupling_ratio * coupling_scale
    eta = nominal.eta_hz_v * eta_scale
    detuning = nominal.detuning_pm + detuning_shift_pm
    dlambda_dv = -(laser_m**2 / C_MPS) * eta
    lambda_res_dc = laser_m + detuning * 1e-12
    lambda_res_0 = lambda_res_dc - dlambda_dv * nominal.baseline_v
    omega_0 = 2 * np.pi * C_MPS / lambda_res_0
    kappa = omega_0 / q
    return replace(
        nominal,
        omega_res_0=omega_0,
        eta_hz_v=eta,
        kappa_e=kappa * coupling / (1 + coupling),
        kappa_i=kappa / (1 + coupling),
        coupling_ratio=coupling,
        loaded_q=q,
        detuning_pm=detuning,
    )


def voltage_grid(
    baseline_v: np.ndarray, half_span_v: float, samples: int
) -> np.ndarray:
    relative = np.linspace(-half_span_v, half_span_v, samples)
    v0, v1, v2 = np.meshgrid(relative, relative, relative, indexing="ij")
    return baseline_v[None, :] + np.column_stack((v0.ravel(), v1.ravel(), v2.ravel()))


def mmi_matrix(params: dict, imbalance_scale: float) -> np.ndarray:
    config = params["recombination_model"]
    a_diag, a_far = config["mmi_amplitude_parameters"]
    fitted_mag = np.array([[a_diag, 1, a_far], [1, a_diag, 1], [a_far, 1, a_diag]])
    mag = 1 + imbalance_scale * (fitted_mag - 1)
    w = np.exp(2j * np.pi / 3)
    phase = np.array([[1, 1, 1], [1, w, w.conjugate()], [1, w.conjugate(), w]])
    mmi = mag * phase
    return mmi / np.sqrt(np.mean(np.abs(mmi) ** 2) * 3)


def reflected_amplitudes(optical: OpticalParameters, voltage: np.ndarray) -> np.ndarray:
    omega_res = (
        optical.omega_res_0[None, :] + 2 * np.pi * voltage * optical.eta_hz_v[None, :]
    )
    delta = omega_res - optical.laser_omega
    return 1 - 2 * optical.kappa_e[None, :] / (
        2j * delta + optical.kappa_e[None, :] + optical.kappa_i[None, :]
    )


def output_powers(
    params: dict,
    optical: OpticalParameters,
    voltage: np.ndarray,
    *,
    imbalance_scale: float = 1.0,
    roundtrip_phase_offset_rad: np.ndarray | None = None,
) -> np.ndarray:
    config = params["recombination_model"]
    mmi = mmi_matrix(params, imbalance_scale)
    reflection = reflected_amplitudes(optical, voltage)
    dv = voltage - optical.baseline_v[None, :]
    arm_phase = (
        2 * np.asarray(config["one_way_arm_phase_rad"])[None, :]
        + float(config["voltage_phase_slope_rad_per_v"]) * dv
    )
    if roundtrip_phase_offset_rad is not None:
        arm_phase = (
            arm_phase + np.asarray(roundtrip_phase_offset_rad, dtype=float)[None, :]
        )
    incident = mmi[:, int(config["input_port_index_zero_based"])]
    internal = reflection * incident[None, :] * np.exp(1j * arm_phase)
    field = internal @ mmi

    # Preserve the red/green/blue manuscript channel order.
    detector_order = np.asarray(config["source_detector_port_order_zero_based"])
    figure_order = np.asarray(config["figure_channel_source_detector_indices"])
    field = field[:, detector_order][:, figure_order]
    return np.abs(field) ** 2


def normalize_power(power: np.ndarray) -> np.ndarray:
    return power / np.maximum(power.sum(axis=1, keepdims=True), np.finfo(float).tiny)


def route_metrics(power: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return best ER, target fraction at best ER, and selected row per output."""
    normalized = normalize_power(power)
    best_er = np.empty(3)
    target_at_best = np.empty(3)
    selected = np.empty(3, dtype=int)
    for target in range(3):
        competitors = np.delete(power, target, axis=1).max(axis=1)
        er = 10 * np.log10(
            np.maximum(power[:, target], np.finfo(float).tiny)
            / np.maximum(competitors, np.finfo(float).tiny)
        )
        selected[target] = int(np.argmax(er))
        best_er[target] = er[selected[target]]
        target_at_best[target] = normalized[selected[target], target]
    return best_er, target_at_best, selected


def refined_route_metrics(
    params: dict,
    optical: OpticalParameters,
    voltage: np.ndarray,
    **model_options: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Refine the best grid points with bounded continuous optimization."""
    grid_power = output_powers(params, optical, voltage, **model_options)
    grid_er, grid_target, selected = route_metrics(grid_power)
    bounds = [
        (float(voltage[:, j].min()), float(voltage[:, j].max())) for j in range(3)
    ]
    best_er = np.empty(3)
    target_at_best = np.empty(3)
    best_voltage = np.empty((3, 3))

    for target in range(3):

        def objective(v: np.ndarray) -> float:
            power = output_powers(
                params, optical, np.asarray(v, dtype=float)[None, :], **model_options
            )[0]
            competitor = np.delete(power, target).max()
            return -10 * np.log10(
                max(power[target], np.finfo(float).tiny)
                / max(competitor, np.finfo(float).tiny)
            )

        result = minimize(
            objective,
            voltage[selected[target]],
            method="Powell",
            bounds=bounds,
            options={"xtol": 1e-6, "ftol": 1e-9, "maxiter": 350},
        )
        candidate_voltage = result.x
        power = output_powers(
            params, optical, candidate_voltage[None, :], **model_options
        )[0]
        competitor = np.delete(power, target).max()
        candidate_er = 10 * np.log10(
            max(power[target], np.finfo(float).tiny)
            / max(competitor, np.finfo(float).tiny)
        )
        if candidate_er >= grid_er[target]:
            best_voltage[target] = candidate_voltage
            best_er[target] = candidate_er
            target_at_best[target] = power[target] / power.sum()
        else:
            # Powell can occasionally leave a good nonsmooth max/min-ratio
            # grid point for a worse local basin; never accept that regression.
            best_voltage[target] = voltage[selected[target]]
            best_er[target] = grid_er[target]
            target_at_best[target] = grid_target[target]
    return best_er, target_at_best, best_voltage


def barycentric_xy(power: np.ndarray) -> np.ndarray:
    return np.column_stack(
        (power[:, 1] + 0.5 * power[:, 2], np.sqrt(3) * power[:, 2] / 2)
    )


def ideal_boundary(samples: int = 1601) -> np.ndarray:
    phase = np.linspace(0, 2 * np.pi, samples)
    powers = (
        np.column_stack(
            (
                (1 + 2 * np.cos(phase)) ** 2,
                (1 + 2 * np.cos(phase - 2 * np.pi / 3)) ** 2,
                (1 + 2 * np.cos(phase + 2 * np.pi / 3)) ** 2,
            )
        )
        / 9
    )
    return barycentric_xy(powers)


def worst_route_metrics(
    params: dict,
    optical: OpticalParameters,
    voltage: np.ndarray,
    **model_options: object,
) -> tuple[float, float]:
    er, target, _ = refined_route_metrics(params, optical, voltage, **model_options)
    return float(np.min(er)), float(np.min(target))


def random_variation_study(
    params: dict,
    nominal: OpticalParameters,
    *,
    samples_per_level: int = 80,
    grid_samples: int = 13,
) -> dict[float, np.ndarray]:
    """Sensitivity ensemble; levels are assumed spreads, not fitted errors."""
    rng = np.random.default_rng(17)
    voltage = voltage_grid(nominal.baseline_v, 5.0, grid_samples)
    results: dict[float, np.ndarray] = {}
    for level in (0.0, 0.05, 0.10, 0.20):
        values = []
        count = 1 if level == 0 else samples_per_level
        for _ in range(count):
            if level == 0:
                varied = nominal
            else:
                # Positive quantities use log-normal perturbations.  Resonance
                # jitter is 50 pm times the stated fractional sensitivity level,
                # comparable to the fitted ~44 pm loaded linewidths at 100%.
                varied = rebuild_cavities(
                    nominal,
                    q_scale=np.exp(level * rng.normal(size=3)),
                    coupling_scale=np.exp(level * rng.normal(size=3)),
                    eta_scale=np.exp(level * rng.normal(size=3)),
                    detuning_shift_pm=50 * level * rng.normal(size=3),
                )
            values.append(
                worst_route_metrics(
                    params,
                    varied,
                    voltage,
                    imbalance_scale=1.0,
                )[0]
            )
        results[level] = np.asarray(values)
    return results


def write_summary(rows: list[dict[str, object]]) -> None:
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    params, nominal = load_model()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fine_voltage = voltage_grid(nominal.baseline_v, 5.0, 51)
    nominal_power = output_powers(
        params,
        nominal,
        fine_voltage,
        imbalance_scale=1.0,
    )
    nominal_normalized = normalize_power(nominal_power)
    nominal_er, nominal_target, nominal_voltage = refined_route_metrics(
        params,
        nominal,
        fine_voltage,
        imbalance_scale=1.0,
    )

    variants = [
        ("Fitted cavities\nideal MMI", 0.0),
        ("Fitted cavities\n+ fitted MMI", 1.0),
    ]
    variant_er = []
    variant_target = []
    summary_rows: list[dict[str, object]] = []
    for label, imbalance in variants:
        er, target, _ = refined_route_metrics(
            params,
            nominal,
            fine_voltage,
            imbalance_scale=imbalance,
        )
        variant_er.append(er)
        variant_target.append(target)
        for port in range(3):
            summary_rows.append(
                {
                    "analysis": "nominal_variant",
                    "case": label.replace("\n", " "),
                    "port": port + 1,
                    "worst_or_port_er_db": er[port],
                    "target_fraction": target[port],
                }
            )
    variant_er = np.asarray(variant_er)
    variant_target = np.asarray(variant_target)

    sweep_voltage = voltage_grid(nominal.baseline_v, 5.0, 17)
    loss_scales = np.linspace(0.0, 2.0, 17)
    loss_er = []
    loss_target = []
    for scale in loss_scales:
        changed = replace(nominal, kappa_i=nominal.kappa_i * scale)
        er, target = worst_route_metrics(
            params,
            changed,
            sweep_voltage,
            imbalance_scale=1.0,
        )
        loss_er.append(er)
        loss_target.append(target)

    voltage_spans = np.linspace(1.0, 8.0, 15)
    span_er = []
    span_target = []
    for span in voltage_spans:
        voltage = voltage_grid(nominal.baseline_v, float(span), 17)
        er, target = worst_route_metrics(
            params,
            nominal,
            voltage,
            imbalance_scale=1.0,
        )
        span_er.append(er)
        span_target.append(target)

    # Voltage boxes are nested, so their true optimum cannot decrease with
    # span.  Use the numerical upper envelope to suppress small optimizer/grid
    # fluctuations when an already-feasible narrow-span optimum is missed.
    span_er = np.maximum.accumulate(np.asarray(span_er))

    random_results = random_variation_study(params, nominal)
    for scale, er, target in zip(loss_scales, loss_er, loss_target):
        summary_rows.append(
            {
                "analysis": "intrinsic_loss_sweep",
                "case": f"kappa_i_multiplier={scale:.3f}",
                "port": "worst of three",
                "worst_or_port_er_db": er,
                "target_fraction": target,
            }
        )
    for span, er, target in zip(voltage_spans, span_er, span_target):
        summary_rows.append(
            {
                "analysis": "voltage_span_sweep",
                "case": f"half_span_v={span:.3f}",
                "port": "worst of three",
                "worst_or_port_er_db": er,
                "target_fraction": target,
            }
        )
    for level, values in random_results.items():
        summary_rows.append(
            {
                "analysis": "random_sensitivity",
                "case": f"assumed_fractional_spread={level:.2f}",
                "port": "worst of three",
                "worst_or_port_er_db": float(np.median(values)),
                "target_fraction": "",
            }
        )
    write_summary(summary_rows)

    fig, axes = plt.subplots(2, 3, figsize=(10.4, 6.35), constrained_layout=False)
    fig.subplots_adjust(
        left=0.055, right=0.985, bottom=0.10, top=0.90, wspace=0.34, hspace=0.47
    )
    fig.suptitle(
        "Calibrated $3\\times3$ router: reachable states and parameter sensitivity",
        fontsize=11,
        y=0.975,
    )

    # a) Nominal reachable region.
    ax = axes[0, 0]
    boundary = ideal_boundary()
    ax.fill(
        boundary[:, 0],
        boundary[:, 1],
        color="#E7E7E7",
        label="ideal phase-only region",
        zorder=0,
    )
    ax.plot(boundary[:, 0], boundary[:, 1], color=GRAY, lw=0.9, ls="--", zorder=1)
    xy = barycentric_xy(nominal_normalized)
    ax.hexbin(
        xy[:, 0],
        xy[:, 1],
        gridsize=135,
        extent=(-0.01, 1.01, -0.01, 0.88),
        mincnt=1,
        cmap=mpl.colors.ListedColormap([BLUE]),
        alpha=0.62,
        linewidths=0,
        rasterized=True,
        zorder=2,
    )
    best_power = output_powers(
        params,
        nominal,
        nominal_voltage,
        imbalance_scale=1.0,
    )
    best_xy = barycentric_xy(normalize_power(best_power))
    ax.scatter(
        best_xy[:, 0],
        best_xy[:, 1],
        color=ORANGE,
        edgecolor="white",
        lw=0.6,
        s=30,
        zorder=4,
    )
    simplex = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2], [0, 0]])
    ax.plot(simplex[:, 0], simplex[:, 1], color="#444444", lw=1.0, zorder=3)
    ax.text(-0.025, -0.035, "$P_1$", ha="center", va="top")
    ax.text(1.025, -0.035, "$P_2$", ha="center", va="top")
    ax.text(0.5, np.sqrt(3) / 2 + 0.025, "$P_3$", ha="center", va="bottom")
    ax.text(
        0.5,
        0.46,
        "blue: fitted cavities + MMI\norange: optimized routes",
        ha="center",
        va="center",
        fontsize=6.7,
    )
    ax.set_title("a) Reachable region in the actual $\\pm5$ V cube", loc="left")
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.08, 0.94)
    ax.set_aspect("equal")
    ax.axis("off")

    # b) Model-component ablation.
    ax = axes[0, 1]
    x = np.arange(len(variants))
    colors = [GRAY, BLUE]
    bars = ax.bar(x, np.min(variant_er, axis=1), color=colors, alpha=0.88, width=0.72)
    for bar, er_value, target_value in zip(
        bars, np.min(variant_er, axis=1), np.min(variant_target, axis=1)
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            er_value + 0.7,
            f"{er_value:.1f} dB\n{100 * target_value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=6.5,
        )
    ax.set_xticks(x, [item[0] for item in variants])
    ax.set_ylabel("Worst of the three optimized routes (dB)")
    ax.set_title("b) What limits the fitted model?", loc="left")
    ax.set_ylim(0, max(variant_er.min(axis=1)) + 8)
    ax.grid(axis="y", color="#DDDDDD", lw=0.5)
    ax.text(
        0.02,
        0.98,
        "labels: ER and target fraction",
        transform=ax.transAxes,
        va="top",
        fontsize=6.5,
        color="#555555",
    )

    # c) Intrinsic-loss sensitivity.
    ax = axes[0, 2]
    ax.plot(loss_scales, loss_er, color=BLUE, lw=1.8)
    ax.axvline(1, color=ORANGE, ls="--", lw=1.0, label="fitted value")
    ax.set_xlabel("Intrinsic-loss multiplier $\\kappa_i/\\kappa_{i,\\mathrm{fit}}$")
    ax.set_ylabel("Worst-route optimized ER (dB)")
    ax.set_title("c) Cavity absorption", loc="left")
    ax.grid(color="#DDDDDD", lw=0.5)
    ax.legend(frameon=False, loc="upper right")

    # d) Available-control sensitivity.
    ax = axes[1, 0]
    ax.plot(voltage_spans, span_er, color=BLUE, lw=1.8)
    ax.axvline(5, color=ORANGE, ls="--", lw=1.0, label="measured cube")
    ax.set_xlabel("Independent voltage half-span per cavity (V)")
    ax.set_ylabel("Worst-route optimized ER (dB)")
    ax.set_title("d) Available control range", loc="left")
    ax.grid(color="#DDDDDD", lw=0.5)
    ax.legend(frameon=False, loc="lower right")

    # e) Randomized cavity sensitivity.
    ax = axes[1, 1]
    levels = list(random_results)
    data = [random_results[level] for level in levels]
    positions = np.arange(len(levels))
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "white", "lw": 1.2},
        whiskerprops={"color": BLUE},
        capprops={"color": BLUE},
        boxprops={"facecolor": BLUE, "edgecolor": BLUE, "alpha": 0.78},
    )
    del bp
    ax.set_xticks(positions, [f"{100 * level:.0f}%" for level in levels])
    ax.set_xlabel("Assumed independent parameter spread")
    ax.set_ylabel("Worst-route optimized ER (dB)")
    ax.set_title("e) Joint cavity-parameter sensitivity", loc="left")
    ax.grid(axis="y", color="#DDDDDD", lw=0.5)
    ax.text(
        0.02,
        0.03,
        "$Q$, $\\kappa_e/\\kappa_i$, $\\eta$: log-normal\n"
        "resonance jitter: 2.5/5/10 pm",
        transform=ax.transAxes,
        fontsize=6.4,
        va="bottom",
        color="#555555",
    )

    # f) Compact interpretation panel.
    ax = axes[1, 2]
    ax.axis("off")
    fitted_index = int(np.argmin(np.abs(loss_scales - 1)))
    span_index = int(np.argmin(np.abs(voltage_spans - 5)))
    random_10 = random_results[0.10]
    text = (
        "Nominal reciprocal optical model\n"
        f"  port ERs: {nominal_er[0]:.1f}, {nominal_er[1]:.1f}, {nominal_er[2]:.1f} dB\n"
        f"  target fractions: {100*nominal_target[0]:.1f}, {100*nominal_target[1]:.1f}, "
        f"{100*nominal_target[2]:.1f}%\n\n"
        "Sensitivity at the fitted point\n"
        f"  worst route: {loss_er[fitted_index]:.1f} dB\n"
        f"  actual $\\pm5$ V span: {span_er[span_index]:.1f} dB\n"
        f"  assumed 10% spread median: {np.median(random_10):.1f} dB\n"
        f"  assumed 10% spread 5th percentile: {np.percentile(random_10, 5):.1f} dB\n\n"
        "Interpretation\n"
        "  finite phase range and unequal cavity loss\n"
        "  deform the region and remove exact vertices.\n"
        "  Lower loss helps only when amplitude balance\n"
        "  is improved at the same time."
    )
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=7.4,
        linespacing=1.35,
        bbox={
            "boxstyle": "round,pad=0.55",
            "facecolor": "#F4F4F4",
            "edgecolor": "#BBBBBB",
        },
    )
    ax.set_title("f) Main readout", loc="left")

    fig.text(
        0.5,
        0.025,
        "Sensitivity ensemble is illustrative, not a confidence interval; the saved fit has no covariance. "
        "Optical panels omit detector gains and offsets.",
        ha="center",
        fontsize=7,
        color="#555555",
    )
    fig.savefig(OUTPUT_PNG, dpi=300)
    fig.savefig(OUTPUT_SVG)
    fig.savefig(OUTPUT_PDF)
    plt.close(fig)

    print(f"wrote {OUTPUT_PNG}")
    print(f"wrote {OUTPUT_CSV}")
    print("nominal ER (dB):", nominal_er)
    print("nominal target fractions:", nominal_target)
    print("variant worst ER (dB):", np.min(variant_er, axis=1))
    print("variant worst target fractions:", np.min(variant_target, axis=1))
    for level, values in random_results.items():
        print(
            f"spread={100*level:.0f}%: worst-route ER median={np.median(values):.2f} dB, "
            f"p05={np.percentile(values, 5):.2f}, p95={np.percentile(values, 95):.2f}"
        )


if __name__ == "__main__":
    main()
