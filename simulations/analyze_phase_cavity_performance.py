#!/usr/bin/env python3
"""Monte Carlo study of arm phase, cavity Q, and overcoupling.

The study uses the reciprocal fitted three-port optical model.  Two independent
round-trip arm phases are sampled uniformly; the first arm is the phase
reference.  For every phase realization and cavity design, all three cavity
voltages are swept independently around their fitted baseline values.

The principal observables deliberately describe regions rather than a single
optimized point:

* the worst of the three independently optimized route extinction ratios;
* the smallest of the three voltage-cube fractions above an ER threshold; and
* the largest pairwise minimum L-infinity voltage distance between the three
  ER regions (a lower bound on the driver step between routing regions).

The Q sweep holds kappa_e/kappa_i and the Pockels tuning rate in Hz/V fixed.
The coupling sweep holds loaded Q fixed and sets all three cavities to the
same kappa_e/kappa_i.  These controlled sweeps separate linewidth narrowing
from reduced dissipative loss; they are not fabrication-yield predictions.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from tfln_paths import OUTPUT_ROOT, data_path

import csv
import sys

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import cKDTree

from analyze_fitted_model_variations import (
    OUTPUT_DIR,
    barycentric_xy,
    load_model,
    normalize_power,
    output_powers,
    reflected_amplitudes,
    refined_route_metrics,
    rebuild_cavities,
    mmi_matrix,
    voltage_grid,
)

OUTPUT_PDF = OUTPUT_DIR / "phase_cavity_performance.pdf"
OUTPUT_SVG = OUTPUT_DIR / "phase_cavity_performance.svg"
OUTPUT_PNG = OUTPUT_DIR / "phase_cavity_performance.png"
OUTPUT_REGIONS_PDF = OUTPUT_DIR / "phase_cavity_reachable_regions.pdf"
OUTPUT_REGIONS_SVG = OUTPUT_DIR / "phase_cavity_reachable_regions.svg"
OUTPUT_CSV = OUTPUT_DIR / "phase_cavity_performance.csv"
SUPPLEMENTARY_FIGURE_DIR = OUTPUT_ROOT / "figures/supplementary"
SUPPLEMENTARY_PHASE_PDF = SUPPLEMENTARY_FIGURE_DIR / "FIG_Arm_Phase_Robustness.pdf"
SUPPLEMENTARY_PHASE_SVG = SUPPLEMENTARY_FIGURE_DIR / "FIG_Arm_Phase_Robustness.svg"
SUPPLEMENTARY_CAVITY_PDF = SUPPLEMENTARY_FIGURE_DIR / "FIG_Cavity_Design_Tradeoffs.pdf"
SUPPLEMENTARY_CAVITY_SVG = SUPPLEMENTARY_FIGURE_DIR / "FIG_Cavity_Design_Tradeoffs.svg"

BLUE = "#3378A8"
ORANGE = "#D97722"
GREEN = "#4C956C"
RED = "#B7473A"
PURPLE = "#7956A5"
GRAY = "#777777"


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
        "legend.fontsize": 6.8,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def load_tolerance_reference() -> dict:
    """Load the published rounded reference point; this script does not refit it."""
    import json

    path = data_path("simulations/phase_tolerance_reference.json")
    reference = json.loads(path.read_text())
    for name in ("region_fraction", "worst_best_er_db", "half_width_v"):
        if name not in reference or not np.isfinite(reference[name]):
            raise ValueError(f"Invalid tolerance reference field: {name}")
    if not 0 <= reference["region_fraction"] <= 1 or reference["half_width_v"] < 0:
        raise ValueError("Invalid tolerance reference range.")
    return reference


def er_values(power: np.ndarray) -> np.ndarray:
    """ER in dB for every voltage point (rows) and target output (columns)."""
    tiny = np.finfo(float).tiny
    result = np.empty_like(power)
    for target in range(3):
        competitor = np.delete(power, target, axis=1).max(axis=1)
        result[:, target] = 10 * np.log10(
            np.maximum(power[:, target], tiny) / np.maximum(competitor, tiny)
        )
    return result


def region_metrics(
    voltage: np.ndarray,
    er: np.ndarray,
    threshold_db: float,
) -> tuple[float, float]:
    """Return worst-route region fraction and pairwise region separation.

    Separation is the largest of the three minimum L-infinity distances
    between pairs of thresholded route regions.  It is a lower bound, rather
    than a jointly optimized three-state drive excursion.
    """
    masks = [er[:, target] >= threshold_db for target in range(3)]
    fraction = float(min(np.mean(mask) for mask in masks))
    if any(not np.any(mask) for mask in masks):
        return fraction, np.nan
    separations = []
    for first, second in ((0, 1), (0, 2), (1, 2)):
        points_a = voltage[masks[first]]
        points_b = voltage[masks[second]]
        # Query the smaller set to reduce memory and runtime.
        if len(points_a) > len(points_b):
            points_a, points_b = points_b, points_a
        distances, _ = cKDTree(points_b).query(points_a, k=1, p=np.inf)
        separations.append(float(np.min(distances)))
    return fraction, float(max(separations))


def evaluate(
    params: dict,
    optical,
    voltage: np.ndarray,
    phase_rad: np.ndarray,
    threshold_db: float = 10.0,
) -> dict[str, float | np.ndarray]:
    power = output_powers(
        params,
        optical,
        voltage,
        imbalance_scale=1.0,
        roundtrip_phase_offset_rad=phase_rad,
    )
    er = er_values(power)
    best_by_route = np.max(er, axis=0)
    # Refine each grid maximum on successively smaller local Cartesian grids.
    # This prevents narrow high-Q extrema from being mistaken for degraded
    # performance merely because the publication-region grid is coarse.
    lower = voltage.min(axis=0)
    upper = voltage.max(axis=0)
    grid_steps = []
    for arm in range(3):
        unique = np.unique(voltage[:, arm])
        grid_steps.append(float(np.min(np.diff(unique))))
    for target in range(3):
        center = voltage[int(np.argmax(er[:, target]))].copy()
        half_width = max(grid_steps)
        for _ in range(3):
            offsets = np.linspace(-half_width, half_width, 7)
            a, b, c = np.meshgrid(offsets, offsets, offsets, indexing="ij")
            local = center[None, :] + np.column_stack((a.ravel(), b.ravel(), c.ravel()))
            local = np.clip(local, lower, upper)
            local_power = output_powers(
                params,
                optical,
                local,
                imbalance_scale=1.0,
                roundtrip_phase_offset_rad=phase_rad,
            )
            local_er = er_values(local_power)[:, target]
            selected = int(np.argmax(local_er))
            center = local[selected]
            best_by_route[target] = max(best_by_route[target], local_er[selected])
            half_width /= 3
    fraction, separation = region_metrics(voltage, er, threshold_db)
    return {
        "worst_best_er_db": float(np.min(best_by_route)),
        "best_by_route_db": best_by_route,
        "region_fraction": fraction,
        "region_separation_v": separation,
        "power": power,
        "er": er,
    }


def quantiles(values: np.ndarray) -> tuple[float, float, float]:
    finite = np.asarray(values)[np.isfinite(values)]
    if not len(finite):
        return np.nan, np.nan, np.nan
    return tuple(float(value) for value in np.percentile(finite, [5, 50, 95]))


def summarize_ensemble(
    params: dict,
    optical,
    voltage: np.ndarray,
    phases: np.ndarray,
    threshold_db: float,
) -> dict[str, np.ndarray]:
    best_er = np.empty(len(phases))
    fraction = np.empty(len(phases))
    separation = np.empty(len(phases))
    for index, phase in enumerate(phases):
        metrics = evaluate(params, optical, voltage, phase, threshold_db)
        best_er[index] = metrics["worst_best_er_db"]
        fraction[index] = metrics["region_fraction"]
        separation[index] = metrics["region_separation_v"]
    return {"best_er": best_er, "fraction": fraction, "separation": separation}


def optimize_shared_phase(
    params: dict,
    optical,
    voltage: np.ndarray,
    threshold_db: float,
    metric: str,
    seed: int,
) -> tuple[np.ndarray, dict[str, float | np.ndarray]]:
    """Optimize two shared static relative phases for ER or region volume."""
    if metric not in {"worst_best_er_db", "region_fraction"}:
        raise ValueError(f"unsupported phase objective: {metric}")

    def objective(relative_phase: np.ndarray) -> float:
        phase = np.array([0.0, relative_phase[0], relative_phase[1]])
        return -float(evaluate(params, optical, voltage, phase, threshold_db)[metric])

    result = differential_evolution(
        objective,
        [(-np.pi, np.pi), (-np.pi, np.pi)],
        seed=seed,
        popsize=12,
        maxiter=35,
        tol=2e-3,
        polish=True,
        workers=1,
    )
    phase = np.array([0.0, result.x[0], result.x[1]])
    return phase, evaluate(params, optical, voltage, phase, threshold_db)


def optimize_cavity_survival_at_er(
    params: dict,
    optical,
    voltage: np.ndarray,
    threshold_db: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Maximize worst-route cavity survival subject to a route ER threshold."""
    config = params["recombination_model"]
    incident = mmi_matrix(params, 1.0)[:, int(config["input_port_index_zero_based"])]
    incident_power = np.abs(incident) ** 2
    reflection = reflected_amplitudes(optical, voltage)
    survival = (np.abs(reflection) ** 2 * incident_power[None, :]).sum(
        axis=1
    ) / incident_power.sum()

    def route_survival(relative_phase: np.ndarray) -> np.ndarray:
        phase = np.array([0.0, relative_phase[0], relative_phase[1]])
        power = output_powers(
            params,
            optical,
            voltage,
            imbalance_scale=1.0,
            roundtrip_phase_offset_rad=phase,
        )
        er = er_values(power)
        best = np.zeros(3)
        for target in range(3):
            valid = er[:, target] >= threshold_db
            if np.any(valid):
                best[target] = float(np.max(survival[valid]))
        return best

    result = differential_evolution(
        lambda relative_phase: -float(np.min(route_survival(relative_phase))),
        [(-np.pi, np.pi), (-np.pi, np.pi)],
        seed=seed,
        popsize=10,
        maxiter=30,
        tol=2e-3,
        polish=False,
        workers=1,
    )
    phase = np.array([0.0, result.x[0], result.x[1]])
    return phase, route_survival(result.x)


def minimize_joint_driver_span(
    params: dict,
    optical,
    voltage: np.ndarray,
    threshold_db: float,
    seed: int,
    initial_candidates: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Jointly minimize the maximum electrode excursion for three routes.

    One pair of relative round-trip arm phases is shared by all states.  Each
    routing state has its own three-cavity voltage tuple.  The objective is the
    largest peak-to-peak voltage change required on any electrode, subject to
    every target/off-target power ratio exceeding ``threshold_db``.
    """
    rng = np.random.default_rng(seed)
    baseline = optical.baseline_v

    def state_power(relative_voltage: np.ndarray, phase: np.ndarray) -> np.ndarray:
        return output_powers(
            params,
            optical,
            baseline[None, :] + relative_voltage,
            imbalance_scale=1.0,
            roundtrip_phase_offset_rad=phase,
        )

    def compact_grid_start(
        phase: np.ndarray, required_er_db: float
    ) -> tuple[np.ndarray, float] | None:
        power = output_powers(
            params,
            optical,
            voltage,
            imbalance_scale=1.0,
            roundtrip_phase_offset_rad=phase,
        )
        er = er_values(power)
        candidate_sets = [
            np.where(er[:, target] >= required_er_db)[0] for target in range(3)
        ]
        if any(len(indices) == 0 for indices in candidate_sets):
            return None
        candidates = [voltage[indices] - baseline for indices in candidate_sets]
        for target in range(3):
            if len(candidates[target]) > 1200:
                score = er[candidate_sets[target], target] - 0.02 * np.sum(
                    candidates[target] ** 2, axis=1
                )
                keep = np.argpartition(score, -1200)[-1200:]
                candidates[target] = candidates[target][keep]

        starts = [
            [voltage[int(np.argmax(er[:, target]))] - baseline for target in range(3)]
        ]
        starts.extend(
            [candidate[rng.integers(len(candidate))] for candidate in candidates]
            for _ in range(100)
        )
        best_states = None
        best_span = np.inf
        for selected in starts:
            selected = [np.asarray(state, dtype=float) for state in selected]
            for _ in range(8):
                changed = False
                for target in range(3):
                    fixed = np.asarray(
                        [selected[index] for index in range(3) if index != target]
                    )
                    low = np.minimum(candidates[target], fixed.min(axis=0))
                    high = np.maximum(candidates[target], fixed.max(axis=0))
                    score = np.max(high - low, axis=1)
                    chosen = candidates[target][int(np.argmin(score))]
                    changed = changed or not np.allclose(selected[target], chosen)
                    selected[target] = chosen
                if not changed:
                    break
            states = np.asarray(selected)
            span = float(np.max(np.ptp(states, axis=0)))
            if span < best_span:
                best_states = states
                best_span = span
        if best_states is None:
            return None
        return best_states, best_span

    def constraints(vector: np.ndarray, required_er_db: float) -> np.ndarray:
        phase = np.array([0.0, vector[0], vector[1]])
        states = vector[2:11].reshape(3, 3)
        span = vector[11]
        power = state_power(states, phase)
        values = []
        for target in range(3):
            for competitor in range(3):
                if competitor == target:
                    continue
                values.append(
                    10
                    * np.log10(
                        max(power[target, target], np.finfo(float).tiny)
                        / max(power[target, competitor], np.finfo(float).tiny)
                    )
                    - required_er_db
                )
        for electrode in range(3):
            for first in range(3):
                for second in range(first + 1, 3):
                    difference = states[first, electrode] - states[second, electrode]
                    values.extend((span - difference, span + difference))
        return np.asarray(values)

    bounds = [(-np.pi, np.pi)] * 2 + [(-5.0, 5.0)] * 9 + [(0.0, 10.0)]

    def refine(vector_0: np.ndarray, required_er_db: float):
        result = minimize(
            lambda vector: vector[11],
            vector_0,
            method="SLSQP",
            bounds=bounds,
            constraints={
                "type": "ineq",
                "fun": lambda vector: constraints(vector, required_er_db),
            },
            options={"ftol": 1e-10, "maxiter": 1800, "disp": False},
        )
        if np.min(constraints(result.x, required_er_db)) >= -2e-5:
            return result
        return None

    def multistart(required_er_db: float):
        phase, _ = optimize_cavity_survival_at_er(
            params, optical, voltage, required_er_db, seed=seed + 1000
        )
        phase_starts = [phase, np.zeros(3)]
        phase_starts.extend(
            phase + np.array([0.0, *rng.normal(scale=0.15, size=2)]) for _ in range(6)
        )
        best_result = None
        for phase_start, states_start in initial_candidates or []:
            initial_span = float(np.max(np.ptp(states_start, axis=0)))
            vector_0 = np.r_[phase_start[1:], states_start.ravel(), initial_span]
            result = refine(vector_0, required_er_db)
            if result is not None and (
                best_result is None or result.x[11] < best_result.x[11]
            ):
                best_result = result
        for phase_start in phase_starts:
            grid_start = compact_grid_start(phase_start, required_er_db)
            if grid_start is None:
                continue
            states, initial_span = grid_start
            vector_0 = np.r_[phase_start[1:], states.ravel(), initial_span + 1e-3]
            result = refine(vector_0, required_er_db)
            if result is not None and (
                best_result is None or result.x[11] < best_result.x[11]
            ):
                best_result = result
        return best_result

    # Direct optimization at a stringent ER can converge to a disconnected,
    # unnecessarily large-span branch. Track a feasible low-span branch from
    # 10 dB to the requested threshold and retain the better result.
    best = multistart(threshold_db)
    start_threshold_db = min(10.0, threshold_db)
    continuation = multistart(start_threshold_db)
    if continuation is not None:
        levels = np.arange(start_threshold_db + 0.25, threshold_db + 1e-9, 0.25)
        for required_er_db in levels:
            next_result = refine(continuation.x, float(required_er_db))
            if next_result is None:
                continuation = None
                break
            continuation = next_result
    if continuation is not None and (best is None or continuation.x[11] < best.x[11]):
        best = continuation
    if best is None:
        raise RuntimeError(
            "joint voltage-span optimization did not find a feasible point"
        )

    phase = np.array([0.0, best.x[0], best.x[1]])
    states = best.x[2:11].reshape(3, 3)
    power = state_power(states, phase)
    route_er = er_values(power)[np.arange(3), np.arange(3)]
    return phase, states, float(best.x[11]), route_er


def write_rows(rows: list[dict[str, object]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def add_quantile_band(ax, x, values, color, label):
    q05 = np.array([quantiles(item)[0] for item in values])
    q50 = np.array([quantiles(item)[1] for item in values])
    q95 = np.array([quantiles(item)[2] for item in values])
    ax.fill_between(x, q05, q95, color=color, alpha=0.18, linewidth=0)
    ax.plot(x, q50, color=color, lw=1.8, label=label)
    return q05, q50, q95


def plot_supplementary_phase_figure(
    params,
    nominal,
    phase_voltage,
    phases,
    phase_results,
    fitted_metrics,
    er_optimized_metrics,
    region_optimized_metrics,
    representative,
    threshold_db,
):
    """Render the optical-only phase sensitivity figure used in the supplement."""
    colors = {"5th percentile": RED, "median": BLUE, "95th percentile": GREEN}
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.75))
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.23, top=0.91, wspace=0.30)
    ax = axes[0]
    ax.scatter(
        100 * phase_results["fraction"],
        phase_results["best_er"],
        s=11,
        color="#7F8C99",
        alpha=0.52,
        edgecolors="none",
        label="sampled arm phases",
    )
    phase_markers = (
        ("fitted phase setting", fitted_metrics, "*", RED, 62),
        ("maximum point ER", er_optimized_metrics, "D", ORANGE, 30),
        ("maximum 10-dB region", region_optimized_metrics, "s", GREEN, 34),
        ("maximum 10-dB tolerance", load_tolerance_reference(), "P", PURPLE, 45),
    )
    for label, metrics, marker, color, size in phase_markers:
        ax.scatter(
            100 * metrics["region_fraction"],
            metrics["worst_best_er_db"],
            marker=marker,
            s=size,
            color=color,
            edgecolor="white",
            linewidth=0.55,
            label=label,
            zorder=4,
        )
    ax.set_xlabel("Worst-route fraction of $\\pm5$ V cube with ER $\\geq10$ dB (%)")
    ax.set_ylabel("Optimized worst-route ER (dB)")
    ax.set_title("a  Relative arm-phase sensitivity", loc="left", fontweight="bold")
    ax.grid(color="#E1E1E1", lw=0.5)
    ax.legend(frameon=False, fontsize=7.0, loc="upper right")

    ax = axes[1]
    er_thresholds = np.linspace(0, 32, 129)
    phase_curves = []
    for label in ("5th percentile", "median", "95th percentile"):
        result = evaluate(
            params,
            nominal,
            phase_voltage,
            phases[representative[label]],
            threshold_db,
        )
        phase_curves.append((label, result, colors[label], "-"))
    phase_curves.append(("region-optimized", region_optimized_metrics, "#222222", "--"))
    for label, result, color, linestyle in phase_curves:
        curve = [
            100 * min(np.mean(result["er"][:, route] >= level) for route in range(3))
            for level in er_thresholds
        ]
        ax.semilogy(
            er_thresholds,
            np.maximum(curve, 1e-3),
            color=color,
            ls=linestyle,
            lw=1.45,
            label=label,
        )
    ax.axvline(10, color=GRAY, ls=":", lw=0.9)
    ax.set_xlabel("Extinction-ratio threshold (dB)")
    ax.set_ylabel("Worst-route voltage-cube fraction (%)")
    ax.set_title("b  High-extinction region size", loc="left", fontweight="bold")
    ax.set_xlim(0, 32)
    ax.set_ylim(1e-3, 100)
    ax.grid(color="#E1E1E1", lw=0.5, which="both")
    ax.legend(frameon=False, fontsize=7.0, loc="upper right")
    fig.text(
        0.5,
        0.025,
        "Optical-only sensitivity model: empirical coherent backgrounds and detector calibration excluded.",
        ha="center",
        fontsize=7.0,
        color="#444444",
    )
    fig.savefig(SUPPLEMENTARY_PHASE_PDF)
    fig.savefig(SUPPLEMENTARY_PHASE_SVG)
    plt.close(fig)


def main() -> None:
    load_tolerance_reference()
    params, nominal = load_model()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUPPLEMENTARY_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(240917)
    threshold_db = 10.0

    # The first round-trip arm phase is the reference.  The other two cover
    # all inequivalent static phase offsets modulo 2 pi.
    phase_count = 240
    phases = np.column_stack(
        (
            np.zeros(phase_count),
            rng.uniform(-np.pi, np.pi, phase_count),
            rng.uniform(-np.pi, np.pi, phase_count),
        )
    )
    phase_voltage = voltage_grid(nominal.baseline_v, 5.0, 21)
    phase_results = summarize_ensemble(
        params, nominal, phase_voltage, phases, threshold_db
    )

    # Include the fitted phase setting as a named reference (zero additional
    # phase means the fitted static arm phases remain in the model).
    fitted_phase = np.zeros(3)
    fitted_metrics = evaluate(
        params, nominal, phase_voltage, fitted_phase, threshold_db
    )
    er_optimized_phase, er_optimized_metrics = optimize_shared_phase(
        params,
        nominal,
        phase_voltage,
        threshold_db,
        "worst_best_er_db",
        seed=23,
    )
    region_optimized_phase, region_optimized_metrics = optimize_shared_phase(
        params,
        nominal,
        phase_voltage,
        threshold_db,
        "region_fraction",
        seed=29,
    )

    # Select actual Monte Carlo samples closest to the 5th, 50th and 95th
    # percentile optimized worst-route ERs for representative region plots.
    representative = {}
    for label, percentile in (
        ("5th percentile", 5),
        ("median", 50),
        ("95th percentile", 95),
    ):
        target = np.percentile(phase_results["best_er"], percentile)
        representative[label] = int(
            np.argmin(np.abs(phase_results["best_er"] - target))
        )

    if "--phase-figure-only" in sys.argv:
        plot_supplementary_phase_figure(
            params,
            nominal,
            phase_voltage,
            phases,
            phase_results,
            fitted_metrics,
            er_optimized_metrics,
            region_optimized_metrics,
            representative,
            threshold_db,
        )
        print(f"wrote {SUPPLEMENTARY_PHASE_PDF}")
        return

    # Controlled Q sweep: fixed coupling ratios and eta in Hz/V.  This isolates
    # linewidth narrowing.  Coarser grids/ensemble keep the run reproducible.
    sweep_phases = phases[:72]
    sweep_voltage = voltage_grid(nominal.baseline_v, 5.0, 17)
    q_scales = np.geomspace(0.5, 3.0, 13)
    q_results = []
    q_fitted_phase_er = []
    q_fitted_phase_span = []
    for scale in q_scales:
        changed = rebuild_cavities(
            nominal,
            q_scale=np.full(3, scale),
            coupling_scale=np.ones(3),
            eta_scale=np.ones(3),
            detuning_shift_pm=np.zeros(3),
        )
        q_results.append(
            summarize_ensemble(
                params, changed, sweep_voltage, sweep_phases, threshold_db
            )
        )
        optimized_er, _, optimized_voltage = refined_route_metrics(
            params,
            changed,
            sweep_voltage,
            imbalance_scale=1.0,
            roundtrip_phase_offset_rad=fitted_phase,
        )
        q_fitted_phase_er.append(float(np.min(optimized_er)))
        # Maximum peak-to-peak excursion required on any one electrode across
        # the three independently ER-maximizing routing tuples.
        q_fitted_phase_span.append(float(np.max(np.ptp(optimized_voltage, axis=0))))

    # Controlled overcoupling sweep: identical x for the three cavities, fixed
    # loaded Q and eta.  x > 1 is the overcoupled regime.
    coupling_values = np.geomspace(1.05, 12.0, 13)
    coupling_results = []
    coupling_fitted_phase_er = []
    coupling_fitted_phase_span = []
    for coupling in coupling_values:
        coupling_scale = np.full(3, coupling) / nominal.coupling_ratio
        changed = rebuild_cavities(
            nominal,
            q_scale=np.ones(3),
            coupling_scale=coupling_scale,
            eta_scale=np.ones(3),
            detuning_shift_pm=np.zeros(3),
        )
        coupling_results.append(
            summarize_ensemble(
                params, changed, sweep_voltage, sweep_phases, threshold_db
            )
        )
        optimized_er, _, optimized_voltage = refined_route_metrics(
            params,
            changed,
            sweep_voltage,
            imbalance_scale=1.0,
            roundtrip_phase_offset_rad=fitted_phase,
        )
        coupling_fitted_phase_er.append(float(np.min(optimized_er)))
        coupling_fitted_phase_span.append(
            float(np.max(np.ptp(optimized_voltage, axis=0)))
        )

    # Optimize cavity-only survival subject to a useful routing contrast.  The
    # same pair of static phases is shared by all three routing states.
    throughput_threshold_db = 15.0
    throughput_cases = [("fitted heterogeneous", nominal)]
    for coupling in (3.0, 5.0, 10.0, 1e9):
        throughput_cases.append(
            (
                "lossless" if coupling > 1e8 else f"common x={coupling:g}",
                rebuild_cavities(
                    nominal,
                    q_scale=np.ones(3),
                    coupling_scale=np.full(3, coupling) / nominal.coupling_ratio,
                    eta_scale=np.ones(3),
                    detuning_shift_pm=np.zeros(3),
                ),
            )
        )
    throughput_results = []
    for case_index, (label, changed) in enumerate(throughput_cases):
        phase, survival = optimize_cavity_survival_at_er(
            params,
            changed,
            phase_voltage,
            throughput_threshold_db,
            seed=115 + case_index,
        )
        throughput_results.append((label, phase, survival))

    # True constrained driver requirement: jointly choose one shared phase
    # setting and three voltage tuples while enforcing ER >= 15 dB.
    joint_voltage_grid = voltage_grid(nominal.baseline_v, 5.0, 25)
    joint_cases = [
        ("fitted", nominal),
        (
            "$2Q$",
            rebuild_cavities(
                nominal, np.full(3, 2.0), np.ones(3), np.ones(3), np.zeros(3)
            ),
        ),
        (
            "$x=5$",
            rebuild_cavities(
                nominal,
                np.ones(3),
                np.full(3, 5.0) / nominal.coupling_ratio,
                np.ones(3),
                np.zeros(3),
            ),
        ),
        (
            "$2Q$, $x=5$",
            rebuild_cavities(
                nominal,
                np.full(3, 2.0),
                np.full(3, 5.0) / nominal.coupling_ratio,
                np.ones(3),
                np.zeros(3),
            ),
        ),
        (
            "lossless",
            rebuild_cavities(
                nominal,
                np.ones(3),
                np.full(3, 1e9) / nominal.coupling_ratio,
                np.ones(3),
                np.zeros(3),
            ),
        ),
    ]
    joint_results = []
    # Preserve the verified reciprocal-model solution as a deterministic
    # feasible start. The state rows are ordered by target output here.
    fitted_initial_candidates = [
        (
            np.array([0.0, 0.60531448, 0.92487868]),
            np.array(
                [
                    [-0.79705967, 0.74789970, -0.99445371],
                    [-0.01671784, -0.58758762, -1.56846997],
                    [-2.47548620, 1.87118073, -1.87733616],
                ]
            ),
        )
    ]
    for case_index, (label, changed) in enumerate(joint_cases):
        phase, states, span, route_er = minimize_joint_driver_span(
            params,
            changed,
            joint_voltage_grid,
            throughput_threshold_db,
            seed=2600 + case_index,
            initial_candidates=fitted_initial_candidates if label == "fitted" else None,
        )
        joint_results.append((label, phase, states, span, route_er))

    rows: list[dict[str, object]] = []
    for objective, phase, metrics in (
        ("maximize_worst_route_er", er_optimized_phase, er_optimized_metrics),
        ("maximize_10db_region", region_optimized_phase, region_optimized_metrics),
    ):
        rows.append(
            {
                "analysis": "optimized_shared_arm_phase",
                "case": objective,
                "phase_arm2_rad": phase[1],
                "phase_arm3_rad": phase[2],
                "optimized_worst_route_er_db": metrics["worst_best_er_db"],
                "worst_route_region_fraction_10db": metrics["region_fraction"],
                "max_pairwise_min_linf_region_distance_v": metrics[
                    "region_separation_v"
                ],
            }
        )
    for label, phase, survival in throughput_results:
        for target in range(3):
            rows.append(
                {
                    "analysis": "cavity_survival_at_15db_er",
                    "case": label,
                    "metric": f"route_{target + 1}",
                    "phase_arm2_rad": phase[1],
                    "phase_arm3_rad": phase[2],
                    "cavity_survival": survival[target],
                    "cavity_only_loss_db": -10 * np.log10(survival[target]),
                }
            )
    for label, phase, states, span, route_er in joint_results:
        rows.append(
            {
                "analysis": "joint_best_found_driver_span_15db",
                "case": label,
                "phase_arm2_rad": phase[1],
                "phase_arm3_rad": phase[2],
                "max_electrode_span_between_states_v": span,
                "route_1_er_db": route_er[0],
                "route_2_er_db": route_er[1],
                "route_3_er_db": route_er[2],
                "route_1_relative_voltages_v": states[0].tolist(),
                "route_2_relative_voltages_v": states[1].tolist(),
                "route_3_relative_voltages_v": states[2].tolist(),
            }
        )
    for index, phase in enumerate(phases):
        rows.append(
            {
                "analysis": "phase_monte_carlo",
                "case": index,
                "phase_arm2_rad": phase[1],
                "phase_arm3_rad": phase[2],
                "optimized_worst_route_er_db": phase_results["best_er"][index],
                "worst_route_region_fraction_10db": phase_results["fraction"][index],
                "max_pairwise_min_linf_region_distance_v": phase_results["separation"][
                    index
                ],
            }
        )
    for sweep_index, (scale, result) in enumerate(zip(q_scales, q_results)):
        for metric, values in result.items():
            q05, q50, q95 = quantiles(values)
            rows.append(
                {
                    "analysis": "loaded_q_scale",
                    "case": scale,
                    "metric": metric,
                    "p05": q05,
                    "median": q50,
                    "p95": q95,
                }
            )
        rows.append(
            {
                "analysis": "loaded_q_scale_fitted_phases",
                "case": scale,
                "optimized_worst_route_er_db": q_fitted_phase_er[sweep_index],
                "max_electrode_span_between_optimized_states_v": q_fitted_phase_span[
                    sweep_index
                ],
            }
        )
    for sweep_index, (coupling, result) in enumerate(
        zip(coupling_values, coupling_results)
    ):
        for metric, values in result.items():
            q05, q50, q95 = quantiles(values)
            rows.append(
                {
                    "analysis": "common_coupling_ratio",
                    "case": coupling,
                    "metric": metric,
                    "p05": q05,
                    "median": q50,
                    "p95": q95,
                }
            )
        rows.append(
            {
                "analysis": "common_coupling_fitted_phases",
                "case": coupling,
                "optimized_worst_route_er_db": coupling_fitted_phase_er[sweep_index],
                "max_electrode_span_between_optimized_states_v": coupling_fitted_phase_span[
                    sweep_index
                ],
            }
        )
    write_rows(rows)

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.3))
    fig.subplots_adjust(
        left=0.07, right=0.98, bottom=0.10, top=0.91, wspace=0.34, hspace=0.45
    )
    fig.suptitle(
        "Arm-phase and cavity-design effects on the $3\\times3$ routing regions",
        y=0.975,
        fontsize=11,
    )

    # a) Phase ensemble: optimized ER against ER-region volume.
    ax = axes[0, 0]
    scatter = ax.scatter(
        100 * phase_results["fraction"],
        phase_results["best_er"],
        c=phase_results["separation"],
        cmap="viridis",
        s=16,
        alpha=0.75,
        edgecolors="none",
    )
    ax.scatter(
        100 * fitted_metrics["region_fraction"],
        fitted_metrics["worst_best_er_db"],
        marker="*",
        s=75,
        color=RED,
        edgecolor="white",
        linewidth=0.5,
        label="fitted arm phases",
        zorder=4,
    )
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Region separation (V)", fontsize=7)
    ax.set_xlabel("Worst-route fraction of $\\pm5$ V cube with ER $\\geq10$ dB (%)")
    ax.set_ylabel("Optimized worst-route ER (dB)")
    ax.set_title("a) Random relative arm phases", loc="left")
    ax.grid(color="#DDDDDD", lw=0.5)
    ax.legend(frameon=False)

    # b) Representative survival curves: voltage volume above each ER.
    ax = axes[0, 1]
    er_thresholds = np.linspace(0, 30, 121)
    colors = {"5th percentile": RED, "median": BLUE, "95th percentile": GREEN}
    for label in ("5th percentile", "median", "95th percentile"):
        result = evaluate(
            params,
            nominal,
            phase_voltage,
            phases[representative[label]],
            threshold_db,
        )
        curve = [
            100 * min(np.mean(result["er"][:, route] >= level) for route in range(3))
            for level in er_thresholds
        ]
        ax.semilogy(
            er_thresholds,
            np.maximum(curve, 1e-3),
            color=colors[label],
            lw=1.7,
            label=label,
        )
    ax.axvline(threshold_db, color=GRAY, ls="--", lw=0.9)
    ax.set_xlabel("Extinction-ratio threshold (dB)")
    ax.set_ylabel("Worst-route voltage-cube fraction (%)")
    ax.set_title("b) Size of the extinction regions", loc="left")
    ax.set_ylim(1e-3, 100)
    ax.grid(color="#DDDDDD", lw=0.5, which="both")
    ax.legend(frameon=False)

    # c) Loaded Q versus optimized voltage span at the fitted phase setting.
    ax = axes[0, 2]
    ax.plot(
        q_scales, q_fitted_phase_span, color=BLUE, lw=1.8, label="optimized-state span"
    )
    inverse_q_guide = q_fitted_phase_span[np.argmin(np.abs(q_scales - 1))] / q_scales
    ax.plot(q_scales, inverse_q_guide, color=GRAY, ls=":", lw=1.2, label="$1/Q$ guide")
    ax.axvline(1, color=ORANGE, ls="--", lw=1.0, label="fitted $Q$")
    ax.set_xscale("log")
    ax.set_xlabel("Loaded-$Q$ multiplier")
    ax.set_ylabel("Maximum electrode span across routing states (V)")
    ax.set_title("c) Higher $Q$ reduces the required drive", loc="left")
    ax.grid(color="#DDDDDD", lw=0.5)
    ax.legend(frameon=False)

    # d) Q trades lower voltage separation against smaller voltage regions.
    ax = axes[1, 0]
    _, fraction_median, _ = add_quantile_band(
        ax,
        q_scales,
        [100 * item["fraction"] for item in q_results],
        BLUE,
        "ER-region volume",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Loaded-$Q$ multiplier")
    ax.set_ylabel("Median 10-dB region fraction (%)", color=BLUE)
    ax.tick_params(axis="y", colors=BLUE)
    ax2 = ax.twinx()
    normalized_bandwidth = 1 / q_scales
    ax2.plot(q_scales, normalized_bandwidth, color=ORANGE, lw=1.8)
    ax2.set_ylabel("Relative cavity bandwidth", color=ORANGE)
    ax2.tick_params(axis="y", colors=ORANGE)
    ax.set_title("d) Voltage and robustness trade-off", loc="left")
    ax.grid(color="#DDDDDD", lw=0.5)

    # e) Overcoupling versus ER.
    ax = axes[1, 1]
    add_quantile_band(
        ax,
        coupling_values,
        [item["best_er"] for item in coupling_results],
        GREEN,
        "phase median",
    )
    nominal_geomean = float(np.exp(np.mean(np.log(nominal.coupling_ratio))))
    ax.axvline(
        nominal_geomean, color=ORANGE, ls="--", lw=1.0, label="fitted geometric mean"
    )
    ax.set_xscale("log")
    ax.set_xlabel("Common overcoupling ratio $x=\\kappa_e/\\kappa_i$")
    ax.set_ylabel("Optimized worst-route ER (dB)")
    ax.set_title("e) More nearly all-pass cavities", loc="left")
    ax.grid(color="#DDDDDD", lw=0.5)
    ax.legend(frameon=False)

    # f) Coupling effect on ER regions and resonant reflection amplitude.
    ax = axes[1, 2]
    _, coupling_fraction_median, _ = add_quantile_band(
        ax,
        coupling_values,
        [100 * item["fraction"] for item in coupling_results],
        GREEN,
        "10-dB region volume",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Common overcoupling ratio $x=\\kappa_e/\\kappa_i$")
    ax.set_ylabel("Median 10-dB region fraction (%)", color=GREEN)
    ax.tick_params(axis="y", colors=GREEN)
    ax2 = ax.twinx()
    resonant_amplitude = (coupling_values - 1) / (coupling_values + 1)
    ax2.plot(coupling_values, resonant_amplitude, color=ORANGE, lw=1.8)
    ax2.set_ylabel("On-resonance $|r|=(x-1)/(x+1)$", color=ORANGE)
    ax2.tick_params(axis="y", colors=ORANGE)
    ax.set_title("f) Loss and useful-region size", loc="left")
    ax.grid(color="#DDDDDD", lw=0.5)

    fig.text(
        0.5,
        0.025,
        "Bands show 5th--95th percentiles over random relative round-trip arm phases; lines show medians. "
        "Region separation is pairwise and is a lower bound on a jointly optimized three-state drive span.",
        ha="center",
        fontsize=6.8,
        color="#555555",
    )
    fig.savefig(OUTPUT_PDF)
    fig.savefig(OUTPUT_SVG)
    fig.savefig(OUTPUT_PNG, dpi=300)
    plt.close(fig)

    # Reachable-simplex plots for the representative phase cases.
    fig, axes = plt.subplots(1, 4, figsize=(10.4, 2.75))
    cases = [("fitted", None)] + [
        (label, representative[label])
        for label in ("5th percentile", "median", "95th percentile")
    ]
    for ax, (label, index) in zip(axes, cases):
        phase = fitted_phase if index is None else phases[index]
        result = evaluate(params, nominal, phase_voltage, phase, threshold_db)
        normalized = normalize_power(result["power"])
        xy = barycentric_xy(normalized)
        ax.hexbin(
            xy[:, 0],
            xy[:, 1],
            gridsize=100,
            extent=(-0.01, 1.01, -0.01, 0.88),
            mincnt=1,
            cmap=mpl.colors.ListedColormap([BLUE]),
            alpha=0.62,
            linewidths=0,
            rasterized=True,
        )
        simplex = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2], [0, 0]])
        ax.plot(simplex[:, 0], simplex[:, 1], color="#444444", lw=0.9)
        ax.text(-0.02, -0.035, "$P_1$", ha="center", va="top")
        ax.text(1.02, -0.035, "$P_2$", ha="center", va="top")
        ax.text(0.5, np.sqrt(3) / 2 + 0.02, "$P_3$", ha="center", va="bottom")
        ax.set_title(
            f"{label}\nworst ER={result['worst_best_er_db']:.1f} dB; "
            f"region={100*result['region_fraction']:.2f}%",
            fontsize=8,
        )
        ax.set_xlim(-0.07, 1.07)
        ax.set_ylim(-0.07, 0.93)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.suptitle(
        "Reachable normalized powers for representative arm-phase offsets", fontsize=10
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(OUTPUT_REGIONS_PDF)
    fig.savefig(OUTPUT_REGIONS_SVG)
    plt.close(fig)

    # Compact publication figure 1: relative-arm-phase sensitivity.
    plot_supplementary_phase_figure(
        params,
        nominal,
        phase_voltage,
        phases,
        phase_results,
        fitted_metrics,
        er_optimized_metrics,
        region_optimized_metrics,
        representative,
        threshold_db,
    )

    # Compact publication figure 2: cavity-design trade-offs.
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.55))
    fig.subplots_adjust(left=0.070, right=0.975, bottom=0.22, top=0.89, wspace=0.49)

    ax = axes[0]
    joint_labels = [item[0] for item in joint_results]
    joint_spans = [item[3] for item in joint_results]
    bars = ax.bar(
        np.arange(len(joint_labels)),
        joint_spans,
        color=[GRAY, BLUE, "#79A98A", "#3E8061", "#BFD9CA"],
        width=0.72,
    )
    for bar, value in zip(bars, joint_spans):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.07,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=5.7,
        )
    ax.set_xticks(np.arange(len(joint_labels)), joint_labels, rotation=25, ha="right")
    ax.set_ylabel("Best-found worst-electrode span (V)")
    ax.set_title("a  Joint drive optimization", loc="left", fontweight="bold")
    ax.set_ylim(0, max(joint_spans) * 1.25)
    ax.grid(axis="y", color="#E1E1E1", lw=0.5)

    ax = axes[1]
    coupling_er_median = np.array(
        [quantiles(item["best_er"])[1] for item in coupling_results]
    )
    coupling_fraction_median = np.array(
        [100 * quantiles(item["fraction"])[1] for item in coupling_results]
    )
    (line_er,) = ax.plot(
        coupling_values,
        coupling_er_median,
        color=GREEN,
        lw=1.65,
        label="optimized ER",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Overcoupling $\\kappa_e/\\kappa_i$")
    ax.set_ylabel("Median optimized ER (dB)", color=GREEN)
    ax.tick_params(axis="y", colors=GREEN)
    ax2 = ax.twinx()
    (line_region,) = ax2.plot(
        coupling_values,
        coupling_fraction_median,
        color=BLUE,
        lw=1.35,
        ls="--",
        label="10-dB region",
    )
    ax2.set_ylabel("Region (%)", color=BLUE, labelpad=1)
    ax2.tick_params(axis="y", colors=BLUE)
    ax.axvline(
        float(np.exp(np.mean(np.log(nominal.coupling_ratio)))),
        color=ORANGE,
        ls=":",
        lw=0.9,
    )
    ax.set_title("b  Extinction regions", loc="left", fontweight="bold")
    ax.grid(color="#E1E1E1", lw=0.5)
    ax.legend(
        [line_er, line_region],
        [line_er.get_label(), line_region.get_label()],
        frameon=False,
        fontsize=5.5,
        loc="lower right",
    )

    ax = axes[2]
    loss_labels = ["fitted", "$3$", "$5$", "$10$", "lossless"]
    worst_loss = [float(np.max(-10 * np.log10(item[2]))) for item in throughput_results]
    bars = ax.bar(
        np.arange(len(loss_labels)),
        worst_loss,
        color=[GRAY, "#79A98A", "#5A9875", "#3E8061", "#BFD9CA"],
        width=0.72,
    )
    for bar, value in zip(bars, worst_loss):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.045,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=5.7,
        )
    ax.set_xticks(np.arange(len(loss_labels)), loss_labels, rotation=25, ha="right")
    ax.set_ylabel("Worst cavity-only loss (dB)")
    ax.set_title("c  Loss at ER $\\geq15$ dB", loc="left", fontweight="bold")
    ax.set_ylim(0, max(worst_loss) * 1.22)
    ax.grid(axis="y", color="#E1E1E1", lw=0.5)
    fig.savefig(SUPPLEMENTARY_CAVITY_PDF)
    fig.savefig(SUPPLEMENTARY_CAVITY_SVG)
    plt.close(fig)

    phase_q = quantiles(phase_results["best_er"])
    fraction_q = quantiles(100 * phase_results["fraction"])
    separation_q = quantiles(phase_results["separation"])
    print(f"wrote {OUTPUT_PDF}")
    print(f"wrote {OUTPUT_REGIONS_PDF}")
    print(f"wrote {OUTPUT_CSV}")
    print(f"wrote {SUPPLEMENTARY_PHASE_PDF}")
    print(f"wrote {SUPPLEMENTARY_CAVITY_PDF}")
    print("phase MC optimized worst-route ER p05/median/p95:", phase_q)
    print("phase MC 10-dB region fraction (%) p05/median/p95:", fraction_q)
    print("phase MC region separation (V) p05/median/p95:", separation_q)
    print(
        "fitted phases: ER={:.2f} dB, region={:.4f}%, separation={:.3f} V".format(
            fitted_metrics["worst_best_er_db"],
            100 * fitted_metrics["region_fraction"],
            fitted_metrics["region_separation_v"],
        )
    )
    print(
        "ER-optimized shared phases:",
        er_optimized_phase,
        "ER=",
        er_optimized_metrics["worst_best_er_db"],
        "region%=",
        100 * er_optimized_metrics["region_fraction"],
    )
    print(
        "region-optimized shared phases:",
        region_optimized_phase,
        "ER=",
        region_optimized_metrics["worst_best_er_db"],
        "region%=",
        100 * region_optimized_metrics["region_fraction"],
    )
    for label, phase, survival in throughput_results:
        print(
            f"{label} at ER >= {throughput_threshold_db:g} dB:",
            "phase=",
            phase,
            "cavity loss dB=",
            -10 * np.log10(survival),
        )
    for label, phase, states, span, route_er in joint_results:
        print(
            f"{label} joint best-found solution at ER >= {throughput_threshold_db:g} dB:",
            "span=",
            span,
            "phase=",
            phase,
            "ER=",
            route_er,
            "relative states=",
            states,
        )
    print(
        "Q scale medians (ER dB):",
        [quantiles(item["best_er"])[1] for item in q_results],
    )
    print(
        "Q scale medians (region %):",
        [100 * quantiles(item["fraction"])[1] for item in q_results],
    )
    print(
        "Q scale medians (separation V):",
        [quantiles(item["separation"])[1] for item in q_results],
    )
    print("Q scale fitted-phase optimized ER (dB):", q_fitted_phase_er)
    print("Q scale fitted-phase state span (V):", q_fitted_phase_span)
    print(
        "coupling medians (ER dB):",
        [quantiles(item["best_er"])[1] for item in coupling_results],
    )
    print(
        "coupling medians (region %):",
        [100 * quantiles(item["fraction"])[1] for item in coupling_results],
    )


if __name__ == "__main__":
    main()
