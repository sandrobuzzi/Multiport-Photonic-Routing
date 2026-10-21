#!/usr/bin/env python3
"""Reconstruct and quantify the targeted voltage-cube extinction capture."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from tfln_paths import DATA_ROOT, OUTPUT_ROOT

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

AWG_SAMPLE_RATE_HZ = 2.4e9
TARGETED_PROJECTION_ITERATIONS = 10
EPSILON = 1e-6
CHANNEL_COLORS = ("#cc0000", "#007700", "#0055cc")
CHANNEL_LABELS = (r"$P_1$", r"$P_2$", r"$P_3$")


def compute_fourier_bins(f_lo_hz: float, f_hi_hz: float, period_s: float) -> np.ndarray:
    return np.arange(
        int(np.ceil(f_lo_hz * period_s)), int(np.floor(f_hi_hz * period_s)) + 1
    )


def synthesize_multisine(n_samples: int, bins: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0.0, 2.0 * np.pi, (3, len(bins)))
    signals = np.zeros((3, n_samples), dtype=np.float64)
    nyquist_index = n_samples // 2
    for channel in range(3):
        spectrum = np.zeros(n_samples, dtype=np.complex128)
        coefficients = (n_samples / 2.0) * np.exp(1j * phases[channel])
        spectrum[bins] = coefficients
        valid = (bins > 0) & (bins < nyquist_index)
        spectrum[n_samples - bins[valid]] = np.conj(coefficients[valid])
        signals[channel] = np.fft.ifft(spectrum).real
    return signals


def orthogonalize_channels(signals: np.ndarray) -> np.ndarray:
    original_rms = np.std(signals, axis=1)
    q_matrix, _ = np.linalg.qr(signals.T)
    orthogonal = q_matrix.T
    return orthogonal * (original_rms / np.std(orthogonal, axis=1))[:, None]


def normalize_uniform_remap(signals: np.ndarray) -> np.ndarray:
    output = np.empty_like(signals)
    target = np.linspace(-1.0, 1.0, signals.shape[1])
    for channel in range(signals.shape[0]):
        order = np.argsort(signals[channel])
        output[channel, order] = target
    return output


def build_uniform_trajectory(
    n_samples: int,
    period_s: float,
    f_lo_hz: float,
    f_hi_hz: float,
    seed: int,
) -> np.ndarray:
    bins = compute_fourier_bins(f_lo_hz, f_hi_hz, period_s)
    signals = orthogonalize_channels(synthesize_multisine(n_samples, bins, seed))
    for channel in range(signals.shape[0]):
        signals[channel] *= 0.999 / np.max(np.abs(signals[channel]))
    return normalize_uniform_remap(signals)


def bandpass_project(
    signals: np.ndarray,
    period_s: float,
    f_lo_hz: float,
    f_hi_hz: float,
) -> np.ndarray:
    bins = compute_fourier_bins(f_lo_hz, f_hi_hz, period_s)
    output = np.empty_like(signals)
    for channel in range(signals.shape[0]):
        spectrum = np.fft.rfft(signals[channel])
        keep = np.zeros(len(spectrum), dtype=bool)
        keep[bins] = True
        spectrum[~keep] = 0.0
        output[channel] = np.fft.irfft(spectrum, n=signals.shape[1])
    return output


def build_targeted_trajectory(
    targets_awg_v: np.ndarray,
    radius_v: float,
    strength: float,
    full_scale_v: float,
    n_samples: int,
    period_s: float,
    f_lo_hz: float,
    f_hi_hz: float,
    seed: int,
) -> np.ndarray:
    """Reproduce the acquisition script's ten-step attract/project waveform."""
    targets = np.asarray(targets_awg_v, dtype=np.float64) / full_scale_v
    distances = [
        np.linalg.norm(targets[first] - targets[second])
        for first in range(3)
        for second in range(first + 1, 3)
    ]
    sigma = max(0.3 * float(np.median(distances)), 0.05)
    signals = build_uniform_trajectory(n_samples, period_s, f_lo_hz, f_hi_hz, seed)

    for _ in range(TARGETED_PROJECTION_ITERATIONS):
        points = signals.T
        squared_distances = np.stack(
            [np.sum((points - target) ** 2, axis=1) for target in targets], axis=1
        )
        log_weights = -squared_distances / (2.0 * sigma**2)
        log_weights -= np.max(log_weights, axis=1, keepdims=True)
        weights = np.exp(log_weights)
        weights /= np.sum(weights, axis=1, keepdims=True)
        signals = ((1.0 - strength) * points + strength * (weights @ targets)).T
        signals = bandpass_project(signals, period_s, f_lo_hz, f_hi_hz)
        for channel in range(signals.shape[0]):
            signals[channel] *= 0.999 / np.max(np.abs(signals[channel]))
    return signals


def periodic_interpolate(
    waveforms: np.ndarray, sample_indices: np.ndarray, delays: np.ndarray
) -> np.ndarray:
    n_samples = waveforms.shape[1]
    output = np.empty((3, len(sample_indices)), dtype=np.float64)
    for channel in range(3):
        effective_index = (sample_indices - delays[channel]) % n_samples
        lower = np.floor(effective_index).astype(int)
        fraction = effective_index - lower
        upper = (lower + 1) % n_samples
        output[channel] = (1.0 - fraction) * waveforms[
            channel, lower
        ] + fraction * waveforms[channel, upper]
    return output


def load_and_reconstruct(path: Path) -> dict[str, np.ndarray | float | str]:
    with np.load(path, allow_pickle=False) as data:
        period_s = float(data["awg_period_s"])
        n_awg_samples = int(round(period_s * AWG_SAMPLE_RATE_HZ))
        full_scale_v = float(data["awg_output_range_v"]) * float(data["awg_amplitude"])
        waveforms = build_targeted_trajectory(
            data["targeted_targets_awg_v"],
            float(data["targeted_radius_v"]),
            float(data["targeted_strength"]),
            full_scale_v,
            n_awg_samples,
            period_s,
            float(data["f_lo_hz"]),
            float(data["f_hi_hz"]),
            int(data["seed"]),
        )

        time_origin_s = float(data["time_origin_s"])
        output_delays_s = np.asarray(data["chirp_seed_d_out_s"], dtype=np.float64)
        input_delays_samples = (
            np.asarray(data["chirp_seed_d_in_s"], dtype=np.float64) * AWG_SAMPLE_RATE_HZ
        )

        output_times = []
        raw_outputs = []
        for channel in (2, 3, 4):
            output_times.append(
                np.asarray(data[f"scope_ch{channel}_time"], dtype=np.float64)
                - time_origin_s
                - output_delays_s[channel - 2]
            )
            raw_outputs.append(
                np.asarray(data[f"scope_ch{channel}_voltage"], dtype=np.float64)
            )

        reference_time = output_times[0]
        outputs = np.column_stack(
            [
                raw_outputs[0],
                np.interp(reference_time, output_times[1], raw_outputs[1]),
                np.interp(reference_time, output_times[2], raw_outputs[2]),
            ]
        )
        inputs = (
            periodic_interpolate(
                waveforms,
                reference_time * AWG_SAMPLE_RATE_HZ,
                input_delays_samples,
            ).T
            * full_scale_v
        )
        dc_biases = np.array(
            [
                float(data["k1_voltage"]),
                float(data["k2_voltage"]),
                float(data["k3_voltage"]),
            ]
        )

        return {
            "timestamp": str(data["timestamp"]),
            "laser_wavelength_nm": float(data["laser_wavelength_nm"]),
            "period_s": period_s,
            "time_s": reference_time,
            "outputs_v": outputs,
            "awg_voltages_v": inputs,
            "absolute_voltages_v": inputs + dc_biases,
        }


def normalize_outputs(outputs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    minima = np.min(outputs, axis=0)
    spans = np.ptp(outputs, axis=0)
    return (outputs - minima) / spans, minima, spans


def extinction_ratio(normalized_outputs: np.ndarray, target_channel: int) -> np.ndarray:
    leakage = np.max(np.delete(normalized_outputs, target_channel, axis=-1), axis=-1)
    return 10.0 * np.log10(
        (normalized_outputs[..., target_channel] + EPSILON) / (leakage + EPSILON)
    )


def analyze_capture(reconstructed: dict[str, np.ndarray | float | str]) -> dict:
    outputs = np.asarray(reconstructed["outputs_v"])
    normalized, minima, spans = normalize_outputs(outputs)
    all_extinction = np.column_stack(
        [extinction_ratio(normalized, target) for target in range(3)]
    )
    peak_flat_index = int(np.argmax(all_extinction))
    peak_index, target_channel = np.unravel_index(peak_flat_index, all_extinction.shape)

    sample_interval_s = float(np.median(np.diff(np.asarray(reconstructed["time_s"]))))
    samples_per_period = int(
        round(float(reconstructed["period_s"]) / sample_interval_s)
    )
    n_periods = len(outputs) // samples_per_period
    usable_samples = n_periods * samples_per_period
    if n_periods < 2:
        raise ValueError("At least two complete acquisition periods are required.")

    cycle_outputs = outputs[:usable_samples].reshape(n_periods, samples_per_period, 3)
    cycle_normalized = normalized[:usable_samples].reshape(
        n_periods, samples_per_period, 3
    )
    phase_index = peak_index % samples_per_period
    repeated_outputs = cycle_outputs[:, phase_index, :]
    repeated_normalized = cycle_normalized[:, phase_index, :]
    repeated_extinction = extinction_ratio(repeated_normalized, target_channel)
    phase_mean_output = np.mean(repeated_outputs, axis=0)
    phase_mean_normalized = (phase_mean_output - minima) / spans
    phase_mean_extinction = float(
        extinction_ratio(phase_mean_normalized, target_channel)
    )
    phase_repeatability = np.std(repeated_outputs, axis=0, ddof=1) / spans
    off_target_channels = [channel for channel in range(3) if channel != target_channel]
    one_sigma_er_ceiling = float(
        10.0
        * np.log10(
            (phase_mean_normalized[target_channel] + EPSILON)
            / (np.max(phase_repeatability[off_target_channels]) + EPSILON)
        )
    )

    return {
        "outputs_v": outputs,
        "normalized": normalized,
        "all_extinction_db": all_extinction,
        "cycle_outputs_v": cycle_outputs,
        "cycle_normalized": cycle_normalized,
        "minima_v": minima,
        "spans_v": spans,
        "sample_interval_s": sample_interval_s,
        "samples_per_period": samples_per_period,
        "n_periods": n_periods,
        "peak_index": peak_index,
        "phase_index": phase_index,
        "target_channel": target_channel,
        "peak_extinction_db": float(all_extinction[peak_index, target_channel]),
        "peak_outputs_v": outputs[peak_index],
        "peak_normalized": normalized[peak_index],
        "repeat_indices": phase_index + np.arange(n_periods) * samples_per_period,
        "repeat_extinction_db": repeated_extinction,
        "repeat_mean_extinction_db": float(np.mean(repeated_extinction)),
        "repeat_std_extinction_db": float(np.std(repeated_extinction, ddof=1)),
        "phase_mean_normalized": phase_mean_normalized,
        "phase_mean_extinction_db": phase_mean_extinction,
        "phase_repeatability_normalized": phase_repeatability,
        "one_sigma_er_ceiling_db": one_sigma_er_ceiling,
    }


def plot_summary(analysis: dict, output_stem: Path) -> None:
    mm = 1.0 / 25.4
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.0,
            "axes.titlesize": 7.0,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
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
    fig, axes = plt.subplots(1, 2, figsize=(178 * mm, 66 * mm))
    cycle_normalized = np.asarray(analysis["cycle_normalized"])
    n_periods = int(analysis["n_periods"])
    samples_per_period = int(analysis["samples_per_period"])
    phase_index = int(analysis["phase_index"])
    target_channel = int(analysis["target_channel"])
    dt_ns = float(analysis["sample_interval_s"]) * 1e9
    half_window = max(1, int(round(10.0 / dt_ns)))
    offsets = np.arange(-half_window, half_window + 1)
    phase_indices = (phase_index + offsets) % samples_per_period
    relative_time_ns = offsets * dt_ns

    local_outputs = cycle_normalized[:, phase_indices, :]
    mean_outputs = np.mean(local_outputs, axis=0)
    std_outputs = np.std(local_outputs, axis=0, ddof=1)
    for channel, (color, label) in enumerate(zip(CHANNEL_COLORS, CHANNEL_LABELS)):
        lower = np.clip(mean_outputs[:, channel] - std_outputs[:, channel], 1e-4, None)
        upper = np.clip(mean_outputs[:, channel] + std_outputs[:, channel], 1e-4, None)
        axes[0].fill_between(
            relative_time_ns, lower, upper, color=color, alpha=0.16, linewidth=0
        )
        axes[0].plot(
            relative_time_ns,
            np.clip(mean_outputs[:, channel], 1e-4, None),
            color=color,
            lw=0.9,
            label=label,
        )
    axes[0].axvline(0.0, color="#999999", ls="--", lw=0.5, zorder=0)
    axes[0].set_yscale("log")
    axes[0].set_ylim(1e-3, 2.6)
    axes[0].set_xlim(relative_time_ns[0], relative_time_ns[-1])
    axes[0].set_xticks([-10, -5, 0, 5, 10])
    axes[0].set_xlabel("Time from optimum (ns)")
    axes[0].set_ylabel("Normalized output power")
    axes[0].set_title("Output signals", fontweight="bold", pad=4)
    axes[0].legend(
        frameon=False,
        loc="upper right",
        ncol=3,
        columnspacing=0.8,
        handlelength=1.2,
        borderpad=0.3,
        labelspacing=0.2,
    )
    axes[0].grid(axis="y", which="major", color="#cccccc", lw=0.3, zorder=0)
    axes[0].text(
        -0.11, 1.06, "a", transform=axes[0].transAxes, fontweight="bold", fontsize=8
    )

    for cycle in range(n_periods):
        local_er = extinction_ratio(local_outputs[cycle], target_channel)
        axes[1].plot(
            relative_time_ns,
            local_er,
            color=CHANNEL_COLORS[target_channel],
            alpha=0.25,
            lw=0.65,
            label="individual periods" if cycle == 0 else "_nolegend_",
        )
    mean_local_output = np.mean(
        np.asarray(analysis["cycle_outputs_v"])[:, phase_indices, :], axis=0
    )
    mean_local_normalized = (
        mean_local_output - np.asarray(analysis["minima_v"])
    ) / np.asarray(analysis["spans_v"])
    axes[1].plot(
        relative_time_ns,
        extinction_ratio(mean_local_normalized, target_channel),
        color="black",
        lw=0.9,
        label="phase average",
    )
    repeat_values = np.asarray(analysis["repeat_extinction_db"])
    peak_value = float(analysis["peak_extinction_db"])
    axes[1].scatter(
        np.zeros(n_periods),
        repeat_values,
        s=11,
        color=CHANNEL_COLORS[target_channel],
        edgecolor="white",
        linewidth=0.35,
        zorder=4,
    )
    axes[1].axvline(0.0, color="#999999", ls="--", lw=0.5, zorder=0)
    axes[1].annotate(
        f"Maximum = {peak_value:.1f} dB",
        xy=(0.0, peak_value),
        xytext=(9.2, 25.5),
        arrowprops={"arrowstyle": "-", "color": "#777777", "lw": 0.5},
        ha="right",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0},
    )
    axes[1].set_xlabel("Time from optimum (ns)")
    axes[1].set_ylabel(r"Extinction ratio, $\mathrm{ER}_1$ (dB)")
    axes[1].set_xlim(relative_time_ns[0], relative_time_ns[-1])
    axes[1].set_xticks([-10, -5, 0, 5, 10])
    axes[1].set_ylim(-15.0, 28.0)
    axes[1].set_title("Extinction ratio", fontweight="bold", pad=4)
    axes[1].legend(
        frameon=False,
        loc="upper left",
        ncol=1,
        columnspacing=0.9,
        handlelength=1.3,
        borderpad=0.3,
        labelspacing=0.2,
    )
    axes[1].grid(axis="y", color="#cccccc", lw=0.3, zorder=0)
    axes[1].text(
        -0.11, 1.06, "b", transform=axes[1].transAxes, fontweight="bold", fontsize=8
    )

    for axis in axes:
        axis.set_axisbelow(True)

    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.20, top=0.84, wspace=0.31)
    for suffix in (".pdf", ".svg", ".png"):
        fig.savefig(
            output_stem.with_suffix(suffix),
            dpi=600 if suffix == ".png" else 300,
            facecolor="white",
            bbox_inches="tight",
            pad_inches=0.0,
        )
    plt.close(fig)


def write_results(
    reconstructed: dict[str, np.ndarray | float | str],
    analysis: dict,
    output_dir: Path,
) -> None:
    target_channel = int(analysis["target_channel"])
    peak_index = int(analysis["peak_index"])
    summary = {
        "source_timestamp": reconstructed["timestamp"],
        "laser_wavelength_nm": reconstructed["laser_wavelength_nm"],
        "metric": "independent full-capture min-max normalization; target / strongest off-target; epsilon=1e-6",
        "target_port": target_channel + 1,
        "maximum_instantaneous_extinction_db": analysis["peak_extinction_db"],
        "phase_averaged_extinction_db": analysis["phase_mean_extinction_db"],
        "same_phase_repeat_extinction_db": np.asarray(
            analysis["repeat_extinction_db"]
        ).tolist(),
        "same_phase_repeat_mean_extinction_db": analysis["repeat_mean_extinction_db"],
        "same_phase_repeat_std_extinction_db": analysis["repeat_std_extinction_db"],
        "peak_raw_detector_outputs_v": np.asarray(analysis["peak_outputs_v"]).tolist(),
        "peak_normalized_detector_outputs": np.asarray(
            analysis["peak_normalized"]
        ).tolist(),
        "same_phase_repeatability_normalized_1sigma": np.asarray(
            analysis["phase_repeatability_normalized"]
        ).tolist(),
        "one_sigma_repeatability_limited_ceiling_db": analysis[
            "one_sigma_er_ceiling_db"
        ],
        "peak_awg_voltages_v": np.asarray(reconstructed["awg_voltages_v"])[
            peak_index
        ].tolist(),
        "peak_absolute_voltages_v": np.asarray(reconstructed["absolute_voltages_v"])[
            peak_index
        ].tolist(),
        "complete_periods": analysis["n_periods"],
        "assumed_awg_sample_rate_hz": AWG_SAMPLE_RATE_HZ,
        "assumed_targeted_projection_iterations": TARGETED_PROJECTION_ITERATIONS,
    }
    (output_dir / "targeted_extinction_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    with (output_dir / "targeted_extinction_repeats.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["repeat", "extinction_db"])
        for repeat, value in enumerate(
            np.asarray(analysis["repeat_extinction_db"]), start=1
        ):
            writer.writerow([repeat, f"{value:.9f}"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DATA_ROOT
        / "fig4/single_channel_extinction/raw/uniform_cube_08_targeted_20260416_095737.npz",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_ROOT / "figures/supplementary"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reconstructed = load_and_reconstruct(args.input)
    analysis = analyze_capture(reconstructed)
    plot_summary(analysis, args.output_dir / "targeted_single_channel_extinction")
    write_results(reconstructed, analysis, args.output_dir)

    repeats = np.asarray(analysis["repeat_extinction_db"])
    print(f"Target port: P{int(analysis['target_channel']) + 1}")
    print(f"Maximum instantaneous extinction: {analysis['peak_extinction_db']:.3f} dB")
    print(f"Same-phase repeats: {', '.join(f'{value:.3f}' for value in repeats)} dB")
    print(
        "Same-phase repeat mean +/- standard deviation: "
        f"{analysis['repeat_mean_extinction_db']:.3f} +/- "
        f"{analysis['repeat_std_extinction_db']:.3f} dB"
    )
    print(f"Phase-averaged extinction: {analysis['phase_mean_extinction_db']:.3f} dB")
    print(
        f"One-sigma repeatability-limited ceiling: {analysis['one_sigma_er_ceiling_db']:.3f} dB"
    )


if __name__ == "__main__":
    main()
