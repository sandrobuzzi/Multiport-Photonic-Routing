#!/usr/bin/env python3
"""Fit low-pass models to the through-normalized electro-optic S21 trace.

The script treats the Touchstone S21 columns as real/imaginary data, removes
the separately measured through response in dB, and compares models on the
same samples. The best fixed-order fit's low-frequency level defines 0 dB
for the displayed measurement and model curves.

An important convention: the order-1 Butterworth magnitude response,

    |H(f)| = 1 / sqrt(1 + (f / f_3dB)^2),

is simultaneously the magnitude of a single-pole RC response and the square
root of a Lorentzian power response.  Magnitude-only data cannot distinguish
those two physical labels.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


@dataclass(frozen=True)
class FitResult:
    name: str
    offset_db: float
    bandwidth_3db_ghz: float
    bandwidth_standard_error_ghz: float
    order: float
    rmse_db: float
    r_squared: float


def parse_s2p(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return frequency in GHz and S21 magnitude in dB from an RI .s2p file."""
    rows: list[list[float]] = []
    parameter_line = ""
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("!"):
                continue
            if line.startswith("#"):
                parameter_line = line.upper()
                continue
            rows.append([float(value) for value in line.split()])

    if " S " not in f" {parameter_line} " or " RI " not in f" {parameter_line} ":
        raise ValueError(f"Expected Touchstone S-parameter data in RI format: {path}")

    data = np.asarray(rows, dtype=float)
    if data.ndim != 2 or data.shape[1] < 9:
        raise ValueError(f"Expected nine-column two-port Touchstone data: {path}")

    if " GHZ " in f" {parameter_line} ":
        frequency_ghz = data[:, 0]
    elif " MHZ " in f" {parameter_line} ":
        frequency_ghz = data[:, 0] / 1e3
    elif " KHZ " in f" {parameter_line} ":
        frequency_ghz = data[:, 0] / 1e6
    elif " HZ " in f" {parameter_line} ":
        frequency_ghz = data[:, 0] / 1e9
    else:
        raise ValueError(f"Unsupported or missing Touchstone frequency unit: {path}")

    s21_magnitude = np.hypot(data[:, 3], data[:, 4])
    s21_db = 20.0 * np.log10(np.clip(s21_magnitude, 1e-300, None))
    return frequency_ghz, s21_db


def through_corrected_response(
    dut_path: Path,
    through_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    frequency, dut_db = parse_s2p(dut_path)
    through_frequency, through_db = parse_s2p(through_path)
    response_db = dut_db - np.interp(frequency, through_frequency, through_db)
    return frequency, response_db


def normalized_response(
    dut_path: Path,
    through_path: Path,
    reference_window_ghz: tuple[float, float] = (0.5, 2.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Preserve band-mean referencing for the separate CMT analysis script."""
    frequency, response_db = through_corrected_response(dut_path, through_path)

    low, high = reference_window_ghz
    reference_mask = (frequency >= low) & (frequency <= high)
    if not np.any(reference_mask):
        raise ValueError("The reference window contains no measured samples")
    response_db -= np.mean(response_db[reference_mask])
    return frequency, response_db


def butterworth_db(
    frequency_ghz: np.ndarray,
    offset_db: float,
    bandwidth_3db_ghz: float,
    order: float | None,
) -> np.ndarray:
    """Butterworth magnitude in dB; bandwidth_3db_ghz is the -3.0103 dB point."""
    ratio = frequency_ghz / bandwidth_3db_ghz
    return offset_db - 10.0 * np.log10(1.0 + ratio ** (2.0 * order))


def direct_crossing(frequency_ghz: np.ndarray, response_db: np.ndarray) -> float:
    indices = np.flatnonzero(response_db <= -3.0)
    if not len(indices):
        return float("nan")
    upper = int(indices[0])
    if upper == 0:
        return float(frequency_ghz[0])
    lower = upper - 1
    fraction = (-3.0 - response_db[lower]) / (response_db[upper] - response_db[lower])
    return float(
        frequency_ghz[lower] + fraction * (frequency_ghz[upper] - frequency_ghz[lower])
    )


def metrics(
    name: str,
    response_db: np.ndarray,
    prediction_db: np.ndarray,
    offset_db: float,
    bandwidth_3db_ghz: float,
    bandwidth_standard_error_ghz: float,
    order: float,
) -> FitResult:
    residual = response_db - prediction_db
    rss = float(np.sum(residual**2))
    sample_count = response_db.size
    centered_rss = float(np.sum((response_db - np.mean(response_db)) ** 2))
    return FitResult(
        name=name,
        offset_db=float(offset_db),
        bandwidth_3db_ghz=float(bandwidth_3db_ghz),
        bandwidth_standard_error_ghz=float(bandwidth_standard_error_ghz),
        order=float(order),
        rmse_db=float(np.sqrt(rss / sample_count)),
        r_squared=float(1.0 - rss / centered_rss),
    )


def fit_fixed_order(
    frequency_ghz: np.ndarray, response_db: np.ndarray, order: int
) -> tuple[FitResult, np.ndarray]:
    model = lambda frequency, offset, bandwidth: butterworth_db(  # noqa: E731
        frequency, offset, bandwidth, order
    )
    initial_offset = float(np.mean(response_db[:30]))
    parameters, covariance = curve_fit(
        model,
        frequency_ghz,
        response_db,
        p0=(initial_offset, 6.4),
        bounds=((initial_offset - 20.0, 0.01), (initial_offset + 20.0, 100.0)),
    )
    prediction = model(frequency_ghz, *parameters)
    result = metrics(
        name=f"Butterworth order {order}",
        response_db=response_db,
        prediction_db=prediction,
        offset_db=parameters[0],
        bandwidth_3db_ghz=parameters[1],
        bandwidth_standard_error_ghz=np.sqrt(covariance[1, 1]),
        order=float(order),
    )
    return result, prediction


def fit_free_order(
    frequency_ghz: np.ndarray, response_db: np.ndarray
) -> tuple[FitResult, np.ndarray]:
    initial_offset = float(np.mean(response_db[:30]))
    parameters, covariance = curve_fit(
        butterworth_db,
        frequency_ghz,
        response_db,
        p0=(initial_offset, 6.4, 3.0),
        bounds=(
            (initial_offset - 20.0, 0.01, 0.1),
            (initial_offset + 20.0, 100.0, 10.0),
        ),
    )
    prediction = butterworth_db(frequency_ghz, *parameters)
    result = metrics(
        name="Butterworth free order",
        response_db=response_db,
        prediction_db=prediction,
        offset_db=parameters[0],
        bandwidth_3db_ghz=parameters[1],
        bandwidth_standard_error_ghz=np.sqrt(covariance[1, 1]),
        order=parameters[2],
    )
    return result, prediction


def save_plot(
    path: Path,
    frequency_ghz: np.ndarray,
    response_db: np.ndarray,
    predictions: dict[str, np.ndarray],
    best: FitResult,
    crossing_ghz: float,
) -> None:
    fig, axis = plt.subplots(figsize=(7.1, 4.1))
    axis.plot(
        frequency_ghz,
        response_db - best.offset_db,
        color="black",
        lw=1.1,
        label="Measurement",
    )
    axis.plot(
        frequency_ghz,
        predictions["One-pole comparison"] - best.offset_db,
        color="#777777",
        lw=1.0,
        ls=":",
        label="One-pole model (matched bandwidth)",
    )
    axis.plot(
        frequency_ghz,
        predictions[best.name] - best.offset_db,
        color="black",
        lw=1.2,
        ls="--",
        label=f"Best fixed-order fit: n={best.order:.0f}",
    )
    axis.axhline(-10.0 * np.log10(2.0), color="#aaaaaa", lw=0.8, ls="--")
    axis.axvline(best.bandwidth_3db_ghz, color="#aaaaaa", lw=0.8, ls=":")
    axis.set(
        xlim=(0.0, 10.0),
        ylim=(-13.0, 3.0),
        xlabel="Frequency (GHz)",
        ylabel="Normalized EO response (dB)",
    )
    axis.text(
        0.03,
        0.05,
        f"Direct crossing: {crossing_ghz:.2f} GHz\n"
        f"Best-fit bandwidth: {best.bandwidth_3db_ghz:.2f} GHz",
        transform=axis.transAxes,
        fontsize=9,
        va="bottom",
    )
    axis.legend(
        frameon=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor="none",
        fontsize=9,
        loc="upper right",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dut", type=Path, help="DUT Touchstone .s2p file")
    parser.add_argument("through", type=Path, help="Through-reference .s2p file")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("bandwidth_fit"),
        help="Output path without extension (default: bandwidth_fit)",
    )
    arguments = parser.parse_args()

    frequency_ghz, response_db = through_corrected_response(
        arguments.dut, arguments.through
    )

    fits: list[FitResult] = []
    predictions: dict[str, np.ndarray] = {}
    for order in range(1, 5):
        result, prediction = fit_fixed_order(frequency_ghz, response_db, order)
        fits.append(result)
        predictions[result.name] = prediction

    free_result, free_prediction = fit_free_order(frequency_ghz, response_db)
    fits.append(free_result)
    predictions[free_result.name] = free_prediction

    fixed_order_fits = [
        result for result in fits if result.name.startswith("Butterworth order")
    ]
    best = min(fixed_order_fits, key=lambda result: result.rmse_db)
    crossing_ghz = direct_crossing(frequency_ghz, response_db - best.offset_db)

    # For the plotted order comparison, hold the passband level and 3-dB
    # bandwidth at the best fixed-order values. This prevents the one-pole
    # curve from shifting its corner to imitate the measured roll-off, and
    # makes the difference in post-corner decay the visual comparison.
    predictions["One-pole comparison"] = butterworth_db(
        frequency_ghz,
        best.offset_db,
        best.bandwidth_3db_ghz,
        order=1.0,
    )

    output_prefix = arguments.output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    plot_path = output_prefix.with_suffix(".png")
    result_path = output_prefix.with_suffix(".json")
    save_plot(plot_path, frequency_ghz, response_db, predictions, best, crossing_ghz)
    payload = {
        "dut": str(arguments.dut.resolve()),
        "through": str(arguments.through.resolve()),
        "normalization": "Best fixed-order fit's low-frequency offset defines 0 dB",
        "direct_minus_3db_crossing_ghz": crossing_ghz,
        "best_fixed_order_model": best.name,
        "plotted_one_pole_comparison": {
            "description": "Order-one model using the best fixed-order model's offset and 3-dB bandwidth; not independently fitted. The same best-fit offset is subtracted from the measured and model curves for display.",
            "offset_db": best.offset_db,
            "bandwidth_3db_ghz": best.bandwidth_3db_ghz,
            "order": 1.0,
        },
        "fits": [asdict(result) for result in fits],
    }
    result_path.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"Direct -3 dB crossing: {crossing_ghz:.6f} GHz\n")
    print(f"{'Model':28s} {'f_3dB (GHz)':>12s} {'RMSE (dB)':>10s} {'R^2':>8s}")
    for result in sorted(fits, key=lambda item: item.rmse_db):
        print(
            f"{result.name:28s} {result.bandwidth_3db_ghz:12.6f} "
            f"{result.rmse_db:10.6f} {result.r_squared:8.5f}"
        )
    print(f"\nBest fixed-order model: {best.name}")
    print(f"Plot: {plot_path}")
    print(f"Vector plot: {plot_path.with_suffix('.pdf')}")
    print(f"Results: {result_path}")


if __name__ == "__main__":
    main()
