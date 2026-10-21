"""Offline waveform generation, cube reconstruction, fitting and evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import model_v1 as model
import routing
import waveforms


def checked_path(value: str, *, exists: bool = True) -> Path:
    if "INSERT PATH HERE" in value.upper() or "INSERT_PATH_HERE" in value.upper():
        raise ValueError("Replace the INSERT PATH HERE placeholder with your path.")
    path = Path(value).expanduser().resolve()
    if exists and not path.is_file():
        raise FileNotFoundError(path)
    return path


def read_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as source:
        return {key: source[key] for key in source.files}


def evaluate_saved_fit(
    fit: dict, reconstruction: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the saved model in its archived internal axis/channel order."""
    mesh = np.meshgrid(*(fit[key] for key in ("c1", "c2", "c3")), indexing="ij")
    permutation = np.asarray(fit.get("applied_voltage_perm", [0, 1, 2]), dtype=int)
    bias = np.asarray(reconstruction["dc_volts"])[permutation]
    names = (
        "delta_dc_pm",
        "x_arr",
        "eta_arr",
        "q_arr",
        "theta_rel",
        "amp_rel",
        "phi_rel",
        "phase_v_slope",
        "bg_amp",
        "bg_phase_rel",
    )
    power = model.model_power(
        *(axis.ravel() for axis in mesh),
        float(reconstruction["laser_wavelength_nm"]),
        bias,
        *(fit[key] for key in names),
    )
    calibration = np.asarray(fit["cal"])
    signal = power * calibration[:, 0] + calibration[:, 1]
    shape = tuple(len(fit[key]) for key in ("c1", "c2", "c3")) + (3,)
    return power.reshape(shape), signal.reshape(shape)


def fit_capture(args, output: Path) -> dict:
    """Use the original permutation-aware fit with explicit initialization."""
    import run_saved_result_with_3d_v2 as refinement
    import run_saved_result_with_3d_v4 as fitting

    fitting.UNIFORM_OUTDIR_NAME = str(output)
    fitting._configure_runners()
    refinement.INITIAL_FIT_FILES = [
        str(checked_path(value)) for value in args.initial_fit
    ]
    for module in (refinement, fitting):
        module.UNIFORM_OUTDIR_NAME = str(output)

    if args.raw:
        return fitting._fit_uniform_cube_capture_v4(str(checked_path(args.raw)))

    source = checked_path(args.reconstructed)
    payload = read_npz(source)
    original_loader = model.reconstruct_exp_cube_from_raw

    def from_reconstruction(raw_capture_path, awg_sr, cube_bins, cube_out_path):
        model.save_reconstructed_cube(payload, cube_out_path)
        data = model.cube_payload_to_data(
            payload,
            str(source),
            str(cube_out_path),
            awg_sr,
            cube_bins,
        )
        metadata = {
            "awg_period_s": args.period_s,
            "f_lo_hz": float(payload["f_lo_hz"]),
            "f_hi_hz": float(payload["f_hi_hz"]),
            "seed": int(payload["seed"]),
        }
        return data, metadata

    model.reconstruct_exp_cube_from_raw = from_reconstruction
    try:
        return fitting._fit_uniform_cube_capture_v4(str(source))
    finally:
        model.reconstruct_exp_cube_from_raw = original_loader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    multisine = commands.add_parser("multisine", help="Generate the cube drive arrays.")
    multisine.add_argument("--seed", type=int, default=42)
    multisine.add_argument("--period-s", type=float, default=25e-6)
    multisine.add_argument("--sample-rate-hz", type=float, default=2.4e9)
    multisine.add_argument("--f-low-hz", type=float, default=10e6)
    multisine.add_argument("--f-high-hz", type=float, default=50e6)
    multisine.add_argument("--output", required=True)
    reconstruct = commands.add_parser(
        "reconstruct", help="Reconstruct a raw cube capture."
    )
    reconstruct.add_argument("--raw", required=True)
    reconstruct.add_argument("--output", required=True)
    targets = commands.add_parser("targets", help="Select measured routing states.")
    targets.add_argument("--reconstructed", required=True)
    targets.add_argument("--output", required=True)
    strobe = commands.add_parser(
        "strobe", help="Rebuild the saved programmed waveform."
    )
    strobe.add_argument("--capture", required=True)
    strobe.add_argument("--output", required=True)
    evaluation = commands.add_parser(
        "evaluate", help="Evaluate saved fitted parameters."
    )
    evaluation.add_argument("--fit", required=True)
    evaluation.add_argument("--reconstructed", required=True)
    evaluation.add_argument("--output", required=True)
    fit = commands.add_parser("fit", help="Run the original nonlinear fitting stages.")
    source = fit.add_mutually_exclusive_group(required=True)
    source.add_argument("--raw")
    source.add_argument("--reconstructed")
    fit.add_argument("--initial-fit", action="append", default=[])
    fit.add_argument("--period-s", type=float, default=25e-6)
    fit.add_argument("--output", required=True)
    args = parser.parse_args()
    output = checked_path(args.output, exists=False)
    if output.exists():
        raise FileExistsError(f"Choose a new output path: {output}")
    if (
        args.command in {"multisine", "reconstruct", "strobe", "evaluate"}
        and output.suffix.lower() != ".npz"
    ):
        parser.error("Choose an output filename ending in .npz.")
    if args.command == "fit":
        output.mkdir(parents=True)
        result = fit_capture(args, output)
        print(f"Saved fit: {result['full_cube_fit_path']}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "multisine":
        samples_float = args.period_s * args.sample_rate_hz
        samples = round(samples_float)
        if not np.isclose(samples_float, samples, atol=1e-9, rtol=0):
            raise ValueError("The period must contain an integer number of samples.")
        values, _ = waveforms.build_cube_trajectory(
            n_samples=samples,
            period_s=args.period_s,
            sample_rate_hz=args.sample_rate_hz,
            f_lo_hz=args.f_low_hz,
            f_hi_hz=args.f_high_hz,
            seed=args.seed,
        )
        np.savez_compressed(
            output,
            waveforms_normalized=values,
            seed=args.seed,
            period_s=args.period_s,
            sample_rate_hz=args.sample_rate_hz,
            f_lo_hz=args.f_low_hz,
            f_hi_hz=args.f_high_hz,
        )
    elif args.command == "reconstruct":
        model.reconstruct_exp_cube_from_raw(
            str(checked_path(args.raw)),
            model.AWG_SR_DEFAULT,
            model.FIT_CUBE_BINS,
            str(output),
        )
    elif args.command == "targets":
        selected = routing.select_measured_routing_points(
            read_npz(checked_path(args.reconstructed))
        )
        portable = {
            name: {
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in info.items()
            }
            for name, info in selected.items()
        }
        output.write_text(json.dumps(portable, indent=2) + "\n")
    elif args.command == "strobe":
        capture = read_npz(checked_path(args.capture))
        selected = {
            name: {"awg_v": capture[f"strobe_{name}_awg_v"]}
            for name in ("red", "green", "blue")
        }
        values, states, names = waveforms.build_strobe_waveform(
            selected,
            n_idle_slots=int(capture["strobe_n_idle_slots"]),
            cycle_freq_hz=1e9 / float(capture["strobe_t_cycle_ns"]),
        )
        np.savez_compressed(
            output,
            waveforms_normalized=values,
            states_awg_v=np.asarray(states),
            state_names=names,
            sample_rate_hz=waveforms.AWG_SR,
        )
    elif args.command == "evaluate":
        power, signal = evaluate_saved_fit(
            read_npz(checked_path(args.fit)),
            read_npz(checked_path(args.reconstructed)),
        )
        np.savez_compressed(output, p_cube=power, t_cube=signal)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
