"""Regression checks for saved-fit ordering and permutation-aware fallback."""

from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workflow"))
import model_v1 as model
import run_saved_result_with_3d_v2 as refinement
import run_saved_result_with_3d_v4 as fitting
import run_workflow


def synthetic_cube(directory):
    edges = [np.linspace(bias - 2, bias + 2, 3) for bias in (-17, 19, -16)]
    centers = [(edge[:-1] + edge[1:]) / 2 for edge in edges]
    mesh = np.meshgrid(*centers, indexing="ij")
    data = {
        "cube_path": str(directory / "reconstructed.npz"),
        "raw_capture_path": str(directory / "capture.npz"),
        "dc_volts": np.array([-17.0, 19.0, -16.0]),
        "laser_nm": 1541.66,
        "count": np.ones((2, 2, 2)),
        "mask": np.ones((2, 2, 2), dtype=bool),
        "weights": np.ones(8),
        "awg_sr": 2.4e9,
        "cube_bins": 2,
        "awg_output_range_v": 5.0,
    }
    for i in range(3):
        data[f"e{i+1}"] = edges[i]
        data[f"c{i+1}"] = centers[i]
        data[f"v{i+1}"] = mesh[i].ravel()
    guess = fitting._default_free_q_guess()
    delta, x, eta, theta, amp, phi, slope, bg, bg_phi, q = model.unpack_params(
        guess, True
    )
    powers = model.model_power(
        *(axis.ravel() for axis in mesh),
        data["laser_nm"],
        data["dc_volts"],
        delta,
        x,
        eta,
        q,
        theta,
        amp,
        phi,
        slope,
        bg,
        bg_phi,
    )
    data["t_exp"] = powers * np.array([1.2, 0.8, 1.5]) + np.array([0.02, -0.01, 0.03])
    data["cubes"] = [data["t_exp"][:, i].reshape(2, 2, 2) for i in range(3)]
    return data, guess


class FittingTests(unittest.TestCase):
    def test_exported_permuted_fit_can_be_evaluated_again(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            original, guess = synthetic_cube(directory)
            data = fitting._apply_full_perm(original, [2, 1, 0], [0, 2, 1])
            result = model.evaluate_solution(guess, data, 0.015, True)
            model.save_params(result, data, temporary)
            model.build_full_cube(result, data, temporary)
            for name in ("fit_params.npz", "full_cube_fit.npz"):
                with np.load(directory / name, allow_pickle=True) as saved:
                    np.testing.assert_array_equal(
                        saved["applied_voltage_perm"], [2, 1, 0]
                    )
                    np.testing.assert_array_equal(
                        saved["applied_detector_perm"], [0, 2, 1]
                    )
                    self.assertEqual(
                        saved["detector_labels"].tolist(),
                        ["P1 / CH2", "P3 / CH4", "P2 / CH3"],
                    )
            fit = run_workflow.read_npz(directory / "full_cube_fit.npz")
            power, signal = run_workflow.evaluate_saved_fit(
                fit,
                {
                    "dc_volts": original["dc_volts"],
                    "laser_wavelength_nm": original["laser_nm"],
                },
            )
            np.testing.assert_array_equal(power, fit["p_cube"])
            np.testing.assert_array_equal(signal, fit["t_cube"])
            self.assertEqual(model.arm_labels(data), ["K3", "K2", "K1"])

    def test_global_fallback_keeps_prepared_permuted_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            original, guess = synthetic_cube(directory)
            capture = directory / "capture.npz"
            capture.touch()
            data = fitting._apply_full_perm(original, [2, 1, 0], [0, 2, 1])
            with patch.object(
                model,
                "reconstruct_exp_cube_from_raw",
                side_effect=AssertionError("Must retain prepared ordering"),
            ), patch.object(
                model,
                "differential_evolution",
                return_value=SimpleNamespace(x=guess[:-3], fun=0.0),
            ), patch.object(
                model, "minimize", return_value=SimpleNamespace(x=guess)
            ):
                result = model.run_fit_from_raw_capture(
                    str(capture), temporary, prepared=(data, {}), restarts=1
                )
            self.assertIs(result["data"], data)
            fit = run_workflow.read_npz(Path(result["full_cube_fit_path"]))
            power, signal = run_workflow.evaluate_saved_fit(
                fit,
                {
                    "dc_volts": original["dc_volts"],
                    "laser_wavelength_nm": original["laser_nm"],
                },
            )
            np.testing.assert_array_equal(power, fit["p_cube"])
            np.testing.assert_array_equal(signal, fit["t_cube"])

    def test_permutation_runner_passes_ordering_to_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            original, guess = synthetic_cube(Path(temporary))
            original["cube_path"] = str(Path(temporary) / "reconstructed_exp_cube.npz")
            evaluate = refinement._evaluate_start(original, guess, 0.015)
            with patch.object(fitting, "UNIFORM_OUTDIR_NAME", temporary), patch.object(
                model, "reconstruct_exp_cube_from_raw", return_value=(original, {})
            ), patch.object(
                fitting,
                "_select_best_permutation",
                return_value=([2, 1, 0], [0, 2, 1], guess),
            ), patch.object(
                refinement, "_discover_recent_fit_candidates", return_value=[]
            ), patch.object(
                refinement, "_polish_guess", return_value=(evaluate, guess, 1, 0.0)
            ), patch.object(
                refinement, "_fast_path_is_weak", return_value=True
            ), patch.object(
                model, "run_fit_from_raw_capture", return_value={"result": evaluate}
            ) as fallback:
                fitting._fit_uniform_cube_capture_v4(
                    str(Path(temporary) / "capture.npz")
                )
            prepared, _ = fallback.call_args.kwargs["prepared"]
            np.testing.assert_array_equal(
                prepared["dc_volts"], original["dc_volts"][[2, 1, 0]]
            )
            np.testing.assert_array_equal(
                prepared["t_exp"], original["t_exp"][:, [0, 2, 1]]
            )


if __name__ == "__main__":
    unittest.main()
