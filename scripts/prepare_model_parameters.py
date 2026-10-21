"""Derive supplementary model inputs from the downloaded Fig. 4 fit and capture."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tfln_paths import DATA_ROOT, data_path


def build_parameters(fit: dict, reconstruction: dict, strobe: dict) -> dict:
    permutation = np.asarray(fit["applied_voltage_perm"], dtype=int)
    arms = [f"K{index + 1}" for index in permutation]
    fields = {
        "delta_at_baseline_pm": "delta_dc_pm",
        "external_to_internal_coupling": "x_arr",
        "tuning_efficiency_hz_per_v": "eta_arr",
        "loaded_q": "q_arr",
    }
    cavities = {
        arm: {name: float(fit[key][index]) for name, key in fields.items()}
        for index, arm in enumerate(arms)
    }
    states = []
    for index, color in enumerate(("red", "green", "blue"), 1):
        voltages = np.asarray(strobe[f"strobe_{color}_abs_v"])
        states.append(
            {
                "label": f"Route to P{index}",
                "source_state": color,
                "voltage_v": {
                    f"K{j+1}": float(np.round(value, 8))
                    for j, value in enumerate(voltages)
                },
            }
        )
    return {
        "description": "Reciprocal model derived from the archived Fig. 4 fit.",
        "provenance": {
            "record_doi": "10.5281/zenodo.23194529",
            "derived_from": [
                "fig4/voltage_cube/inputs/full_cube_fit_reciprocal_MT.npz",
                "fig4/voltage_cube/inputs/reconstructed_exp_cube.npz",
                "fig4/voltage_cube/inputs/uniform_cube_16_strobe_20260505_230858.npz",
            ],
        },
        "laser_wavelength_nm": float(reconstruction["laser_wavelength_nm"]),
        "baseline_voltage_v": {
            f"K{i+1}": float(value)
            for i, value in enumerate(reconstruction["dc_volts"])
        },
        "cavities": {name: cavities[name] for name in ("K1", "K2", "K3")},
        "switching_states": states,
        "recombination_model": {
            "model_variant": str(fit["model_variant"]),
            "arm_order": arms,
            "mmi_amplitude_parameters": np.asarray(fit["amp_rel"]).tolist(),
            "one_way_arm_phase_rad": [0.0, *np.asarray(fit["phi_rel"]).tolist()],
            "voltage_phase_slope_rad_per_v": float(fit["phase_v_slope"]),
            "coherent_background_amplitude": np.asarray(fit["bg_amp"]).tolist(),
            "coherent_background_phase_rad": [
                0.0,
                *np.asarray(fit["bg_phase_rel"]).tolist(),
            ],
            "input_port_index_zero_based": int(fit["input_port"]) - 1,
            "source_detector_port_order_zero_based": (
                np.asarray(fit["detector_port_order"]) - 1
            ).tolist(),
            "figure_channel_source_detector_indices": np.argsort(
                fit["applied_detector_perm"]
            ).tolist(),
            "detector_gain_offset": np.asarray(fit["cal"]).tolist(),
        },
    }


def main() -> None:
    def load(name):
        with np.load(
            data_path(f"fig4/voltage_cube/inputs/{name}"), allow_pickle=True
        ) as source:
            return {key: source[key] for key in source.files}

    result = build_parameters(
        load("full_cube_fit_reciprocal_MT.npz"),
        load("reconstructed_exp_cube.npz"),
        load("uniform_cube_16_strobe_20260505_230858.npz"),
    )
    output = DATA_ROOT / "derived/model_parameters.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
