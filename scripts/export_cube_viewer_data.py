#!/usr/bin/env python3
"""Export the Fig. 4 voltage cubes for the browser viewer.

The arrays are stored as little-endian float64 byte strings inside gzipped
JSON. This preserves the numerical values exactly while remaining portable
and considerably smaller than textual JSON arrays.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from tfln_paths import ROOT, DATA_ROOT

import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
INPUTS = DATA_ROOT / "fig4/voltage_cube/inputs"
DEFAULT_OUTPUT = ROOT / "viewer/data/cube_data.json.gz"


def encoded(array: np.ndarray) -> dict[str, object]:
    values = np.ascontiguousarray(np.asarray(array, dtype="<f8").ravel())
    return {
        "dtype": "<f8",
        "length": int(values.size),
        "base64": base64.b64encode(values.tobytes()).decode("ascii"),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_payload() -> dict[str, object]:
    rec_path = INPUTS / "reconstructed_exp_cube.npz"
    fit_path = INPUTS / "full_cube_fit_reciprocal_MT.npz"
    capture_path = INPUTS / "uniform_cube_16_strobe_20260505_230858.npz"

    rec = np.load(rec_path, allow_pickle=True)
    fit = np.load(fit_path, allow_pickle=True)
    capture = np.load(capture_path, allow_pickle=True)

    c1, c2, c3 = fit["c1"], fit["c2"], fit["c3"]
    grid1, grid2, grid3 = np.meshgrid(c1, c2, c3, indexing="ij")
    transmission = fit["t_cube"]

    experiment = {
        "v1": encoded(rec["V1_raw"]),
        "v2": encoded(rec["V2_raw"]),
        "v3": encoded(rec["V3_raw"]),
        "out1": encoded(rec["out1_raw"]),
        "out2": encoded(rec["out2_raw"]),
        "out3": encoded(rec["out3_raw"]),
    }
    model = {
        # Reciprocal M^T fit: c1=K3, c2=K2, c3=K1.
        "v1": encoded(grid3.ravel()),
        "v2": encoded(grid2.ravel()),
        "v3": encoded(grid1.ravel()),
        # Detector order used by the displayed output channels.
        "out1": encoded(transmission[..., 0].ravel()),
        "out2": encoded(transmission[..., 2].ravel()),
        "out3": encoded(transmission[..., 1].ravel()),
    }

    return {
        "schema_version": 1,
        "title": "TFLN photon router voltage-cube data",
        "description": (
            "Exact float64 coordinates and output values used for the "
            "interactive views of main-text Fig. 4a-d."
        ),
        "channel_mapping": {
            "out1": "P1 / CH2 / red",
            "out2": "P2 / CH3 / green",
            "out3": "P3 / CH4 / blue",
        },
        "source_files": [
            {"path": str(rec_path.relative_to(DATA_ROOT)), "sha256": sha256(rec_path)},
            {"path": str(fit_path.relative_to(DATA_ROOT)), "sha256": sha256(fit_path)},
            {
                "path": str(capture_path.relative_to(DATA_ROOT)),
                "sha256": sha256(capture_path),
            },
        ],
        "experiment": experiment,
        "model": model,
        "routing_states": {
            "S1": encoded(capture["strobe_red_abs_v"]),
            "S2": encoded(capture["strobe_green_abs_v"]),
            "S3": encoded(capture["strobe_blue_abs_v"]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = build_payload()
    with gzip.open(output, "wt", encoding="utf-8", compresslevel=9) as stream:
        json.dump(payload, stream, separators=(",", ":"), allow_nan=False)
    print(f"Wrote {output} ({output.stat().st_size / 1024**2:.2f} MiB)")


if __name__ == "__main__":
    main()
