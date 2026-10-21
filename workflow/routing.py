"""Measured-extinction routing-state selection.

P1/P2/P3 correspond to CH2/CH3/CH4 and red/green/blue. The returned
voltages are selected from measured samples using the original calculation."""

from __future__ import annotations
import numpy as np


def select_measured_routing_points(reconstructed, eps=1e-06):
    k_biases = np.asarray(reconstructed["dc_volts"], dtype=float)
    v1_in, v2_in, v3_in = [
        np.asarray(reconstructed[f"V{i}_raw"]) - k_biases[i - 1] for i in (1, 2, 3)
    ]
    out1, out2, out3 = [np.asarray(reconstructed[f"out{i}_raw"]) for i in (1, 2, 3)]

    def norm01(x):
        lo, hi = (float(x.min()), float(x.max()))
        return (x - lo) / (hi - lo) if hi > lo else np.full_like(x, 0.5)

    r, g, b = (norm01(out1), norm01(out2), norm01(out3))
    lr = 10.0 * np.log10(r + eps)
    lg = 10.0 * np.log10(g + eps)
    lb = 10.0 * np.log10(b + eps)
    Q_R = lr - np.maximum(lg, lb)
    Q_G = lg - np.maximum(lr, lb)
    Q_B = lb - np.maximum(lr, lg)
    results = {}
    for name, Q_arr in [("red", Q_R), ("green", Q_G), ("blue", Q_B)]:
        idx = int(np.argmax(Q_arr))
        awg_v = np.array([v1_in[idx], v2_in[idx], v3_in[idx]])
        abs_v = awg_v + k_biases
        results[name] = {
            "awg_v": awg_v,
            "abs_v": abs_v,
            "Q_dB": float(Q_arr[idx]),
            "outputs": (float(out1[idx]), float(out2[idx]), float(out3[idx])),
            "index": idx,
        }
    return results
