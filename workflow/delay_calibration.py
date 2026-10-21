"""Physically constrained offline cable-delay calibration.

Fits tau_ij = d_in_i + d_out_j with d_in_1 = 0. Bounds match the
original numerical implementation: detector delays 5–150 ns and relative
input delays within 80 ns. Requires a saved multi-drive calibration capture.
"""

import argparse
import json
from pathlib import Path
import numpy as np
from scipy.signal import hilbert, butter, filtfilt
from scipy.interpolate import interp1d
from scipy.optimize import differential_evolution

D_OUT_LO = 5e-09
D_OUT_HI = 1.5e-07
D_IN_BOUND = 8e-08
INTERP_LO = -1e-08
INTERP_HI = 2e-07
DE_POPSIZE = 20
DE_MAXITER = 2000
DE_TOL = 1e-11


def load_envelopes(npz_path):
    """
    Compute 9 bandpass cross-correlation envelopes from an NPZ file.
    Stores each as an interpolator over [INTERP_LO, INTERP_HI].
    Returns  envs[i][j] (callable),  lags_w,  dt.
    """
    d = np.load(npz_path, allow_pickle=True)
    dt = float(d["drive1_scope1_time"][1] - d["drive1_scope1_time"][0])
    fs = 1.0 / dt
    N = len(d["drive1_scope1_time"])
    T_S = float(d["awg_period_s"])
    f_lo = (np.array(d["tone_bins"].tolist()) / T_S).min() * 0.7
    f_hi = (np.array(d["tone_bins"].tolist()) / T_S).max() * 1.3

    def bp(v):
        b, a = butter(4, [f_lo / (fs / 2), f_hi / (fs / 2)], btype="band")
        return filtfilt(b, a, v - v.mean())

    lags = (np.arange(N) - N // 2) * dt
    center = N // 2
    lo_idx = center + int(INTERP_LO / dt)
    hi_idx = center + int(INTERP_HI / dt) + 1
    lags_w = lags[lo_idx:hi_idx]
    envs = []
    for dc in [1, 2, 3]:
        row = []
        xr = bp(d[f"drive{dc}_scope1_voltage"].astype(float))
        for sc in [2, 3, 4]:
            xd = bp(d[f"drive{dc}_scope{sc}_voltage"].astype(float))
            c = np.fft.irfft(np.fft.rfft(xr).conj() * np.fft.rfft(xd), n=N)
            c = np.fft.fftshift(c)
            env = np.abs(hilbert(c))[lo_idx:hi_idx]
            env = env / (env.max() + 1e-30)
            row.append(
                interp1d(lags_w, env, kind="linear", bounds_error=False, fill_value=0.0)
            )
        envs.append(row)
    return (envs, lags_w, dt)


def build_cost_single(envs):
    """5-parameter cost function for a single file."""

    def cost(x):
        d_in = np.array([0.0, x[0], x[1]])
        d_out = np.array([x[2], x[3], x[4]])
        total = 0.0
        for i in range(3):
            for j in range(3):
                total += envs[i][j](d_in[i] + d_out[j])
        return -total

    return cost


def fit_single(envs):
    """Run differential evolution with physical constraints."""
    bounds = [(-D_IN_BOUND, D_IN_BOUND)] * 2 + [(D_OUT_LO, D_OUT_HI)] * 3
    result = differential_evolution(
        build_cost_single(envs),
        bounds,
        maxiter=DE_MAXITER,
        tol=DE_TOL,
        popsize=DE_POPSIZE,
        seed=42,
        polish=True,
        disp=False,
    )
    x = result.x
    d_in = np.array([0.0, x[0], x[1]])
    d_out = np.array([x[2], x[3], x[4]])
    score = -result.fun
    return (d_in, d_out, score, result.success)


def peak_ratios(envs, lags_w, d_in, d_out):
    """
    For each pair, compute:
      predicted_tau, envelope_at_predicted, global_max_env, ratio
    Returns a (3,3) dict of these four quantities.
    """
    results = {}
    for i in range(3):
        for j in range(3):
            tau_pred = d_in[i] + d_out[j]
            ev_arr = np.array([envs[i][j](t) for t in lags_w])
            global_max_idx = int(np.argmax(ev_arr))
            global_max_tau = lags_w[global_max_idx]
            global_max_val = ev_arr[global_max_idx]
            pred_val = float(envs[i][j](tau_pred))
            results[i, j] = dict(
                tau_pred=tau_pred,
                pred_val=pred_val,
                global_max_tau=global_max_tau,
                global_max_val=global_max_val,
                ratio=pred_val / (global_max_val + 1e-30),
            )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if "INSERT" in str(args.capture).upper():
        parser.error("Replace INSERT PATH HERE with the calibration capture path.")
    envs, lags, dt = load_envelopes(args.capture)
    d_in, d_out, score, success = fit_single(envs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "d_in_s": d_in.tolist(),
                "d_out_s": d_out.tolist(),
                "score": float(score),
                "success": bool(success),
                "sample_interval_s": float(dt),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
