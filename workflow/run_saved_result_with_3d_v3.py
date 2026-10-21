"""Model-based operating-point and balance optimization."""

from __future__ import annotations
import os
import numpy as np
import model_v1

BIAS_SAFE_FRACTION = 0.7
CHANNEL_OPTIMIZATION_STARTS = 6
CHANNEL_OPTIMIZATION_MAXITER = 140
BALANCE_OPTIMIZATION_MAXITER = 180
BALANCE_EPS = 0.05
OPERATING_REPORT_NAME = "operating_point_report.txt"
_C_MPS = 299792458.0


def _scan_bounds(data: dict) -> list[tuple[float, float]]:
    return [
        (
            float(np.asarray(data["e1"], dtype=np.float64)[0]),
            float(np.asarray(data["e1"], dtype=np.float64)[-1]),
        ),
        (
            float(np.asarray(data["e2"], dtype=np.float64)[0]),
            float(np.asarray(data["e2"], dtype=np.float64)[-1]),
        ),
        (
            float(np.asarray(data["e3"], dtype=np.float64)[0]),
            float(np.asarray(data["e3"], dtype=np.float64)[-1]),
        ),
    ]


def _predict_at_point(
    result: dict, data: dict, point: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    point = np.asarray(point, dtype=np.float64).reshape(3)
    p_sim = model_v1.model_power(
        np.array([point[0]], dtype=np.float64),
        np.array([point[1]], dtype=np.float64),
        np.array([point[2]], dtype=np.float64),
        data["laser_nm"],
        data["dc_volts"],
        result["delta_dc_pm"],
        result["x_arr"],
        result["eta_arr"],
        result["q_arr"],
        result["theta_rel"],
        result["amp_rel"],
        result["phi_rel"],
        result["phase_v_slope"],
        result["bg_amp"],
        result["bg_phase_rel"],
    )[0]
    t_sim = result["cal"][:, 0] * p_sim + result["cal"][:, 1]
    return (p_sim, t_sim)


def _points_within_scan_flags(
    point: np.ndarray, bounds: list[tuple[float, float]]
) -> list[bool]:
    out = []
    for idx, (lo, hi) in enumerate(bounds):
        out.append(lo <= float(point[idx]) <= hi)
    return out


def _pm_per_volt(laser_nm: float, eta_hz_v: float) -> float:
    laser_m = float(laser_nm) * 1e-09
    return -(laser_m * laser_m / _C_MPS) * float(eta_hz_v) * 1000000000000.0


def _recommend_bias_points(result: dict, data: dict) -> dict:
    dc_volts = np.asarray(data["dc_volts"], dtype=np.float64)
    delta_dc_pm = np.asarray(result["delta_dc_pm"], dtype=np.float64)
    eta_arr = np.asarray(result["eta_arr"], dtype=np.float64)
    bounds = _scan_bounds(data)
    per_cavity = []
    full_target = np.zeros(3, dtype=np.float64)
    safe_target = np.zeros(3, dtype=np.float64)
    for idx, name in enumerate(model_v1.arm_labels(data)):
        slope_pm_per_v = _pm_per_volt(data["laser_nm"], eta_arr[idx])
        delta_v = (
            -delta_dc_pm[idx] / slope_pm_per_v if abs(slope_pm_per_v) > 1e-12 else 0.0
        )
        full_target[idx] = dc_volts[idx] + delta_v
        safe_target[idx] = dc_volts[idx] + BIAS_SAFE_FRACTION * delta_v
        per_cavity.append(
            {
                "name": name,
                "current_v": float(dc_volts[idx]),
                "delta_dc_pm": float(delta_dc_pm[idx]),
                "eta_hz_v": float(eta_arr[idx]),
                "slope_pm_per_v": float(slope_pm_per_v),
                "delta_v": float(delta_v),
                "full_target_v": float(full_target[idx]),
                "safe_target_v": float(safe_target[idx]),
                "scan_lo": float(bounds[idx][0]),
                "scan_hi": float(bounds[idx][1]),
            }
        )
    _, t_full = _predict_at_point(result, data, full_target)
    _, t_safe = _predict_at_point(result, data, safe_target)
    return {
        "per_cavity": per_cavity,
        "full_target": full_target,
        "safe_target": safe_target,
        "full_target_in_scan": _points_within_scan_flags(full_target, bounds),
        "safe_target_in_scan": _points_within_scan_flags(safe_target, bounds),
        "pred_t_full": t_full,
        "pred_t_safe": t_safe,
    }


def _top_grid_start_points(
    t_cube: np.ndarray, c1: np.ndarray, c2: np.ndarray, c3: np.ndarray, port: int
) -> list[np.ndarray]:
    flat = np.asarray(t_cube[..., port], dtype=np.float64).ravel()
    n_keep = min(CHANNEL_OPTIMIZATION_STARTS, flat.size)
    if n_keep <= 0:
        return []
    top_idx = np.argpartition(flat, -n_keep)[-n_keep:]
    top_idx = top_idx[np.argsort(flat[top_idx])[::-1]]
    points = []
    for idx in top_idx:
        i, j, k = np.unravel_index(int(idx), t_cube.shape[:3])
        points.append(np.array([c1[i], c2[j], c3[k]], dtype=np.float64))
    return points


def _optimize_detector_maximum(
    result: dict,
    data: dict,
    t_cube: np.ndarray,
    c1: np.ndarray,
    c2: np.ndarray,
    c3: np.ndarray,
    port: int,
) -> dict:
    bounds = _scan_bounds(data)
    seeds = _top_grid_start_points(t_cube, c1, c2, c3, port)
    seeds.append(np.asarray(data["dc_volts"], dtype=np.float64))
    seen = set()
    best = None
    for seed in seeds:
        seed_key = tuple(np.round(np.asarray(seed, dtype=np.float64), 6))
        if seed_key in seen:
            continue
        seen.add(seed_key)

        def objective(v_arr: np.ndarray) -> float:
            _, t_sim = _predict_at_point(result, data, v_arr)
            return -float(t_sim[port])

        res = model_v1.minimize(
            objective,
            np.asarray(seed, dtype=np.float64),
            method="L-BFGS-B",
            bounds=bounds,
            options=dict(maxiter=CHANNEL_OPTIMIZATION_MAXITER, ftol=1e-12, gtol=1e-09),
        )
        point = np.asarray(res.x, dtype=np.float64)
        p_sim, t_sim = _predict_at_point(result, data, point)
        score = float(t_sim[port])
        if best is None or score > best["detector_value"]:
            best = {
                "point": point,
                "detector_value": score,
                "all_t": np.asarray(t_sim, dtype=np.float64),
                "all_p": np.asarray(p_sim, dtype=np.float64),
            }
    return best


def _balanced_score(
    t_vals: np.ndarray, t_lo: np.ndarray, t_hi: np.ndarray
) -> tuple[float, np.ndarray]:
    denom = np.maximum(t_hi - t_lo, 1e-12)
    norm = np.clip((np.asarray(t_vals, dtype=np.float64) - t_lo) / denom, 0.0, 1.0)
    return (float(np.sum(np.log(norm + BALANCE_EPS))), norm)


def _optimize_balanced_point(
    result: dict,
    data: dict,
    t_cube: np.ndarray,
    c1: np.ndarray,
    c2: np.ndarray,
    c3: np.ndarray,
) -> dict:
    bounds = _scan_bounds(data)
    t_lo = np.asarray(np.nanmin(t_cube, axis=(0, 1, 2)), dtype=np.float64)
    t_hi = np.asarray(np.nanmax(t_cube, axis=(0, 1, 2)), dtype=np.float64)
    grid_norm = np.clip((t_cube - t_lo) / np.maximum(t_hi - t_lo, 1e-12), 0.0, 1.0)
    grid_primary = np.min(grid_norm, axis=3)
    grid_secondary = np.mean(grid_norm, axis=3)
    flat_score = grid_primary.ravel() + 0.001 * grid_secondary.ravel()
    best_idx = int(np.argmax(flat_score))
    i0, j0, k0 = np.unravel_index(best_idx, grid_primary.shape)
    x0 = np.array([c1[i0], c2[j0], c3[k0]], dtype=np.float64)

    def objective(v_arr: np.ndarray) -> float:
        _, t_sim = _predict_at_point(result, data, v_arr)
        score, _ = _balanced_score(t_sim, t_lo=t_lo, t_hi=t_hi)
        return -score

    res = model_v1.minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options=dict(maxiter=BALANCE_OPTIMIZATION_MAXITER, ftol=1e-12, gtol=1e-09),
    )
    point = np.asarray(res.x, dtype=np.float64)
    p_sim, t_sim = _predict_at_point(result, data, point)
    score, norm = _balanced_score(t_sim, t_lo=t_lo, t_hi=t_hi)
    return {
        "point": point,
        "all_t": np.asarray(t_sim, dtype=np.float64),
        "all_p": np.asarray(p_sim, dtype=np.float64),
        "score": score,
        "norm_t": norm,
    }


def _format_point(point: np.ndarray) -> str:
    point = np.asarray(point, dtype=np.float64)
    return f"axis1={point[0]:+.3f} V  axis2={point[1]:+.3f} V  axis3={point[2]:+.3f} V"


def _format_triplet(values: np.ndarray) -> str:
    values = np.asarray(values, dtype=np.float64)
    return f"[{values[0]:.4f}, {values[1]:.4f}, {values[2]:.4f}]"


def _build_operating_report(outputs: dict) -> str:
    result = outputs["result"]
    data = outputs["data"]
    fit_cube_path = outputs["full_cube_fit_path"]
    with np.load(fit_cube_path, allow_pickle=True) as fit_cube:
        t_cube = np.asarray(fit_cube["t_cube"], dtype=np.float64)
        c1 = np.asarray(fit_cube["c1"], dtype=np.float64)
        c2 = np.asarray(fit_cube["c2"], dtype=np.float64)
        c3 = np.asarray(fit_cube["c3"], dtype=np.float64)
        detector_labels = model_v1.detector_labels(data)
    bias = _recommend_bias_points(result, data)
    per_channel = []
    for port, name in enumerate(detector_labels):
        optimum = _optimize_detector_maximum(result, data, t_cube, c1, c2, c3, port)
        per_channel.append((name, optimum))
    balanced = _optimize_balanced_point(result, data, t_cube, c1, c2, c3)
    lines = [
        "=" * 68,
        "OPERATING POINT REPORT",
        f"Model-axis electrode order: {model_v1.arm_labels(data)}",
        "=" * 68,
        f"Measurement source: {data['raw_capture_path']}",
        f"Fit cube    : {fit_cube_path}",
        "",
        "Suggested cavity recentering",
        f"  Safe-step fraction : {BIAS_SAFE_FRACTION:.2f}",
        f"  Current bias       : {_format_point(data['dc_volts'])}",
    ]
    for idx, item in enumerate(bias["per_cavity"]):
        full_flag = (
            "inside-scan" if bias["full_target_in_scan"][idx] else "outside-scan"
        )
        safe_flag = (
            "inside-scan" if bias["safe_target_in_scan"][idx] else "outside-scan"
        )
        lines.extend(
            [
                f"  {item['name']} : delta_dc_pm={item['delta_dc_pm']:+8.3f}  eta={item['eta_hz_v'] / 1000000000.0:7.4f} GHz/V  slope={item['slope_pm_per_v']:+7.3f} pm/V",
                f"       full move : {item['delta_v']:+7.3f} V  -> {item['full_target_v']:+7.3f} V  ({full_flag})",
                f"       safe move : {BIAS_SAFE_FRACTION * item['delta_v']:+7.3f} V  -> {item['safe_target_v']:+7.3f} V  ({safe_flag})",
            ]
        )
    lines.extend(
        [
            "",
            f"  Proposed centered bias : {_format_point(bias['full_target'])}",
            f"  Predicted T at center  : {_format_triplet(bias['pred_t_full'])}",
            f"  Conservative first step: {_format_point(bias['safe_target'])}",
            f"  Predicted T at safe    : {_format_triplet(bias['pred_t_safe'])}",
            "",
            "Best achievable modeled detector response within scanned bias box",
            "  Interpretation: maximum calibrated modeled detector value for each channel",
            "  searched continuously inside the voltage box covered by this scan.",
        ]
    )
    for name, optimum in per_channel:
        lines.extend(
            [
                f"  {name}",
                f"    best point     : {_format_point(optimum['point'])}",
                f"    detector value : {optimum['detector_value']:.4f}",
                f"    all detectors  : {_format_triplet(optimum['all_t'])}",
                f"    raw port power : {_format_triplet(optimum['all_p'])}",
            ]
        )
    lines.extend(
        [
            "",
            "Balanced operating point",
            "  Heuristic: maximize the joint normalized detector response rather than",
            "  optimizing a single channel only.",
            f"  point           : {_format_point(balanced['point'])}",
            f"  detector values : {_format_triplet(balanced['all_t'])}",
            f"  normalized vals : {_format_triplet(balanced['norm_t'])}",
            f"  raw port power  : {_format_triplet(balanced['all_p'])}",
        ]
    )
    return "\n".join(lines) + "\n"


def _save_and_print_operating_report(outputs: dict) -> str:
    text = _build_operating_report(outputs)
    report_path = os.path.join(outputs["outdir"], OPERATING_REPORT_NAME)
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return report_path


def main() -> None:
    import argparse
    from pathlib import Path
    from run_workflow import read_npz
    from run_saved_result_with_3d_v4 import _apply_full_perm

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fit", type=Path, required=True, help="Saved full_cube_fit NPZ."
    )
    parser.add_argument("--reconstructed", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="New operating-point report."
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new output report path.")
    fit = read_npz(args.fit)
    reconstruction = read_npz(args.reconstructed)
    data = model_v1.cube_payload_to_data(
        reconstruction,
        str(args.reconstructed),
        str(args.reconstructed),
        model_v1.AWG_SR_DEFAULT,
        len(reconstruction["e_k1"]) - 1,
    )
    data = _apply_full_perm(
        data,
        np.asarray(fit.get("applied_voltage_perm", [0, 1, 2]), dtype=int).tolist(),
        np.asarray(fit.get("applied_detector_perm", [0, 1, 2]), dtype=int).tolist(),
    )
    outputs = {"result": fit, "data": data, "full_cube_fit_path": str(args.fit)}
    text = _build_operating_report(outputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
