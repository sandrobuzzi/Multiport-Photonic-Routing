"""Multistart fit refinement with explicitly supplied initial-fit files."""

from __future__ import annotations
import os
import time
import numpy as np
import model_v1

UNIFORM_OUTDIR_NAME: str | None = None
FIT_CUBE_BINS = model_v1.FIT_CUBE_BINS
FIT_REG_SCALE = 0.015
MAX_RECENT_CANDIDATES = 8
MOST_RECENT_ANY_BIAS = 4
PREPOLISH_CANDIDATES = 3
PREPOLISH_MAXITER = 120
FINAL_POLISH_MAXITER = 400
ENABLE_DE_FALLBACK = True
DE_FALLBACK_ITERS = 45
DE_FALLBACK_POP = 8
DE_FALLBACK_RESTARTS = 1
DE_FALLBACK_STAGE2_MAXITER = 450
FAST_PATH_MIN_MEAN_R2 = 0.62
FAST_PATH_MIN_SINGLE_R2 = 0.2
_HERE = os.path.dirname(os.path.abspath(__file__))


def _default_free_q_guess() -> np.ndarray:
    return model_v1.pack_params(
        np.zeros(3, dtype=np.float64),
        np.full(3, np.log(model_v1.X_REF), dtype=np.float64),
        np.full(3, np.log(model_v1.ETA_REF_HZ_V), dtype=np.float64),
        np.log(model_v1.AMP_REF),
        np.zeros(2, dtype=np.float64),
        model_v1.PHASE_V_SLOPE_REF,
        model_v1.BG_AMP_REF,
        np.zeros(2, dtype=np.float64),
        np.full(3, np.log(model_v1.Q_REF), dtype=np.float64),
    )


def _safe_npz_scalar(npz_obj: np.lib.npyio.NpzFile, key: str, default=None):
    if key not in npz_obj.files:
        return default
    value = np.asarray(npz_obj[key])
    return value.item() if value.shape == () else value


def _capture_fit_metadata(
    path: str, raw_capture: str, dc_volts: np.ndarray, laser_nm: float
) -> dict:
    path = os.path.abspath(path)
    with np.load(path, allow_pickle=True) as data:
        source_raw = _safe_npz_scalar(data, "source_raw_capture", "")
        fit_dc = np.asarray(
            _safe_npz_scalar(data, "dc_volts", np.full(3, np.nan)), dtype=np.float64
        )
        fit_laser = float(_safe_npz_scalar(data, "laser_nm", np.nan))
        fit_r2 = np.asarray(
            _safe_npz_scalar(data, "r2", np.full(3, np.nan)), dtype=np.float64
        )
        fit_obj = float(_safe_npz_scalar(data, "objective", np.inf))
    source_norm = os.path.normcase(os.path.abspath(str(source_raw)))
    raw_norm = os.path.normcase(os.path.abspath(raw_capture))
    same_capture = source_norm == raw_norm
    dc_diff = (
        float(np.linalg.norm(fit_dc - dc_volts))
        if fit_dc.shape == dc_volts.shape
        else float("inf")
    )
    laser_diff = abs(fit_laser - laser_nm) if np.isfinite(fit_laser) else float("inf")
    finite_r2 = fit_r2[np.isfinite(fit_r2)]
    mean_r2 = float(np.mean(finite_r2)) if finite_r2.size else float("-inf")
    min_r2 = float(np.min(finite_r2)) if finite_r2.size else float("-inf")
    return {
        "path": path,
        "same_capture": same_capture,
        "dc_diff": dc_diff,
        "laser_diff": laser_diff,
        "mean_r2": mean_r2,
        "min_r2": min_r2,
        "objective": fit_obj,
        "mtime": os.path.getmtime(path),
    }


def _candidate_sort_key(meta: dict) -> tuple:
    return (
        0 if meta["same_capture"] else 1,
        meta["dc_diff"],
        meta["laser_diff"],
        -meta["mean_r2"],
        meta["objective"],
        -meta["mtime"],
    )


def _discover_recent_fit_candidates(
    raw_capture: str, dc_volts: np.ndarray, laser_nm: float, outdir: str
) -> list[dict]:
    paths = {os.path.abspath(path) for path in INITIAL_FIT_FILES}
    metas = [
        _capture_fit_metadata(
            path, raw_capture=raw_capture, dc_volts=dc_volts, laser_nm=laser_nm
        )
        for path in sorted(paths)
    ]
    ranked = sorted(metas, key=_candidate_sort_key)
    recent = sorted(metas, key=lambda item: item["mtime"], reverse=True)[
        :MOST_RECENT_ANY_BIAS
    ]
    selected = []
    seen = set()
    primary_limit = max(0, MAX_RECENT_CANDIDATES - MOST_RECENT_ANY_BIAS)
    for meta in ranked[:primary_limit]:
        if meta["path"] in seen:
            continue
        selected.append(meta)
        seen.add(meta["path"])
    for meta in recent:
        if meta["path"] in seen:
            continue
        selected.append(meta)
        seen.add(meta["path"])
        if len(selected) >= MAX_RECENT_CANDIDATES:
            return selected
    for meta in ranked[primary_limit:]:
        if meta["path"] in seen:
            continue
        selected.append(meta)
        seen.add(meta["path"])
        if len(selected) >= MAX_RECENT_CANDIDATES:
            return selected
    return selected


def _evaluate_start(data: dict, init_guess: np.ndarray, reg_scale: float) -> dict:
    result = model_v1.evaluate_solution(
        init_guess, data, reg_scale=reg_scale, free_q=True
    )
    result["r2"] = model_v1._compute_r2(result, data)
    return result


def _polish_guess(
    data: dict, init_guess: np.ndarray, reg_scale: float, maxiter: int
) -> tuple[dict, np.ndarray, int, float]:
    bounds = model_v1.make_bounds(free_q=True)
    objective, calls = model_v1.make_objective(data, reg_scale=reg_scale, free_q=True)
    t0 = time.time()
    res = model_v1.minimize(
        objective,
        init_guess,
        method="L-BFGS-B",
        bounds=bounds,
        options=dict(maxiter=maxiter, ftol=1e-12, gtol=1e-09),
    )
    result = model_v1.evaluate_solution(res.x, data, reg_scale=reg_scale, free_q=True)
    result["r2"] = model_v1._compute_r2(result, data)
    return (result, res.x, calls[0], time.time() - t0)


def _fast_path_is_weak(result: dict) -> bool:
    r2 = np.asarray(result["r2"], dtype=np.float64)
    mean_r2 = float(np.nanmean(r2))
    min_r2 = float(np.nanmin(r2))
    return mean_r2 < FAST_PATH_MIN_MEAN_R2 or min_r2 < FAST_PATH_MIN_SINGLE_R2


def _write_fit_outputs(result: dict, data: dict, outdir: str) -> str:
    model_v1.report(result, data, outdir)
    model_v1.save_params(result, data, outdir)
    model_v1.build_full_cube(result, data, outdir)
    return os.path.join(outdir, "full_cube_fit.npz")


INITIAL_FIT_FILES = []
