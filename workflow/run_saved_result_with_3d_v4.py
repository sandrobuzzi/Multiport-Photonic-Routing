"""Joint voltage/detector permutation search and reciprocal-model fitting."""

from __future__ import annotations
import os
from itertools import permutations as _iterperms
import numpy as np
import model_v1
import run_saved_result_with_3d_v2 as runner_v2

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
PERMUTATION_SEARCH_ALWAYS = True
PERMUTATION_SEARCH_R2_THRESHOLD = 0.55
PERM_SCREEN_MAXITER = 40
PERM_TOP_CANDIDATES = 3
PERM_POLISH_MAXITER = 150
_HERE = os.path.dirname(os.path.abspath(__file__))
_ALL_PERMS = list(_iterperms([0, 1, 2]))
_PERM_NAMES = {
    (0, 1, 2): "identity",
    (1, 0, 2): "K1<->K2",
    (0, 2, 1): "K2<->K3",
    (2, 1, 0): "K1<->K3",
    (1, 2, 0): "cycle K1->K2->K3",
    (2, 0, 1): "cycle K1->K3->K2",
}


def _perm_name(perm: list[int]) -> str:
    return _PERM_NAMES.get(tuple(perm), str(perm))


def _apply_perm_for_fitting(data: dict, perm_v: list[int], perm_d: list[int]) -> dict:
    """Return a shallow-copy of data with voltage and detector axes permuted.

    Only touches the arrays used by the optimiser (v1/v2/v3, dc_volts,
    t_exp, weights, laser_nm).  The cube structure (e1/e2/e3, cubes, count,
    mask) is left unchanged so this function is cheap.
    """
    d = dict(data)
    volt_arrs = [data["v1"], data["v2"], data["v3"]]
    d["v1"] = volt_arrs[perm_v[0]]
    d["v2"] = volt_arrs[perm_v[1]]
    d["v3"] = volt_arrs[perm_v[2]]
    d["dc_volts"] = data["dc_volts"][list(perm_v)]
    d["t_exp"] = data["t_exp"][:, list(perm_d)]
    return d


def _apply_full_perm(data: dict, perm_v: list[int], perm_d: list[int]) -> dict:
    """Return a deep-enough copy of data with ALL arrays permuted.

    Needed after the permutation search to produce correct plots and the
    operating-point report.
    """
    d = dict(data)
    volt_arrs = [data["v1"], data["v2"], data["v3"]]
    d["v1"] = volt_arrs[perm_v[0]]
    d["v2"] = volt_arrs[perm_v[1]]
    d["v3"] = volt_arrs[perm_v[2]]
    d["dc_volts"] = np.asarray(data["dc_volts"])[list(perm_v)]
    edge_arrs = [data["e1"], data["e2"], data["e3"]]
    center_arrs = [data["c1"], data["c2"], data["c3"]]
    d["e1"] = edge_arrs[perm_v[0]]
    d["e2"] = edge_arrs[perm_v[1]]
    d["e3"] = edge_arrs[perm_v[2]]
    d["c1"] = center_arrs[perm_v[0]]
    d["c2"] = center_arrs[perm_v[1]]
    d["c3"] = center_arrs[perm_v[2]]
    axes = list(perm_v)
    if data.get("count") is not None:
        d["count"] = np.transpose(data["count"], axes)
    if data.get("mask") is not None:
        d["mask"] = np.transpose(data["mask"], axes)
    if data.get("cubes") is not None:
        transposed = [np.transpose(cube, axes) for cube in data["cubes"]]
        d["cubes"] = [transposed[perm_d[i]] for i in range(3)]
    raw_v = [data.get("v1_raw"), data.get("v2_raw"), data.get("v3_raw")]
    if all((v is not None for v in raw_v)):
        d["v1_raw"] = raw_v[perm_v[0]]
        d["v2_raw"] = raw_v[perm_v[1]]
        d["v3_raw"] = raw_v[perm_v[2]]
    d["t_exp"] = data["t_exp"][:, list(perm_d)]
    raw_out = [data.get("out1_raw"), data.get("out2_raw"), data.get("out3_raw")]
    if all((v is not None for v in raw_out)):
        d["out1_raw"] = raw_out[perm_d[0]]
        d["out2_raw"] = raw_out[perm_d[1]]
        d["out3_raw"] = raw_out[perm_d[2]]
    d["applied_voltage_perm"] = np.asarray(perm_v, dtype=int)
    d["applied_detector_perm"] = np.asarray(perm_d, dtype=int)
    return d


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


def _quick_polish(
    data: dict, guess: np.ndarray, maxiter: int
) -> tuple[dict, np.ndarray]:
    result, x, _, _ = runner_v2._polish_guess(data, guess, FIT_REG_SCALE, maxiter)
    result["r2"] = model_v1._compute_r2(result, data)
    return (result, x)


def _search_permutations(data: dict) -> list[dict]:
    """Try all 36 (V-perm, D-perm) combinations and return sorted by objective."""
    default_guess = _default_free_q_guess()
    results = []
    for perm_v in _ALL_PERMS:
        for perm_d in _ALL_PERMS:
            perm_data = _apply_perm_for_fitting(data, list(perm_v), list(perm_d))
            result, x = _quick_polish(perm_data, default_guess, PERM_SCREEN_MAXITER)
            mean_r2 = float(np.nanmean(result["r2"]))
            min_r2 = float(np.nanmin(result["r2"]))
            results.append(
                {
                    "perm_v": list(perm_v),
                    "perm_d": list(perm_d),
                    "objective": result["objective"],
                    "mean_r2": mean_r2,
                    "min_r2": min_r2,
                    "r2": result["r2"].copy(),
                    "x": x,
                }
            )
    results.sort(key=lambda r: r["objective"])
    return results


def _select_best_permutation(data: dict) -> tuple[list[int], list[int], np.ndarray]:
    """Run permutation search; return (perm_v, perm_d, warm_start_x)."""
    if not PERMUTATION_SEARCH_ALWAYS:
        id_data = _apply_perm_for_fitting(data, [0, 1, 2], [0, 1, 2])
        id_result, _ = _quick_polish(
            id_data, _default_free_q_guess(), PERM_SCREEN_MAXITER
        )
        id_r2 = model_v1._compute_r2(id_result, id_data)
        mean_r2 = float(np.nanmean(id_r2))
        if mean_r2 >= PERMUTATION_SEARCH_R2_THRESHOLD:
            return ([0, 1, 2], [0, 1, 2], _default_free_q_guess())
    ranked = _search_permutations(data)
    polished = []
    for idx, entry in enumerate(ranked[:PERM_TOP_CANDIDATES], start=1):
        pv = entry["perm_v"]
        pd = entry["perm_d"]
        perm_data = _apply_perm_for_fitting(data, pv, pd)
        result, x = _quick_polish(perm_data, entry["x"], PERM_POLISH_MAXITER)
        r2 = model_v1._compute_r2(result, perm_data)
        polished.append(
            {
                "perm_v": pv,
                "perm_d": pd,
                "objective": result["objective"],
                "mean_r2": float(np.nanmean(r2)),
                "x": x,
            }
        )
    polished.sort(key=lambda r: r["objective"])
    winner = polished[0]
    return (winner["perm_v"], winner["perm_d"], winner["x"])


def _configure_runners() -> None:
    runner_v2.UNIFORM_OUTDIR_NAME = UNIFORM_OUTDIR_NAME
    runner_v2.FIT_CUBE_BINS = FIT_CUBE_BINS
    runner_v2.FIT_REG_SCALE = FIT_REG_SCALE
    runner_v2.MAX_RECENT_CANDIDATES = MAX_RECENT_CANDIDATES
    runner_v2.MOST_RECENT_ANY_BIAS = MOST_RECENT_ANY_BIAS
    runner_v2.PREPOLISH_CANDIDATES = PREPOLISH_CANDIDATES
    runner_v2.PREPOLISH_MAXITER = PREPOLISH_MAXITER
    runner_v2.FINAL_POLISH_MAXITER = FINAL_POLISH_MAXITER
    runner_v2.ENABLE_DE_FALLBACK = ENABLE_DE_FALLBACK
    runner_v2.DE_FALLBACK_ITERS = DE_FALLBACK_ITERS
    runner_v2.DE_FALLBACK_POP = DE_FALLBACK_POP
    runner_v2.DE_FALLBACK_RESTARTS = DE_FALLBACK_RESTARTS
    runner_v2.DE_FALLBACK_STAGE2_MAXITER = DE_FALLBACK_STAGE2_MAXITER
    runner_v2.FAST_PATH_MIN_MEAN_R2 = FAST_PATH_MIN_MEAN_R2
    runner_v2.FAST_PATH_MIN_SINGLE_R2 = FAST_PATH_MIN_SINGLE_R2


def _default_uniform_outdir(base_name: str) -> str:
    if UNIFORM_OUTDIR_NAME:
        return os.path.join(_HERE, UNIFORM_OUTDIR_NAME)
    if base_name.startswith("uniform_cube_"):
        return os.path.join(_HERE, f"fit_uniform_{base_name[len('uniform_cube_'):]}")
    return os.path.join(_HERE, f"fit_{base_name}")


def _fit_uniform_cube_capture_v4(npz_path: str) -> dict:
    base_name = os.path.splitext(os.path.basename(npz_path))[0]
    outdir = _default_uniform_outdir(base_name)
    os.makedirs(outdir, exist_ok=True)
    reconstructed_cube = os.path.join(outdir, "reconstructed_exp_cube.npz")
    data, capture = model_v1.reconstruct_exp_cube_from_raw(
        raw_capture_path=npz_path,
        awg_sr=model_v1.AWG_SR_DEFAULT,
        cube_bins=FIT_CUBE_BINS,
        cube_out_path=reconstructed_cube,
    )
    perm_v, perm_d, warm_x = _select_best_permutation(data)
    if perm_v != [0, 1, 2] or perm_d != [0, 1, 2]:
        data = _apply_full_perm(data, perm_v, perm_d)
    data["applied_voltage_perm"] = np.asarray(perm_v, dtype=int)
    data["applied_detector_perm"] = np.asarray(perm_d, dtype=int)
    perm_info_path = os.path.join(outdir, "applied_permutation.txt")
    with open(perm_info_path, "w", encoding="utf-8") as f:
        f.write(f"voltage_perm : {perm_v}  ({_perm_name(perm_v)})\n")
        f.write(f"detector_perm: {perm_d}  ({_perm_name(perm_d)})\n")
        f.write(f"source_file  : {npz_path}\n")
    candidates = runner_v2._discover_recent_fit_candidates(
        raw_capture=npz_path,
        dc_volts=np.asarray(data["dc_volts"], dtype=np.float64),
        laser_nm=float(data["laser_nm"]),
        outdir=outdir,
    )
    start_infos = []
    perm_start_result = runner_v2._evaluate_start(data, warm_x, FIT_REG_SCALE)
    start_infos.append(
        {
            "label": f"permutation-search winner V={perm_v} D={perm_d}",
            "guess": warm_x,
            "source_path": None,
            "start_result": perm_start_result,
            "same_capture": False,
        }
    )
    for meta in candidates:
        guess = model_v1.load_previous_fit_guess(meta["path"], free_q=True)
        if guess is None:
            continue
        start_result = runner_v2._evaluate_start(data, guess, FIT_REG_SCALE)
        label = os.path.relpath(meta["path"], _HERE)
        start_infos.append(
            {
                "label": label,
                "guess": guess,
                "source_path": meta["path"],
                "start_result": start_result,
                "same_capture": meta["same_capture"],
            }
        )
    default_guess = runner_v2._default_free_q_guess()
    start_infos.append(
        {
            "label": "built-in reference",
            "guess": default_guess,
            "source_path": None,
            "start_result": runner_v2._evaluate_start(
                data, default_guess, FIT_REG_SCALE
            ),
            "same_capture": False,
        }
    )
    start_infos.sort(key=lambda item: item["start_result"]["objective"])
    prepolish_infos = []
    prepolish_infos.append(start_infos[0])
    for info in start_infos[1:]:
        if any(info is selected for selected in prepolish_infos):
            continue
        if info.get("same_capture"):
            prepolish_infos.insert(1, info)
        else:
            prepolish_infos.append(info)
        if len(prepolish_infos) >= PREPOLISH_CANDIDATES:
            break
    polished = []
    for idx, info in enumerate(prepolish_infos, start=1):
        result, x_best, calls, elapsed_s = runner_v2._polish_guess(
            data,
            init_guess=info["guess"],
            reg_scale=FIT_REG_SCALE,
            maxiter=PREPOLISH_MAXITER,
        )
        result["r2"] = model_v1._compute_r2(result, data)
        polished.append(
            {
                "label": info["label"],
                "source_path": info.get("source_path"),
                "x_best": x_best,
                "result": result,
            }
        )
    best_prepolish = min(polished, key=lambda item: item["result"]["objective"])
    best_result, best_x, final_calls, final_elapsed_s = runner_v2._polish_guess(
        data,
        init_guess=best_prepolish["x_best"],
        reg_scale=FIT_REG_SCALE,
        maxiter=FINAL_POLISH_MAXITER,
    )
    best_result["r2"] = model_v1._compute_r2(best_result, data)
    if ENABLE_DE_FALLBACK and runner_v2._fast_path_is_weak(best_result):
        fallback_outputs = model_v1.run_fit_from_raw_capture(
            raw_capture=npz_path,
            outdir=outdir,
            reconstructed_cube=reconstructed_cube,
            cube_bins=FIT_CUBE_BINS,
            reg_scale=FIT_REG_SCALE,
            init_fit=best_prepolish["source_path"],
            de_iters=DE_FALLBACK_ITERS,
            de_pop=DE_FALLBACK_POP,
            restarts=DE_FALLBACK_RESTARTS,
            stage2_maxiter=DE_FALLBACK_STAGE2_MAXITER,
            prepared=(data, capture),
        )
        fallback_outputs["data"] = data
        fallback_outputs["result"]["r2"] = model_v1._compute_r2(
            fallback_outputs["result"], data
        )
        fallback_outputs["perm_v"] = perm_v
        fallback_outputs["perm_d"] = perm_d
        outputs = fallback_outputs
    else:
        full_cube_fit_path = runner_v2._write_fit_outputs(best_result, data, outdir)
        outputs = {
            "raw_capture_path": npz_path,
            "outdir": outdir,
            "reconstructed_cube_path": reconstructed_cube,
            "fit_report_path": os.path.join(outdir, "fit_report.txt"),
            "fit_params_path": os.path.join(outdir, "fit_params.npz"),
            "full_cube_fit_path": full_cube_fit_path,
            "result": best_result,
            "data": data,
            "capture": capture,
            "perm_v": perm_v,
            "perm_d": perm_d,
        }
    return outputs
