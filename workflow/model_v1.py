"""Reciprocal cavity model, saved-capture reconstruction and numerical fitting.

Internal port indices retain the acquisition convention; see README for the
map to manuscript P1/P2/P3. Explicit paths are supplied by run_workflow.py."""

from __future__ import annotations
import os
import numpy as np
from scipy.optimize import differential_evolution, minimize

_HERE = os.path.dirname(os.path.abspath(__file__))
_RESULTS_DIR = "INSERT PATH HERE FOR THE RAW CAPTURE DIRECTORY"
OUTDIR_DEFAULT = "INSERT PATH HERE FOR FIT OUTPUTS"
C_MPS = 299792458.0
INPUT_PORT_INDEX = 2
DETECTOR_PORT_ORDER = np.array([2, 0, 1], dtype=int)
DETECTOR_LABELS = ["P1 / CH2", "P2 / CH3", "P3 / CH4"]
A_IN = np.eye(3, dtype=np.complex128)[INPUT_PORT_INDEX]
ETA_REF_HZ_V = 1700000000.0
Q_REF = 36000.0
X_REF = 5.0
THETA_CANONICAL = np.zeros(3, dtype=np.float64)
AMP_REF = np.ones(2, dtype=np.float64)
PHASE_V_SLOPE_REF = 0.0
BG_AMP_REF = np.full(3, 0.001, dtype=np.float64)
AWG_SR_DEFAULT = 2400000000.0
FIT_CUBE_BINS = 20
DEFAULT_F_LO_HZ = 10000000.0
DEFAULT_F_HI_HZ = 50000000.0
DEFAULT_SEED = 42
SCOPE_CHANNELS = [1, 2, 3, 4]


def arm_labels(data: dict) -> list[str]:
    """Physical electrode labels in the current model-axis order."""
    return [f"K{int(i)+1}" for i in data.get("applied_voltage_perm", [0, 1, 2])]


def detector_labels(data: dict) -> list[str]:
    """Manuscript output and scope labels in the current fitted detector order."""
    return [
        DETECTOR_LABELS[int(i)] for i in data.get("applied_detector_perm", [0, 1, 2])
    ]


def _default_raw_capture():
    raise ValueError(
        "Provide the raw capture path with run_workflow.py fit --raw INSERT_PATH_HERE.npz."
    )


def _default_init_fit_path():
    return None


def _npz_scalar(npz_obj: np.lib.npyio.NpzFile, key: str, default=None):
    if key not in npz_obj.files:
        if default is None:
            source = getattr(npz_obj, "filename", "<npz>")
            raise KeyError(f"Missing required key {key!r} in {source}")
        return default
    value = npz_obj[key]
    return value.item() if np.asarray(value).shape == () else value


def _load_k_biases_from_npz(npz_obj: np.lib.npyio.NpzFile) -> np.ndarray:
    return np.array(
        [
            float(_npz_scalar(npz_obj, "k1_voltage", 0.0)),
            float(_npz_scalar(npz_obj, "k2_voltage", 0.0)),
            float(_npz_scalar(npz_obj, "k3_voltage", 0.0)),
        ],
        dtype=np.float64,
    )


def load_raw_capture(npz_path: str) -> dict:
    npz = np.load(npz_path, allow_pickle=True)
    required_keys = [
        "scope_ch1_time",
        "scope_ch1_voltage",
        "scope_ch2_voltage",
        "scope_ch3_voltage",
        "scope_ch4_voltage",
        "awg_period_s",
        "time_origin_s",
    ]
    required_keys += [
        "f_lo_hz",
        "f_hi_hz",
        "seed",
        "awg_amplitude",
        "awg_output_range_v",
        "chirp_seed_d_in_s",
        "chirp_seed_d_out_s",
        "laser_wavelength_nm",
        "k1_voltage",
        "k2_voltage",
        "k3_voltage",
    ]
    if "is_strobe" in npz.files and bool(np.asarray(npz["is_strobe"]).item()):
        raise ValueError(
            "Use a multisine cube scan here; this input is a strobe capture."
        )
    if "is_targeted_visit" in npz.files and bool(
        np.asarray(npz["is_targeted_visit"]).item()
    ):
        raise ValueError(
            "Use the targeted-extinction analysis for a targeted trajectory capture."
        )
    missing = [key for key in required_keys if key not in npz.files]
    if missing:
        waveform_hint = ""
        if "waveform_type" in npz.files:
            waveform_type = np.asarray(npz["waveform_type"]).item()
            waveform_hint = f" Detected waveform_type={waveform_type!r}; this looks like an AWG sweep capture, not a uniform_cube raw capture."
        raise ValueError(
            f"Unsupported raw capture format: {npz_path}. Missing keys: {', '.join(missing)}.{waveform_hint}"
        )
    scope_t_ref = np.asarray(npz["scope_ch1_time"], dtype=np.float64)
    scope_data = {}
    for ch in SCOPE_CHANNELS:
        time_key = f"scope_ch{ch}_time"
        volt_key = f"scope_ch{ch}_voltage"
        scope_data[ch] = (
            np.asarray(
                npz[time_key] if time_key in npz.files else scope_t_ref,
                dtype=np.float64,
            ),
            np.asarray(npz[volt_key], dtype=np.float64),
        )
    return {
        "npz_path": npz_path,
        "capture_name": os.path.splitext(os.path.basename(npz_path))[0],
        "k_biases": _load_k_biases_from_npz(npz),
        "laser_wavelength_nm": float(_npz_scalar(npz, "laser_wavelength_nm", 1542.0)),
        "awg_amplitude": float(_npz_scalar(npz, "awg_amplitude", 1.0)),
        "awg_output_range_v": float(_npz_scalar(npz, "awg_output_range_v", 5.0)),
        "awg_period_s": float(_npz_scalar(npz, "awg_period_s")),
        "f_lo_hz": float(_npz_scalar(npz, "f_lo_hz", DEFAULT_F_LO_HZ)),
        "f_hi_hz": float(_npz_scalar(npz, "f_hi_hz", DEFAULT_F_HI_HZ)),
        "seed": int(_npz_scalar(npz, "seed", DEFAULT_SEED)),
        "marker_start_s": float(_npz_scalar(npz, "marker_start_s", 1e-06)),
        "chirp_seed_d_in_s": np.asarray(
            _npz_scalar(npz, "chirp_seed_d_in_s", np.zeros(3)), dtype=np.float64
        ),
        "chirp_seed_d_out_s": np.asarray(
            _npz_scalar(npz, "chirp_seed_d_out_s", np.zeros(3)), dtype=np.float64
        ),
        "time_origin_s": float(_npz_scalar(npz, "time_origin_s")),
        "scope_data": scope_data,
    }


def compute_fourier_bins(f_lo_hz: float, f_hi_hz: float, period_s: float) -> np.ndarray:
    k_lo = int(np.ceil(f_lo_hz * period_s))
    k_hi = int(np.floor(f_hi_hz * period_s))
    return np.arange(k_lo, k_hi + 1)


def synthesize_multisine(
    n_samples: int, bins: np.ndarray, n_channels: int = 3, seed: int = DEFAULT_SEED
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    bins = np.asarray(bins, dtype=int)
    phases = rng.uniform(0.0, 2.0 * np.pi, (n_channels, len(bins)))
    signals = np.zeros((n_channels, n_samples), dtype=np.float64)
    half_n = n_samples / 2.0
    nyq = n_samples // 2
    for ch in range(n_channels):
        spec = np.zeros(n_samples, dtype=np.complex128)
        coeffs = half_n * np.exp(1j * phases[ch])
        spec[bins] = coeffs
        valid = (bins > 0) & (bins < nyq)
        spec[n_samples - bins[valid]] = np.conj(coeffs[valid])
        signals[ch] = np.fft.ifft(spec).real
    return signals


def orthogonalize_channels(signals: np.ndarray) -> np.ndarray:
    rms_orig = np.std(signals, axis=1)
    q_mat, _ = np.linalg.qr(signals.T)
    ortho = q_mat.T
    rms_new = np.std(ortho, axis=1)
    scale = np.where(rms_new > 0.0, rms_orig / rms_new, 1.0)
    return ortho * scale[:, None]


def normalize_peak(signals: np.ndarray, target_peak: float = 0.999) -> np.ndarray:
    out = signals.copy()
    for ch in range(out.shape[0]):
        peak = float(np.max(np.abs(out[ch])))
        if peak > 0.0:
            out[ch] *= target_peak / peak
    return out


def normalize_uniform_remap(signals: np.ndarray, bounds=(-1.0, 1.0)) -> np.ndarray:
    lo, hi = bounds
    n_ch, n_samples = signals.shape
    out = np.empty_like(signals)
    target = np.linspace(lo, hi, n_samples)
    for ch in range(n_ch):
        order = np.argsort(signals[ch])
        mapped = np.empty(n_samples, dtype=np.float64)
        mapped[order] = target
        out[ch] = mapped
    return out


def build_cube_trajectory(
    n_samples: int, period_s: float, f_lo_hz: float, f_hi_hz: float, seed: int
) -> np.ndarray:
    bins = compute_fourier_bins(f_lo_hz, f_hi_hz, period_s)
    signals = synthesize_multisine(n_samples, bins, n_channels=3, seed=seed)
    signals = orthogonalize_channels(signals)
    signals = normalize_peak(signals)
    signals = normalize_uniform_remap(signals)
    return signals


def get_input_trajectory(
    waveforms: np.ndarray, t_array: np.ndarray, delays_samples: np.ndarray | None = None
) -> np.ndarray:
    n_ch, n_samples = waveforms.shape
    t_array = np.asarray(t_array, dtype=np.float64)
    if delays_samples is None:
        delays_samples = np.zeros(n_ch, dtype=np.float64)
    out = np.zeros((n_ch, len(t_array)), dtype=np.float64)
    for ch in range(n_ch):
        te = (t_array - delays_samples[ch]) % n_samples
        i0 = np.floor(te).astype(int) % n_samples
        i1 = (i0 + 1) % n_samples
        frac = te - np.floor(te)
        out[ch] = (1.0 - frac) * waveforms[ch, i0] + frac * waveforms[ch, i1]
    return out


def reconstruct_capture(capture: dict, new_waves: np.ndarray, awg_sr: float) -> dict:
    scope_data = capture["scope_data"]
    d_in_s = np.asarray(capture["chirp_seed_d_in_s"], dtype=np.float64)
    d_out_s = np.asarray(capture["chirp_seed_d_out_s"], dtype=np.float64)
    time_origin_s = float(capture["time_origin_s"])
    t2, v2 = [np.asarray(x, np.float64) for x in scope_data[2]]
    t3, v3 = [np.asarray(x, np.float64) for x in scope_data[3]]
    t4, v4 = [np.asarray(x, np.float64) for x in scope_data[4]]
    t_out1 = t2 - time_origin_s - d_out_s[0]
    t_out2 = t3 - time_origin_s - d_out_s[1]
    t_out3 = t4 - time_origin_s - d_out_s[2]
    out1 = v2
    out2 = np.interp(t_out1, t_out2, v3)
    out3 = np.interp(t_out1, t_out3, v4)
    awg_idx = t_out1 * awg_sr
    inputs = get_input_trajectory(new_waves, awg_idx, delays_samples=d_in_s * awg_sr)
    amp = float(capture["awg_amplitude"])
    phys_range = float(capture["awg_output_range_v"])
    v1_in = inputs[0] * phys_range * amp
    v2_in = inputs[1] * phys_range * amp
    v3_in = inputs[2] * phys_range * amp
    k_biases = np.asarray(capture["k_biases"], dtype=np.float64)
    return {
        "t_device_s": t_out1,
        "out1": out1,
        "out2": out2,
        "out3": out3,
        "v1_in": v1_in,
        "v2_in": v2_in,
        "v3_in": v3_in,
        "v1_total": v1_in + k_biases[0],
        "v2_total": v2_in + k_biases[1],
        "v3_total": v3_in + k_biases[2],
    }


def build_reconstructed_cube_payload(
    recon: dict, capture: dict, cube_bins: int
) -> dict:

    def bin_cube(v1, v2, v3, sig, n_bins, e1, e2, e3):
        acc = np.zeros((n_bins, n_bins, n_bins), dtype=np.float64)
        cnt = np.zeros((n_bins, n_bins, n_bins), dtype=int)
        i1 = np.clip(np.digitize(v1, e1) - 1, 0, n_bins - 1)
        i2 = np.clip(np.digitize(v2, e2) - 1, 0, n_bins - 1)
        i3 = np.clip(np.digitize(v3, e3) - 1, 0, n_bins - 1)
        np.add.at(acc, (i1, i2, i3), sig)
        np.add.at(cnt, (i1, i2, i3), 1)
        cube = np.full((n_bins, n_bins, n_bins), np.nan)
        filled = cnt > 0
        cube[filled] = acc[filled] / cnt[filled]
        return (cube, cnt)

    v1 = recon["v1_total"]
    v2 = recon["v2_total"]
    v3 = recon["v3_total"]
    e1 = np.linspace(float(v1.min()), float(v1.max()), cube_bins + 1)
    e2 = np.linspace(float(v2.min()), float(v2.max()), cube_bins + 1)
    e3 = np.linspace(float(v3.min()), float(v3.max()), cube_bins + 1)
    cb1, cnt = bin_cube(v1, v2, v3, recon["out1"], cube_bins, e1, e2, e3)
    cb2, _ = bin_cube(v1, v2, v3, recon["out2"], cube_bins, e1, e2, e3)
    cb3, _ = bin_cube(v1, v2, v3, recon["out3"], cube_bins, e1, e2, e3)
    return {
        "e_k1": e1,
        "e_k2": e2,
        "e_k3": e3,
        "cube_CH2_OUT1": cb1,
        "cube_CH3_OUT2": cb2,
        "cube_CH4_OUT3": cb3,
        "count_CH2_OUT1": cnt,
        "V1_raw": v1,
        "V2_raw": v2,
        "V3_raw": v3,
        "out1_raw": recon["out1"],
        "out2_raw": recon["out2"],
        "out3_raw": recon["out3"],
        "dc_volts": np.asarray(capture["k_biases"], dtype=np.float64),
        "laser_wavelength_nm": np.array(float(capture["laser_wavelength_nm"])),
        "awg_amplitude": np.array(float(capture["awg_amplitude"])),
        "awg_output_range_v": np.array(float(capture["awg_output_range_v"])),
        "t_origin": np.array(float(capture["time_origin_s"])),
        "source_raw_capture": np.array(capture["npz_path"]),
        "capture_name": np.array(capture["capture_name"]),
        "seed": np.array(int(capture["seed"])),
        "f_lo_hz": np.array(float(capture["f_lo_hz"])),
        "f_hi_hz": np.array(float(capture["f_hi_hz"])),
    }


def save_reconstructed_cube(payload: dict, cube_path: str) -> None:
    np.savez(cube_path, **payload)


def cube_payload_to_data(
    payload: dict, raw_capture_path: str, cube_path: str, awg_sr: float, cube_bins: int
) -> dict:
    count = np.asarray(payload["count_CH2_OUT1"])
    mask = count > 0
    e1 = np.asarray(payload["e_k1"], dtype=np.float64)
    e2 = np.asarray(payload["e_k2"], dtype=np.float64)
    e3 = np.asarray(payload["e_k3"], dtype=np.float64)
    c1 = 0.5 * (e1[:-1] + e1[1:])
    c2 = 0.5 * (e2[:-1] + e2[1:])
    c3 = 0.5 * (e3[:-1] + e3[1:])
    i1, i2, i3 = np.where(mask)
    v1 = c1[i1]
    v2 = c2[i2]
    v3 = c3[i3]
    cubes = [
        np.asarray(payload["cube_CH2_OUT1"], dtype=np.float64),
        np.asarray(payload["cube_CH3_OUT2"], dtype=np.float64),
        np.asarray(payload["cube_CH4_OUT3"], dtype=np.float64),
    ]
    t_exp = np.stack([cube[mask] for cube in cubes], axis=1)
    weights = np.sqrt(count[mask].astype(np.float64))
    weights /= np.mean(weights)
    return {
        "cube_path": cube_path,
        "raw_capture_path": raw_capture_path,
        "dc_volts": np.asarray(payload["dc_volts"], dtype=np.float64),
        "laser_nm": float(np.asarray(payload["laser_wavelength_nm"]).item()),
        "e1": e1,
        "e2": e2,
        "e3": e3,
        "c1": c1,
        "c2": c2,
        "c3": c3,
        "count": count,
        "mask": mask,
        "cubes": cubes,
        "v1": v1,
        "v2": v2,
        "v3": v3,
        "t_exp": t_exp,
        "weights": weights,
        "v1_raw": np.asarray(payload["V1_raw"], dtype=np.float64),
        "v2_raw": np.asarray(payload["V2_raw"], dtype=np.float64),
        "v3_raw": np.asarray(payload["V3_raw"], dtype=np.float64),
        "out1_raw": np.asarray(payload["out1_raw"], dtype=np.float64),
        "out2_raw": np.asarray(payload["out2_raw"], dtype=np.float64),
        "out3_raw": np.asarray(payload["out3_raw"], dtype=np.float64),
        "awg_output_range_v": float(np.asarray(payload["awg_output_range_v"]).item()),
        "awg_sr": float(awg_sr),
        "cube_bins": int(cube_bins),
    }


def reconstruct_exp_cube_from_raw(
    raw_capture_path: str, awg_sr: float, cube_bins: int, cube_out_path: str
) -> tuple[dict, dict]:
    capture = load_raw_capture(raw_capture_path)
    n_samples = int(round(capture["awg_period_s"] * awg_sr))
    new_waves = build_cube_trajectory(
        n_samples=n_samples,
        period_s=float(capture["awg_period_s"]),
        f_lo_hz=float(capture["f_lo_hz"]),
        f_hi_hz=float(capture["f_hi_hz"]),
        seed=int(capture["seed"]),
    )
    recon = reconstruct_capture(capture, new_waves, awg_sr=awg_sr)
    payload = build_reconstructed_cube_payload(recon, capture, cube_bins=cube_bins)
    save_reconstructed_cube(payload, cube_out_path)
    data = cube_payload_to_data(
        payload,
        raw_capture_path=raw_capture_path,
        cube_path=cube_out_path,
        awg_sr=awg_sr,
        cube_bins=cube_bins,
    )
    return (data, capture)


def make_mmi(theta_rel: np.ndarray, amp_rel: np.ndarray) -> np.ndarray:
    a_diag, a_far = amp_rel
    w = np.exp(2j * np.pi / 3.0)
    phase_template = np.array(
        [[1.0, 1.0, 1.0], [1.0, w, np.conj(w)], [1.0, np.conj(w), w]],
        dtype=np.complex128,
    )
    mag_template = np.array(
        [[a_diag, 1.0, a_far], [1.0, a_diag, 1.0], [a_far, 1.0, a_diag]],
        dtype=np.float64,
    )
    raw = mag_template * phase_template
    raw /= np.sqrt(np.mean(np.abs(raw) ** 2))
    return raw / np.sqrt(3.0)


def make_bg_field(bg_amp: np.ndarray, bg_phase_rel: np.ndarray) -> np.ndarray:
    phase = np.array([0.0, bg_phase_rel[0], bg_phase_rel[1]], dtype=np.float64)
    return bg_amp * np.exp(1j * phase)


def phcc_reflection_vec(
    omega: float,
    omega_res0: float,
    eta_hz: float,
    v_arr: np.ndarray,
    q_val: float,
    x_val: float,
) -> np.ndarray:
    kappa = omega_res0 / q_val
    kappa_e = kappa * x_val / (1.0 + x_val)
    omega_res = omega_res0 + 2.0 * np.pi * eta_hz * v_arr
    delta = omega_res - omega
    return 1.0 - 2.0 * kappa_e / (2j * delta + kappa)


def load_previous_fit_guess(path: str | None, free_q: bool) -> np.ndarray | None:
    if not path or not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=True)
    required = ["delta_dc_pm", "x_arr", "eta_arr", "amp_rel"]
    if any((key not in d.files for key in required)):
        return None
    log_q = None
    if free_q and "q_arr" in d.files:
        log_q = np.log(np.asarray(d["q_arr"], dtype=np.float64))
    psi_rel = (
        np.asarray(d["psi_rel"], dtype=np.float64)
        if "psi_rel" in d.files
        else np.zeros(2, dtype=np.float64)
    )
    return pack_params(
        np.asarray(d["delta_dc_pm"], dtype=np.float64),
        np.log(np.asarray(d["x_arr"], dtype=np.float64)),
        np.log(np.asarray(d["eta_arr"], dtype=np.float64)),
        np.log(np.asarray(d["amp_rel"], dtype=np.float64)),
        psi_rel,
        (
            float(np.asarray(d["phase_v_slope"]).item())
            if "phase_v_slope" in d.files
            else PHASE_V_SLOPE_REF
        ),
        (
            np.asarray(d["bg_amp"], dtype=np.float64)
            if "bg_amp" in d.files
            else np.zeros(3, dtype=np.float64)
        ),
        (
            np.asarray(d["bg_phase_rel"], dtype=np.float64)
            if "bg_phase_rel" in d.files
            else np.zeros(2, dtype=np.float64)
        ),
        log_q,
    )


def model_power(
    v1_arr: np.ndarray,
    v2_arr: np.ndarray,
    v3_arr: np.ndarray,
    laser_nm: float,
    dc_volts: np.ndarray,
    delta_dc_pm: np.ndarray,
    x_arr: np.ndarray,
    eta_arr: np.ndarray,
    q_arr: np.ndarray,
    theta_rel: np.ndarray,
    amp_rel: np.ndarray,
    phi_rel: np.ndarray,
    phase_v_slope: float,
    bg_amp: np.ndarray,
    bg_phase_rel: np.ndarray,
) -> np.ndarray:
    laser_wl = laser_nm * 1e-09
    omega_src = 2.0 * np.pi * C_MPS / laser_wl
    lambda_res_dc = laser_wl + delta_dc_pm * 1e-12
    dlambda_dv = -(laser_wl**2 / C_MPS) * eta_arr
    lambda_res0 = lambda_res_dc - dlambda_dv * dc_volts
    omega_res0 = 2.0 * np.pi * C_MPS / lambda_res0
    mmi = make_mmi(theta_rel, amp_rel)
    psi = np.array([0.0, phi_rel[0], phi_rel[1]], dtype=np.float64)
    one_way_phase = np.exp(1j * psi)[:, np.newaxis]
    b_vec = mmi @ A_IN
    bg_field = make_bg_field(bg_amp, bg_phase_rel)
    v_arrs = [v1_arr, v2_arr, v3_arr]
    r_field = np.stack(
        [
            phcc_reflection_vec(
                omega_src,
                omega_res0[idx],
                eta_arr[idx],
                v_arrs[idx],
                q_arr[idx],
                x_arr[idx],
            )
            for idx in range(3)
        ],
        axis=0,
    )
    dv_arrs = np.stack(
        [
            np.asarray(v1_arr, dtype=np.float64) - float(dc_volts[0]),
            np.asarray(v2_arr, dtype=np.float64) - float(dc_volts[1]),
            np.asarray(v3_arr, dtype=np.float64) - float(dc_volts[2]),
        ],
        axis=0,
    )
    r_field *= np.exp(1j * phase_v_slope * dv_arrs)
    outbound = one_way_phase * b_vec[:, np.newaxis]
    reflected = r_field * outbound
    returned = one_way_phase * reflected
    e_ports = mmi.T @ returned + bg_field[:, np.newaxis]
    p_ports = np.abs(e_ports).T ** 2
    return p_ports[:, DETECTOR_PORT_ORDER]


def fit_linear_calibration(p_sim: np.ndarray, t_exp: np.ndarray, weights: np.ndarray):
    n_det = t_exp.shape[1]
    cal = np.zeros((n_det, 2), dtype=np.float64)
    pred = np.zeros_like(t_exp)
    sse = 0.0
    for port in range(n_det):
        y = t_exp[:, port] * weights
        x = np.column_stack([p_sim[:, port] * weights, weights])
        coeffs, *_ = np.linalg.lstsq(x, y, rcond=None)
        cal[port] = coeffs
        pred[:, port] = coeffs[0] * p_sim[:, port] + coeffs[1]
        resid = (pred[:, port] - t_exp[:, port]) * weights
        sse += float(np.dot(resid, resid))
    wrmse = np.sqrt(sse / (t_exp.shape[0] * n_det))
    return (wrmse, cal, pred)


def similarity_penalty(
    delta_dc_pm: np.ndarray,
    x_arr: np.ndarray,
    eta_arr: np.ndarray,
    q_arr: np.ndarray,
    amp_rel: np.ndarray,
    phase_v_slope: float,
    bg_amp: np.ndarray,
    reg_scale: float,
) -> float:
    log_x = np.log(x_arr)
    log_q = np.log(q_arr)
    log_amp = np.log(amp_rel)
    spread_x = np.mean((log_x - log_x.mean()) ** 2) / np.log(1.35) ** 2
    spread_eta = np.mean(((eta_arr - eta_arr.mean()) / ETA_REF_HZ_V) ** 2) / 0.11**2
    spread_q = np.mean((log_q - log_q.mean()) ** 2) / np.log(1.12) ** 2
    detune_mag = np.mean((delta_dc_pm / 35.0) ** 2)
    prior_eta = np.mean(((eta_arr - ETA_REF_HZ_V) / 280000000.0) ** 2)
    prior_q = np.mean((log_q - np.log(Q_REF)) ** 2) / np.log(1.18) ** 2
    prior_x = (np.log(x_arr.mean()) - np.log(X_REF)) ** 2 / np.log(3.0) ** 2
    prior_amp = np.mean((log_amp / np.log(1.1)) ** 2)
    prior_phase_v = (phase_v_slope / 0.05) ** 2
    prior_bg = np.mean((bg_amp / 0.1) ** 2)
    return reg_scale * (
        0.35 * spread_x
        + 0.35 * spread_eta
        + 0.15 * spread_q
        + 0.05 * detune_mag
        + 0.08 * prior_eta
        + 0.4 * prior_q
        + 0.01 * prior_x
        + 0.12 * prior_amp
        + 0.06 * prior_phase_v
        + 0.08 * prior_bg
    )


def pack_params(
    delta_dc_pm,
    log_x,
    log_eta,
    log_amp_rel,
    psi_rel,
    phase_v_slope,
    bg_amp,
    bg_phase_rel,
    log_q=None,
):
    parts = [
        delta_dc_pm,
        log_x,
        log_eta,
        log_amp_rel,
        psi_rel,
        np.array([phase_v_slope], dtype=np.float64),
        bg_amp,
        bg_phase_rel,
    ]
    if log_q is not None:
        parts.append(log_q)
    return np.concatenate(parts).astype(np.float64)


def unpack_params(p: np.ndarray, free_q: bool):
    idx = 0
    delta_dc_pm = np.asarray(p[idx : idx + 3], dtype=np.float64)
    idx += 3
    log_x = np.asarray(p[idx : idx + 3], dtype=np.float64)
    idx += 3
    log_eta = np.asarray(p[idx : idx + 3], dtype=np.float64)
    idx += 3
    log_amp_rel = np.asarray(p[idx : idx + 2], dtype=np.float64)
    idx += 2
    psi_rel = np.asarray(p[idx : idx + 2], dtype=np.float64)
    idx += 2
    phase_v_slope = float(p[idx])
    idx += 1
    bg_amp = np.asarray(p[idx : idx + 3], dtype=np.float64)
    idx += 3
    bg_phase_rel = np.asarray(p[idx : idx + 2], dtype=np.float64)
    idx += 2
    log_q = np.full(3, np.log(Q_REF), dtype=np.float64)
    if free_q:
        log_q = np.asarray(p[idx : idx + 3], dtype=np.float64)
    return (
        delta_dc_pm,
        np.exp(log_x),
        np.exp(log_eta),
        THETA_CANONICAL.copy(),
        np.exp(log_amp_rel),
        psi_rel,
        phase_v_slope,
        bg_amp,
        bg_phase_rel,
        np.exp(log_q),
    )


def make_bounds(free_q: bool):
    bounds = []
    bounds += [(-60.0, 60.0)] * 3
    bounds += [(np.log(0.2), np.log(20.0))] * 3
    bounds += [(np.log(950000000.0), np.log(3000000000.0))] * 3
    bounds += [(np.log(0.85), np.log(1.15))] * 2
    bounds += [(-np.pi, np.pi)] * 2
    bounds += [(-0.15, 0.15)]
    bounds += [(0.0, 0.35)] * 3
    bounds += [(-np.pi, np.pi)] * 2
    if free_q:
        bounds += [(np.log(25000.0), np.log(60000.0))] * 3
    return bounds


def make_objective(data: dict, reg_scale: float, free_q: bool):
    call_count = [0]

    def objective(p: np.ndarray) -> float:
        (
            delta_dc_pm,
            x_arr,
            eta_arr,
            theta_rel,
            amp_rel,
            psi_rel,
            phase_v_slope,
            bg_amp,
            bg_phase_rel,
            q_arr,
        ) = unpack_params(p, free_q=free_q)
        p_sim = model_power(
            data["v1"],
            data["v2"],
            data["v3"],
            data["laser_nm"],
            data["dc_volts"],
            delta_dc_pm,
            x_arr,
            eta_arr,
            q_arr,
            theta_rel,
            amp_rel,
            psi_rel,
            phase_v_slope,
            bg_amp,
            bg_phase_rel,
        )
        wrmse, _, _ = fit_linear_calibration(p_sim, data["t_exp"], data["weights"])
        reg = similarity_penalty(
            delta_dc_pm,
            x_arr,
            eta_arr,
            q_arr,
            amp_rel,
            phase_v_slope,
            bg_amp,
            reg_scale=reg_scale,
        )
        call_count[0] += 1
        return float(wrmse + reg)

    return (objective, call_count)


def evaluate_solution(
    best_p: np.ndarray, data: dict, reg_scale: float, free_q: bool
) -> dict:
    (
        delta_dc_pm,
        x_arr,
        eta_arr,
        theta_rel,
        amp_rel,
        psi_rel,
        phase_v_slope,
        bg_amp,
        bg_phase_rel,
        q_arr,
    ) = unpack_params(best_p, free_q=free_q)
    p_sim = model_power(
        data["v1"],
        data["v2"],
        data["v3"],
        data["laser_nm"],
        data["dc_volts"],
        delta_dc_pm,
        x_arr,
        eta_arr,
        q_arr,
        theta_rel,
        amp_rel,
        psi_rel,
        phase_v_slope,
        bg_amp,
        bg_phase_rel,
    )
    wrmse, cal, pred = fit_linear_calibration(p_sim, data["t_exp"], data["weights"])
    reg = similarity_penalty(
        delta_dc_pm,
        x_arr,
        eta_arr,
        q_arr,
        amp_rel,
        phase_v_slope,
        bg_amp,
        reg_scale=reg_scale,
    )
    return {
        "delta_dc_pm": delta_dc_pm,
        "x_arr": x_arr,
        "eta_arr": eta_arr,
        "theta_rel": theta_rel,
        "amp_rel": amp_rel,
        "psi_rel": psi_rel,
        "phi_rel": psi_rel,
        "phase_v_slope": float(phase_v_slope),
        "bg_amp": bg_amp,
        "bg_phase_rel": bg_phase_rel,
        "q_arr": q_arr,
        "p_sim": p_sim,
        "cal": cal,
        "pred": pred,
        "wrmse": float(wrmse),
        "objective": float(wrmse + reg),
        "reg": float(reg),
    }


def report(result: dict, data: dict, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    corr = np.asarray(result.get("r2", _compute_r2(result, data)), dtype=np.float64)
    lines = [
        "=" * 68,
        "UNIFORM CUBE FIT REPORT",
        "=" * 68,
        f"Reconstructed cube: {data['cube_path']}",
        f"Raw capture       : {data['raw_capture_path']}",
        f"Laser wavelength  : {data['laser_nm']:.5f} nm",
        f"Cube bins         : {data['cube_bins']}^3",
        f"Rebuild AWG SR    : {data['awg_sr'] / 1000000000.0:.4f} GSa/s",
        "Internal model input: index 2 (manuscript illuminated input P1)",
        f"Electrode axis order: {arm_labels(data)}",
        f"Fitted output order: {detector_labels(data)}",
        "Model variant     : canonical DFT-like MMI gauge + 2 MMI amplitudes + 2 physical arm phases + coherent detector background",
        f"Filled bins       : {data['v1'].shape[0]} / {data['count'].size}",
        f"Weighted RMS      : {result['wrmse']:.6f} V",
        f"Objective         : {result['objective']:.6f}",
        f"Regularization    : {result['reg']:.6f}",
        "",
        "Cavity parameters",
    ]
    for idx, name in enumerate(arm_labels(data)):
        lines += [
            f"  {name} @ {data['dc_volts'][idx]:+.2f} V",
            f"    delta_dc_pm : {result['delta_dc_pm'][idx]:+8.3f}",
            f"    x           : {result['x_arr'][idx]:8.4f}",
            f"    eta_hz_v    : {result['eta_arr'][idx] / 1000000000.0:8.4f} GHz/V",
            f"    Q           : {result['q_arr'][idx]:8.1f}",
        ]
    lines += ["", "MMI phase gauge"]
    lines.append("  Fixed canonical DFT-like tritter phase template")
    lines += ["", "MMI amplitude imbalance"]
    lines.append(f"  a_diag      : {result['amp_rel'][0]:8.4f}")
    lines.append(f"  a_far       : {result['amp_rel'][1]:8.4f}")
    lines += ["", "Physical one-way arm phases  (relative to arm 1)"]
    lines.append(
        f"  psi_arm[1]   : {result['psi_rel'][0]:+8.4f} rad  ({np.degrees(result['psi_rel'][0]):+7.2f} deg)"
    )
    lines.append(
        f"  psi_arm[2]   : {result['psi_rel'][1]:+8.4f} rad  ({np.degrees(result['psi_rel'][1]):+7.2f} deg)"
    )
    lines += ["", "Weak common voltage-dependent reflection phase"]
    lines.append(
        f"  phase_v_slope : {result['phase_v_slope']:+8.5f} rad/V  ({np.degrees(result['phase_v_slope']):+8.3f} deg/V)"
    )
    lines += ["", "Coherent detector background  (port-relative field)"]
    lines.append(f"  bg_amp[DET1] : {result['bg_amp'][0]:8.4f}")
    lines.append(
        f"  bg_amp[DET2] : {result['bg_amp'][1]:8.4f}  phase={result['bg_phase_rel'][0]:+8.4f} rad"
    )
    lines.append(
        f"  bg_amp[DET3] : {result['bg_amp'][2]:8.4f}  phase={result['bg_phase_rel'][1]:+8.4f} rad"
    )
    lines += ["", "Detector calibration  T_exp = alpha * P_sim + beta"]
    for port, name in enumerate(detector_labels(data)):
        lines.append(
            f"  {name:<10} : alpha={result['cal'][port, 0]:+8.5f}  beta={result['cal'][port, 1]:+8.5f}  R^2={corr[port]:.4f}"
        )
    report_path = os.path.join(outdir, "fit_report.txt")
    text = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(text + "\n")


def save_params(result: dict, data: dict, outdir: str):
    path = os.path.join(outdir, "fit_params.npz")
    np.savez(
        path,
        delta_dc_pm=result["delta_dc_pm"],
        x_arr=result["x_arr"],
        eta_arr=result["eta_arr"],
        q_arr=result["q_arr"],
        theta_rel=result["theta_rel"],
        amp_rel=result["amp_rel"],
        psi_rel=result["psi_rel"],
        phi_rel=result["phi_rel"],
        phase_v_slope=np.array(result["phase_v_slope"]),
        bg_amp=result["bg_amp"],
        bg_phase_rel=result["bg_phase_rel"],
        cal=result["cal"],
        r2=np.asarray(result.get("r2", _compute_r2(result, data)), dtype=np.float64),
        wrmse=result["wrmse"],
        objective=result["objective"],
        model_variant=np.array("canonical_mmi_amp_plus_physical_arm_phase_v1"),
        laser_nm=data["laser_nm"],
        dc_volts=data["dc_volts"],
        input_port=np.array(INPUT_PORT_INDEX + 1),
        detector_port_order=DETECTOR_PORT_ORDER + 1,
        detector_labels=np.asarray(detector_labels(data)),
        applied_voltage_perm=np.asarray(
            data.get("applied_voltage_perm", [0, 1, 2]), dtype=int
        ),
        applied_detector_perm=np.asarray(
            data.get("applied_detector_perm", [0, 1, 2]), dtype=int
        ),
        source_raw_capture=np.array(data["raw_capture_path"]),
        reconstructed_cube=np.array(data["cube_path"]),
        cube_bins=np.array(data["cube_bins"]),
        awg_sr=np.array(data["awg_sr"]),
    )


def build_full_cube(result: dict, data: dict, outdir: str):
    v1g, v2g, v3g = np.meshgrid(data["c1"], data["c2"], data["c3"], indexing="ij")
    p_full = model_power(
        v1g.ravel(),
        v2g.ravel(),
        v3g.ravel(),
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
    )
    t_full = np.stack(
        [
            result["cal"][port, 0] * p_full[:, port] + result["cal"][port, 1]
            for port in range(3)
        ],
        axis=1,
    )
    nb1 = len(data["c1"])
    nb2 = len(data["c2"])
    nb3 = len(data["c3"])
    t_cube = t_full.reshape(nb1, nb2, nb3, 3)
    p_cube = p_full.reshape(nb1, nb2, nb3, 3)
    path = os.path.join(outdir, "full_cube_fit.npz")
    np.savez(
        path,
        t_cube=t_cube,
        p_cube=p_cube,
        e1=data["e1"],
        e2=data["e2"],
        e3=data["e3"],
        c1=data["c1"],
        c2=data["c2"],
        c3=data["c3"],
        delta_dc_pm=result["delta_dc_pm"],
        x_arr=result["x_arr"],
        eta_arr=result["eta_arr"],
        q_arr=result["q_arr"],
        theta_rel=result["theta_rel"],
        amp_rel=result["amp_rel"],
        psi_rel=result["psi_rel"],
        phi_rel=result["phi_rel"],
        phase_v_slope=np.array(result["phase_v_slope"]),
        bg_amp=result["bg_amp"],
        bg_phase_rel=result["bg_phase_rel"],
        cal=result["cal"],
        model_variant=np.array("canonical_mmi_amp_plus_physical_arm_phase_v1"),
        input_port=np.array(INPUT_PORT_INDEX + 1),
        detector_port_order=DETECTOR_PORT_ORDER + 1,
        detector_labels=np.asarray(detector_labels(data)),
        applied_voltage_perm=np.asarray(
            data.get("applied_voltage_perm", [0, 1, 2]), dtype=int
        ),
        applied_detector_perm=np.asarray(
            data.get("applied_detector_perm", [0, 1, 2]), dtype=int
        ),
        source_raw_capture=np.array(data["raw_capture_path"]),
        reconstructed_cube=np.array(data["cube_path"]),
    )
    return t_cube


def _default_stage1_guess() -> np.ndarray:
    return pack_params(
        np.zeros(3, dtype=np.float64),
        np.full(3, np.log(X_REF), dtype=np.float64),
        np.full(3, np.log(ETA_REF_HZ_V), dtype=np.float64),
        np.log(AMP_REF),
        np.zeros(2, dtype=np.float64),
        PHASE_V_SLOPE_REF,
        BG_AMP_REF,
        np.zeros(2, dtype=np.float64),
    )


def _compute_r2(result: dict, data: dict) -> np.ndarray:
    return np.asarray(
        [
            float(
                np.corrcoef(data["t_exp"][:, port], result["pred"][:, port])[0, 1] ** 2
            )
            for port in range(data["t_exp"].shape[1])
        ],
        dtype=np.float64,
    )


def run_fit_from_raw_capture(
    raw_capture: str,
    outdir: str,
    reconstructed_cube: str | None = None,
    *,
    awg_sr: float = AWG_SR_DEFAULT,
    cube_bins: int = FIT_CUBE_BINS,
    de_iters: int = 60,
    de_pop: int = 10,
    restarts: int = 2,
    stage2_maxiter: int = 400,
    reg_scale: float = 0.015,
    init_fit: str | None = None,
    prepared: tuple[dict, dict] | None = None,
) -> dict:
    raw_capture = os.path.abspath(raw_capture)
    outdir = os.path.abspath(outdir)
    os.makedirs(outdir, exist_ok=True)
    reconstructed_cube = (
        os.path.abspath(reconstructed_cube)
        if reconstructed_cube
        else os.path.join(outdir, "reconstructed_exp_cube.npz")
    )
    if not os.path.exists(raw_capture):
        raise FileNotFoundError(f"Raw capture not found: {raw_capture}")
    if prepared is None:
        data, capture = reconstruct_exp_cube_from_raw(
            raw_capture_path=raw_capture,
            awg_sr=awg_sr,
            cube_bins=cube_bins,
            cube_out_path=reconstructed_cube,
        )
    else:
        data, capture = prepared
    resolved_init_fit = (
        os.path.abspath(init_fit) if init_fit else _default_init_fit_path()
    )
    bounds1 = make_bounds(free_q=False)
    obj1, calls1 = make_objective(data, reg_scale=reg_scale, free_q=False)
    best_stage1 = None
    best_loss1 = np.inf
    init_stage1 = load_previous_fit_guess(resolved_init_fit, free_q=False)
    if init_stage1 is None:
        init_stage1 = _default_stage1_guess()
    for restart in range(restarts):
        seed = 42 + 17 * restart
        res = differential_evolution(
            obj1,
            bounds1,
            maxiter=de_iters,
            popsize=de_pop,
            tol=1e-07,
            seed=seed,
            polish=False,
            workers=1,
            init="sobol",
        )
        if obj1(init_stage1) < res.fun:
            res_x = init_stage1.copy()
            res_fun = obj1(init_stage1)
        else:
            res_x = res.x
            res_fun = res.fun
        if res_fun < best_loss1:
            best_loss1 = float(res_fun)
            best_stage1 = res_x.copy()
    p0 = load_previous_fit_guess(resolved_init_fit, free_q=True)
    if p0 is None:
        p0 = np.concatenate([best_stage1, np.full(3, np.log(Q_REF), dtype=np.float64)])
    bounds2 = make_bounds(free_q=True)
    obj2, calls2 = make_objective(data, reg_scale=reg_scale, free_q=True)
    res2 = minimize(
        obj2,
        p0,
        method="L-BFGS-B",
        bounds=bounds2,
        options=dict(maxiter=stage2_maxiter, ftol=1e-12, gtol=1e-09),
    )
    result = evaluate_solution(res2.x, data, reg_scale=reg_scale, free_q=True)
    result["r2"] = _compute_r2(result, data)
    report(result, data, outdir)
    save_params(result, data, outdir)
    build_full_cube(result, data, outdir)
    return {
        "raw_capture_path": raw_capture,
        "outdir": outdir,
        "reconstructed_cube_path": reconstructed_cube,
        "fit_report_path": os.path.join(outdir, "fit_report.txt"),
        "fit_params_path": os.path.join(outdir, "fit_params.npz"),
        "full_cube_fit_path": os.path.join(outdir, "full_cube_fit.npz"),
        "result": result,
        "data": data,
        "capture": capture,
    }
