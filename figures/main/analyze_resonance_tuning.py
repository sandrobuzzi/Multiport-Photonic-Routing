import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from tfln_paths import DATA_ROOT, output_path

# %% Imports and configuration
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, correlate, hilbert
from scipy.ndimage import gaussian_filter
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.optimize import curve_fit

# ============================================================================
# User configuration
# ============================================================================

# Data file path
FILE_NAME = "A46_PowerVoltageSweep_031125_16-14.npz"
DATA_PATH = DATA_ROOT / "fig3/fig3b/original_source" / FILE_NAME

# Cavity selection (options: 'K1', 'K2', 'K3', or list like ['K1', 'K2'])
CAVITIES = ["K1", "K3"]  # Single cavity or list of cavities

# Powers to plot (in dBm) - will be validated against available powers
POWERS_TO_PLOT = [-14, -9, -4, 1, 6]

# Detector correction
DETECTOR1_CORRECTION_DB = 5.61  # dB correction to apply to detector1

# Wavelength range filtering
USE_WAVELENGTH_RANGE = True
WAVELENGTH_MIN = 1541.0  # nm
WAVELENGTH_MAX = 1542.3  # nm

# Voltage range filtering
USE_VOLTAGE_RANGE = True
VOLTAGE_MIN = -20  # V
VOLTAGE_MAX = 20  # V

# Tracking parameters
SIGMAS = (3,)  # Gaussian widths for template generation
MAX_STEP = 1000  # Maximum step size for DP tracking
SMOOTH = 0.0001  # Smoothness penalty for DP tracking
LAM_CORRECTION = -0.02  # Empirical wavelength correction (nm)

# Parametric fit parameters
APPLY_FIT = True
d_init = 0.1
w_init = 10
a1_init = 0.01
a2_init = 0.02
V0_init = 0.0

d_min = 0.1
w_min = 10
a1_min = 0.005
a2_min = 0.005
V0_min = -0.5
lambda_res_0_offset_min = -0.1

d_max = 0.2
w_max = 15
a1_max = 0.025
a2_max = 0.025
V0_max = 0.5
lambda_res_0_offset_max = 0.1

# Detector names
DETECTORS = ("detector1", "detector2", "detector3")

# Normalize CAVITIES to list
if isinstance(CAVITIES, str):
    CAVITIES = [CAVITIES]

# Figure styling
FIG_SIZE_MM = 56.0
FIG_SIZE_IN = FIG_SIZE_MM / 25.4  # ≈ 2.2047 in
FONT_SIZE_PT = 7
COLOR_CURVE = "#d22028"  # Color for the curve

# Power to plot (only -14 dBm)
POWER_TO_PLOT = -14  # dBm


# %% Helper functions


# Remove slow baseline drift with asymmetric least squares
def als_baseline(y, lam=1e6, p=0.05, n_iter=10):
    """Asymmetric Least Squares baseline removal"""
    L = y.size
    D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(L - 2, L), format="csc")
    K = (lam * (D.T @ D)).tocsc()
    w = np.ones(L)
    for _ in range(n_iter):
        W = sparse.diags(w, 0, format="csc")
        z = spsolve(W + K, W @ y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def preprocess(row, window=25, poly=3):
    """Smooth, detrend, and standardize a detector slice"""
    n = row.size
    win = min(window, n - (1 - n % 2))
    if win < poly + 2:
        win = poly + 3
    if win % 2 == 0:
        win += 1
    smooth = savgol_filter(row, win, poly, mode="interp")
    detrended = smooth - als_baseline(smooth)
    std = detrended.std()
    return (detrended - detrended.mean()) / (std if std > 1e-8 else 1.0)


def build_templates(sigmas=SIGMAS):
    """Build Gaussian and derivative templates for feature detection"""
    support = int(np.ceil(6 * max(sigmas)))
    x = np.arange(-support, support + 1)
    templates = []
    for sigma in sigmas:
        # Gaussian template
        g = np.exp(-0.5 * (x / sigma) ** 2)
        g -= g.mean()
        g /= np.linalg.norm(g) + 1e-12
        g_h = np.imag(hilbert(g))
        templates.append((g, g_h))
        # Derivative template
        dg = -(x / sigma**2) * np.exp(-0.5 * (x / sigma) ** 2)
        dg -= dg.mean()
        dg /= np.linalg.norm(dg) + 1e-12
        dg_h = np.imag(hilbert(dg))
        templates.append((dg, dg_h))
    return templates


TEMPLATES = build_templates()


def row_score(row):
    """Score spectral features using template matching"""
    r = preprocess(row)
    norm = np.linalg.norm(r) + 1e-12
    if norm < 1e-6:
        return np.zeros_like(r)
    best = np.zeros_like(r)
    for t, h in TEMPLATES:
        corr = correlate(r, t, mode="same")
        corr_h = correlate(r, h, mode="same")
        best = np.maximum(best, np.hypot(corr, corr_h))
    return best / norm


def robust_normalize(arr, low=10, high=99):
    """Robust normalization using percentiles"""
    a, b = np.percentile(arr, (low, high))
    if b <= a + 1e-12:
        return np.clip(arr - a, 0, None)
    return np.clip((arr - a) / (b - a), 0, 1)


def fuse_scores(score_dict):
    """Fuse detector scores using geometric mean"""
    stack = np.stack(
        [np.clip(robust_normalize(score_dict[det]), 1e-6, None) for det in DETECTORS]
    )
    fused = np.exp(np.mean(np.log(stack), axis=0))
    return gaussian_filter(fused, sigma=(0.5, 1.0), mode="nearest")


def motion_weight(score):
    """Weight features by positive temporal slope"""
    win = 2
    poly = 1
    slope = savgol_filter(
        score, window_length=win, polyorder=poly, deriv=1, axis=0, mode="interp"
    )
    slope = np.clip(slope, 0, None)
    return slope


def dp_track_path(score_map, d=MAX_STEP, alpha=SMOOTH):
    """Track the strongest continuous path through a 2D score map using dynamic programming."""
    cost = -score_map
    n_rows, n_cols = cost.shape

    dp = np.full((n_rows, n_cols), np.inf)
    back = np.full((n_rows, n_cols), -1, dtype=int)

    dp[0] = cost[0]

    for i in range(1, n_rows):
        prev = dp[i - 1]
        for j in range(n_cols):
            lo = max(0, j - d)
            hi = min(n_cols, j + d + 1)
            if lo >= hi:
                continue

            offsets = np.arange(lo, hi)
            vals = prev[offsets]

            finite = np.isfinite(vals)
            if not np.any(finite):
                continue

            offsets = offsets[finite]
            vals = vals[finite]

            shifts = np.abs(j - offsets)
            penalties = alpha * shifts

            idx_best = np.argmin(vals + penalties)
            best_val = vals[idx_best] + penalties[idx_best]

            if not np.isfinite(best_val):
                continue

            dp[i, j] = cost[i, j] + best_val
            back[i, j] = offsets[idx_best]

    last_row = dp[-1]
    if not np.isfinite(last_row).any():
        raise RuntimeError("Tracking failed: no finite path found")

    j = int(np.nanargmin(last_row))
    path = [j]

    for i in range(n_rows - 1, 0, -1):
        j = back[i, j]
        if j < 0:
            raise RuntimeError("Tracking back-pointer missing")
        path.append(j)

    return np.array(path[::-1])


def identify_sweep_column(voltage_matrix, decimals=6):
    """Identify which column in voltage matrix contains the sweep"""
    counts = [
        np.unique(np.round(voltage_matrix[:, col], decimals=decimals)).size
        for col in range(voltage_matrix.shape[1])
    ]
    max_count = max(counts)
    if counts.count(max_count) > 1:
        for col, count in enumerate(counts):
            if count > 1:
                return col
        return 0
    return int(np.argmax(counts))


def split_monotonic_segments(voltages, spectra, tol=1e-6):
    """Split voltage/spectra data into monotonic segments (up/down sweeps)."""
    voltages = np.asarray(voltages, dtype=float)
    spectra = np.asarray(spectra, dtype=float)
    indices = np.arange(voltages.size)

    if voltages.size == 0:
        return []
    if voltages.size == 1:
        return [
            dict(
                direction="flat",
                voltages=voltages.copy(),
                spectra=spectra.copy(),
                indices=indices.copy(),
            )
        ]

    segments = []
    start = 0
    last_dir = None

    for idx in range(1, voltages.size):
        delta = voltages[idx] - voltages[idx - 1]
        if abs(delta) <= tol:
            continue

        direction = "up" if delta > 0 else "down"
        if last_dir is None:
            last_dir = direction
            continue

        if direction != last_dir:
            seg_slice = slice(start, idx)
            segment = dict(
                direction=last_dir,
                voltages=voltages[seg_slice].copy(),
                spectra=spectra[seg_slice].copy(),
                indices=indices[seg_slice].copy(),
            )

            if segment["direction"] == "up":
                while (
                    segment["voltages"].size > 1
                    and abs(segment["voltages"][-1] - segment["voltages"][-2]) <= tol
                ):
                    segment["voltages"] = segment["voltages"][:-1]
                    segment["spectra"] = segment["spectra"][:-1]
                    segment["indices"] = segment["indices"][:-1]

            if last_dir == "down":
                segment["voltages"] = segment["voltages"][::-1]
                segment["spectra"] = segment["spectra"][::-1]
                segment["indices"] = segment["indices"][::-1]

            segments.append(segment)
            start = idx - 1
            last_dir = direction

    seg_slice = slice(start, None)
    final_dir = last_dir or "flat"
    segment = dict(
        direction=final_dir,
        voltages=voltages[seg_slice].copy(),
        spectra=spectra[seg_slice].copy(),
        indices=indices[seg_slice].copy(),
    )
    if final_dir == "down":
        segment["voltages"] = segment["voltages"][::-1]
        segment["spectra"] = segment["spectra"][::-1]
        segment["indices"] = segment["indices"][::-1]

    segments.append(segment)
    return [seg for seg in segments if seg["spectra"].size]


def parametric_model(V, d, w, a1, a2, V0, lambda_res_0):
    """
    Parametric model for cavity resonance vs voltage.

    λ(V) = -d/2 * f1(V-V0) + f2(V-V0)*(V-V0) + lambda_res_0

    where:
    - f1 = tanh(2*arccosh(1/sqrt(p))/w * (V-V0))
    - f2 = 1/2 * (a2-a1) * (f1 + 1) + a1
    """
    p = 0.5
    deltaV = V - V0
    arg_factor = 2 * np.arccosh(1 / np.sqrt(p)) / w
    f1 = np.tanh(arg_factor * deltaV)
    f2 = 0.5 * (a2 - a1) * (f1 + 1) + a1
    return -d / 2 * f1 + f2 * deltaV + lambda_res_0


def fit_parametric_curve(volt_axis, lam_vals):
    """Fit parametric curve to voltage-wavelength data"""
    try:
        zero_idx = np.argmin(np.abs(volt_axis))
        lambda_res_0_init = lam_vals[zero_idx]

        lambda_res_0_min = lambda_res_0_init + lambda_res_0_offset_min
        lambda_res_0_max = lambda_res_0_init + lambda_res_0_offset_max

        p0 = [d_init, w_init, a1_init, a2_init, V0_init, lambda_res_0_init]
        lower_bounds = [d_min, w_min, a1_min, a2_min, V0_min, lambda_res_0_min]
        upper_bounds = [d_max, w_max, a1_max, a2_max, V0_max, lambda_res_0_max]

        popt, pcov = curve_fit(
            parametric_model,
            volt_axis,
            lam_vals,
            p0=p0,
            bounds=(lower_bounds, upper_bounds),
            maxfev=5000,
        )

        lam_fit = parametric_model(volt_axis, *popt)

        ss_res = np.sum((lam_vals - lam_fit) ** 2)
        ss_tot = np.sum((lam_vals - np.mean(lam_vals)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        param_errors = np.sqrt(np.diag(pcov))

        return popt, param_errors, r_squared, lam_fit

    except Exception as e:
        print(f"Parametric fit failed: {e}")
        # Fallback to linear fit
        coeffs = np.polyfit(volt_axis, lam_vals, 1)
        linear_fit = np.polyval(coeffs, volt_axis)
        zero_idx = np.argmin(np.abs(volt_axis))
        intercept_at_zero = linear_fit[zero_idx]
        popt = [0, 30, coeffs[0], coeffs[0], 0, intercept_at_zero]
        param_errors = [0, 0, 0, 0, 0, 0]
        ss_res = np.sum((lam_vals - linear_fit) ** 2)
        ss_tot = np.sum((lam_vals - np.mean(lam_vals)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        return popt, param_errors, r_squared, linear_fit


# %% Matplotlib global style
plt.rcParams.update(
    {
        "font.size": FONT_SIZE_PT,
        "axes.labelsize": FONT_SIZE_PT,
        "xtick.labelsize": FONT_SIZE_PT,
        "ytick.labelsize": FONT_SIZE_PT,
    }
)

# %% Load and process data

print("Loading data...")
data = np.load(DATA_PATH, allow_pickle=True)

available_powers = data["power_levels"]
wavelength_axis = data["wavelengths"]
device_id = str(data["device_id"])

# Validate and select powers
if POWERS_TO_PLOT is None:
    POWERS_TO_PLOT = available_powers.tolist()
else:
    invalid_powers = [p for p in POWERS_TO_PLOT if p not in available_powers]
    if invalid_powers:
        raise ValueError(f"Requested powers {invalid_powers} not found in dataset.")


def power_to_key(power_dbm):
    """Convert power in dBm to key prefix"""
    if power_dbm < 0:
        return f"Pm{abs(int(power_dbm))}"
    else:
        return f"Pp{int(power_dbm)}"


DETECTOR1_CORRECTION_FACTOR = 10 ** (DETECTOR1_CORRECTION_DB / 10)

# Load spectra and voltages
cavity_power_data = {}
for cavity in CAVITIES:
    power_data = {}
    for power in POWERS_TO_PLOT:
        prefix = power_to_key(power)
        spectra_key = f"{prefix}_{cavity}_spectra"
        voltage_key = f"{prefix}_{cavity}_voltage_combinations"

        spectra = data[spectra_key].copy()
        voltages = data[voltage_key].copy()
        spectra[:, 0, :] *= DETECTOR1_CORRECTION_FACTOR

        power_data[power] = {"spectra": spectra, "voltages": voltages}
    cavity_power_data[cavity] = power_data

# Prepare wavelength axis
lam_axis_full = wavelength_axis.copy()
if USE_WAVELENGTH_RANGE:
    lam_mask = (lam_axis_full >= WAVELENGTH_MIN) & (lam_axis_full <= WAVELENGTH_MAX)
    lam_axis = lam_axis_full[lam_mask]
else:
    lam_mask = slice(None)
    lam_axis = lam_axis_full

# Process each cavity
all_cavity_results = {}

for CAVITY in CAVITIES:
    print(f"\nProcessing {CAVITY}...")
    power_data = cavity_power_data[CAVITY]
    processed_data = {}

    # Extract up sweeps
    for power in POWERS_TO_PLOT:
        spectra = power_data[power]["spectra"].copy()
        voltage_matrix = power_data[power]["voltages"]

        if isinstance(lam_mask, np.ndarray):
            spectra = spectra[:, :, lam_mask]

        sweep_col = identify_sweep_column(voltage_matrix)
        volt_axis = voltage_matrix[:, sweep_col].astype(float)

        segments = split_monotonic_segments(volt_axis, spectra)
        up_segment = None
        for seg in segments:
            if seg["direction"] == "up":
                up_segment = seg
                break

        if up_segment is None:
            continue

        volt_axis_up = up_segment["voltages"]
        spectra_up = up_segment["spectra"]

        if USE_VOLTAGE_RANGE:
            v_mask = (volt_axis_up >= VOLTAGE_MIN) & (volt_axis_up <= VOLTAGE_MAX)
            volt_axis_up = volt_axis_up[v_mask]
            spectra_up = spectra_up[v_mask, :, :]

        if volt_axis_up.size == 0:
            continue

        processed_data[power] = {"volt_axis": volt_axis_up, "spectra": spectra_up}

    # Extract resonance curves
    results = {}
    for power in sorted(processed_data.keys()):
        volt_axis = processed_data[power]["volt_axis"]
        spectra = processed_data[power]["spectra"]

        score_stack = {
            det: np.vstack([row_score(row) for row in spectra[:, det_idx, :]])
            for det_idx, det in enumerate(DETECTORS)
        }

        fused_score = fuse_scores(score_stack)
        tracker_map = motion_weight(fused_score)

        try:
            path_idx = dp_track_path(tracker_map)
            lam_vals = lam_axis[path_idx] + LAM_CORRECTION
        except Exception as e:
            print(f"  Ridge tracking failed for {power:+.0f} dBm: {e}")
            lam_vals = None

        # Fit parametric curve
        lam_fit = None
        fit_params = None
        fit_errors = None
        r_squared = None

        if APPLY_FIT and lam_vals is not None:
            fit_params, fit_errors, r_squared, lam_fit = fit_parametric_curve(
                volt_axis, lam_vals
            )

        results[power] = {
            "volt_axis": volt_axis,
            "lam_vals": lam_vals,
            "lam_fit": lam_fit,
            "fit_params": fit_params,
            "fit_errors": fit_errors,
            "r_squared": r_squared,
        }

    all_cavity_results[CAVITY] = results

print("\nData processing complete.")


# %% Plot resonance wavelength vs voltage

print("\nPlotting resonance wavelength vs voltage...")

for CAVITY in CAVITIES:
    results = all_cavity_results[CAVITY]

    if not results:
        print(f"No results for {CAVITY}")
        continue

    # Only plot the specified power level
    if POWER_TO_PLOT not in results:
        print(f"Power {POWER_TO_PLOT} dBm not available for {CAVITY}")
        continue

    res = results[POWER_TO_PLOT]
    if res["lam_fit"] is None:
        print(f"No fit available for {CAVITY} at {POWER_TO_PLOT} dBm")
        continue

    fig, ax = plt.subplots(figsize=(FIG_SIZE_IN, FIG_SIZE_IN))

    volt_axis = res["volt_axis"]
    lam_fit = res["lam_fit"]

    # Convert wavelength to frequency detuning (GHz)
    # Reference: wavelength at V=0 (or center of voltage range)
    zero_idx = int(np.argmin(np.abs(volt_axis)))
    lam_ref = lam_fit[zero_idx]

    # Frequency detuning: Δf = -c/λ² * Δλ
    c = 299792458.0  # Speed of light (m/s)
    lam_ref_m = lam_ref * 1e-9  # Convert to meters
    lam_fit_m = lam_fit * 1e-9

    # Frequency at reference: f_ref = c / lam_ref
    f_ref = c / lam_ref_m  # Hz
    f_fit = c / lam_fit_m  # Hz

    # Detuning in GHz
    detuning_GHz = (f_fit - f_ref) / 1e9

    # Calculate frequency tuning efficiency df/dV in GHz/V
    # First compute dλ/dV from the fitted curve
    window_length = min(41, volt_axis.size - (1 - volt_axis.size % 2))
    if window_length <= 3:
        window_length = 5
    if window_length % 2 == 0:
        window_length += 1
    polyorder = 3
    delta_v = np.mean(np.diff(volt_axis))
    if not np.isfinite(delta_v) or delta_v <= 0:
        delta_v = 1.0

    dlam_dV = savgol_filter(lam_fit, window_length, polyorder, deriv=1, delta=delta_v)

    # Convert to frequency tuning: dω/dV = -(2πc/λ²) * dλ/dV
    dlam_dV_m = dlam_dV * 1e-9
    domega_dV = -(2 * np.pi * c / lam_fit_m**2) * dlam_dV_m
    domega_dV_GHz = domega_dV / (2 * np.pi * 1e9)  # Convert to GHz/V

    # Find maximum absolute tuning efficiency
    max_tuning_GHz_per_V = np.max(np.abs(domega_dV_GHz))

    ax.plot(
        volt_axis,
        detuning_GHz,
        color=COLOR_CURVE,
        linewidth=1,
        alpha=1,
        marker=".",
        markersize=6,
    )

    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Detuning (GHz)")

    # Add maximum tuning efficiency text in top right
    text_str = f"slope: {max_tuning_GHz_per_V:.2f} GHz/V"
    ax.text(
        0.98,
        0.98,
        text_str,
        transform=ax.transAxes,
        fontsize=FONT_SIZE_PT,
        fontfamily="Arial",
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", alpha=1, edgecolor="none"
        ),
    )

    # No grid for a clean look
    ax.grid(False)

    # Tight layout and minimal outer padding
    fig.tight_layout(pad=0.0)

    output_name = f"{CAVITY}_tuning_curve_{POWER_TO_PLOT}dBm.svg"
    fig.savefig(
        output_path("figures/main/" + output_name), bbox_inches="tight", pad_inches=0.0
    )
    print(f"  Saved: {output_name}")

    plt.close(fig)

    print(f"    Done for {CAVITY}.")

print("\nDone.")
