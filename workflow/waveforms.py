"""Offline multisine, targeted-trajectory and DC-balanced strobe generation.

Defaults describe the Fig. 4 scan. Uniform rank remapping changes the final
spectrum and correlation; compute_spectral_quality evaluates the final arrays."""

from __future__ import annotations
import numpy as np

AWG_SR = 2400000000.0
T_PERIOD_S = 2.5e-05
N_SAMPLES = 60000
F_LO_HZ = 10000000.0
F_HI_HZ = 50000000.0
SEED = 42
DO_ORTHOGONALISE = True
DO_UNIFORM_REMAP = True
AWG_OUTPUT_RANGE_V = 5.0
AWG_AMPLITUDE = 1.0
STROBE_CYCLE_FREQ_HZ = 10000000.0
STROBE_TUKEY_ALPHA = 0.7
STROBE_MAX_IDLE_SLOTS = 15


def compute_fourier_bins(f_lo_hz, f_hi_hz, period_s):
    k_lo = int(np.ceil(f_lo_hz * period_s))
    k_hi = int(np.floor(f_hi_hz * period_s))
    return np.arange(k_lo, k_hi + 1)


def synthesize_multisine(n_samples, bins, n_channels=3, seed=42, amplitudes=None):
    rng = np.random.default_rng(seed)
    bins = np.asarray(bins, dtype=int)
    if amplitudes is None:
        amplitudes = np.ones(len(bins), dtype=np.float64)
    phases = rng.uniform(0, 2 * np.pi, (n_channels, len(bins)))
    signals = np.zeros((n_channels, n_samples), dtype=np.float64)
    half_N = n_samples / 2.0
    nyq = n_samples // 2
    for ch in range(n_channels):
        spec = np.zeros(n_samples, dtype=np.complex128)
        coeffs = half_N * amplitudes * np.exp(1j * phases[ch])
        spec[bins] = coeffs
        valid = (bins > 0) & (bins < nyq)
        spec[n_samples - bins[valid]] = np.conj(coeffs[valid])
        signals[ch] = np.fft.ifft(spec).real
    return (signals, phases)


def orthogonalize_channels(signals):
    n_ch, N = signals.shape
    rms_orig = np.std(signals, axis=1)
    Q, R = np.linalg.qr(signals.T)
    ortho = Q.T
    rms_new = np.std(ortho, axis=1)
    scale = np.where(rms_new > 0, rms_orig / rms_new, 1.0)
    ortho *= scale[:, None]
    R_inv = np.linalg.inv(R)
    mixing_matrix = (R_inv * scale[None, :]).T
    return (ortho, mixing_matrix)


def normalize_peak(signals, target_peak=0.999):
    out = signals.copy()
    peaks = np.zeros(signals.shape[0])
    for ch in range(out.shape[0]):
        pk = np.max(np.abs(out[ch]))
        if pk > 0:
            out[ch] *= target_peak / pk
            peaks[ch] = pk
    return (out, peaks)


def normalize_uniform_remap(signals, bounds=(-1.0, 1.0)):
    lo, hi = bounds
    n_ch, N = signals.shape
    out = np.empty_like(signals)
    luts = []
    target = np.linspace(lo, hi, N)
    for ch in range(n_ch):
        order = np.argsort(signals[ch])
        mapped = np.empty(N, dtype=np.float64)
        mapped[order] = target
        out[ch] = mapped
        luts.append((signals[ch, order].copy(), target.copy()))
    return (out, luts)


def build_cube_trajectory(
    n_samples=N_SAMPLES,
    period_s=T_PERIOD_S,
    sample_rate_hz=AWG_SR,
    f_lo_hz=F_LO_HZ,
    f_hi_hz=F_HI_HZ,
    n_channels=3,
    seed=SEED,
    do_orthogonalise=DO_ORTHOGONALISE,
    do_uniform_remap=DO_UNIFORM_REMAP,
):
    if not isinstance(n_samples, (int, np.integer)) or n_samples < 2:
        raise ValueError("n_samples must be an integer greater than one.")
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive and finite.")
    if not np.isclose(n_samples, period_s * sample_rate_hz, rtol=0, atol=1e-7):
        raise ValueError("n_samples must equal period_s times sample_rate_hz.")
    if not np.isfinite(f_hi_hz) or not 0 < f_lo_hz <= f_hi_hz < sample_rate_hz / 2:
        raise ValueError("Require 0 < f_lo_hz <= f_hi_hz < the Nyquist frequency.")
    bins = compute_fourier_bins(f_lo_hz, f_hi_hz, period_s)
    if len(bins) < n_channels:
        raise ValueError("Use at least one Fourier bin per channel.")
    signals, phases = synthesize_multisine(n_samples, bins, n_channels, seed)
    meta = dict(
        bins=bins,
        phases=phases,
        seed=seed,
        f_lo_hz=f_lo_hz,
        f_hi_hz=f_hi_hz,
        period_s=period_s,
        sample_rate_hz=sample_rate_hz,
        mixing_matrix=np.eye(n_channels),
        peak_values=np.ones(n_channels),
        remap_luts=None,
        did_orthogonalise=do_orthogonalise,
        did_uniform_remap=do_uniform_remap,
    )
    if do_orthogonalise:
        signals, mix = orthogonalize_channels(signals)
        meta["mixing_matrix"] = mix
    signals, peaks = normalize_peak(signals)
    meta["peak_values"] = peaks
    if do_uniform_remap:
        signals, luts = normalize_uniform_remap(signals)
        meta["remap_luts"] = luts
    return (signals, meta)


def compute_spectral_quality(waveforms, period_s, f_lo_hz, f_hi_hz):
    n_ch, n_samples = waveforms.shape
    bins_inband = compute_fourier_bins(f_lo_hz, f_hi_hz, period_s)
    results = {}
    for ch in range(n_ch):
        spec = np.fft.rfft(waveforms[ch])
        power = np.abs(spec) ** 2
        total_power = float(power[1:].sum())
        inband_power = float(power[bins_inband].sum())
        dc_power = float(power[0])
        results[ch] = {
            "inband_fraction": inband_power / total_power if total_power > 0 else 0,
            "dc_fraction": (
                dc_power / (total_power + dc_power) if total_power + dc_power > 0 else 0
            ),
        }
    return results


def _bandpass_project(signals, n_samples, period_s, f_lo_hz, f_hi_hz):
    bins = compute_fourier_bins(f_lo_hz, f_hi_hz, period_s)
    n_ch = signals.shape[0]
    out = np.empty_like(signals)
    for ch in range(n_ch):
        spec = np.fft.rfft(signals[ch])
        mask = np.zeros(len(spec), dtype=bool)
        mask[bins] = True
        spec[~mask] = 0.0
        out[ch] = np.fft.irfft(spec, n=n_samples)
    return out


def _attractor_step(signals, targets_norm, strength, sigma):
    pts = signals.T.copy()
    N = pts.shape[0]
    sq_dists = np.zeros((N, 3), dtype=np.float64)
    for k in range(3):
        diff = pts - targets_norm[k]
        sq_dists[:, k] = np.sum(diff**2, axis=1)
    log_w = -sq_dists / (2.0 * sigma**2)
    log_w -= log_w.max(axis=1, keepdims=True)
    w = np.exp(log_w)
    w /= w.sum(axis=1, keepdims=True)
    attractor = w @ targets_norm
    pulled = (1.0 - strength) * pts + strength * attractor
    return pulled.T


def verify_target_concentration(signals, targets_norm, radius_norm):
    """Return the sample fractions within each target neighborhood."""
    points = np.asarray(signals).T
    inside = np.column_stack(
        [
            np.linalg.norm(points - target, axis=1) <= radius_norm
            for target in np.asarray(targets_norm)
        ]
    )
    return {
        "per_target_fraction": inside.mean(axis=0).tolist(),
        "any_target_fraction": float(inside.any(axis=1).mean()),
    }


def build_targeted_trajectory(
    targets_awg_v,
    radius_v,
    n_samples=N_SAMPLES,
    period_s=T_PERIOD_S,
    sample_rate_hz=AWG_SR,
    f_lo_hz=F_LO_HZ,
    f_hi_hz=F_HI_HZ,
    n_channels=3,
    seed=SEED,
    strength=0.7,
    n_iterations=10,
):
    targets_awg_v = np.asarray(targets_awg_v, dtype=np.float64)
    full_scale = AWG_OUTPUT_RANGE_V * AWG_AMPLITUDE
    targets_norm = targets_awg_v / full_scale
    R_norm = radius_v / full_scale
    dists = []
    for i in range(3):
        for j in range(i + 1, 3):
            dists.append(np.linalg.norm(targets_norm[i] - targets_norm[j]))
    sigma = max(0.3 * float(np.median(dists)), 0.05)
    signals, _ = build_cube_trajectory(
        n_samples=n_samples,
        period_s=period_s,
        sample_rate_hz=sample_rate_hz,
        f_lo_hz=f_lo_hz,
        f_hi_hz=f_hi_hz,
        n_channels=n_channels,
        seed=seed,
        do_orthogonalise=True,
        do_uniform_remap=True,
    )
    for it in range(n_iterations):
        signals = _attractor_step(signals, targets_norm, strength, sigma)
        signals = _bandpass_project(signals, n_samples, period_s, f_lo_hz, f_hi_hz)
        for ch in range(n_channels):
            pk = float(np.max(np.abs(signals[ch])))
            if pk > 1e-10:
                signals[ch] *= 0.999 / pk
    concentration = verify_target_concentration(signals, targets_norm, R_norm)
    spec_quality = compute_spectral_quality(signals, period_s, f_lo_hz, f_hi_hz)
    meta = dict(
        targets_awg_v=targets_awg_v,
        targets_norm=targets_norm,
        radius_v=radius_v,
        R_norm=R_norm,
        strength=strength,
        sigma=sigma,
        n_iterations=n_iterations,
        spectral_quality=spec_quality,
        target_concentration=concentration,
        seed=seed,
        f_lo_hz=f_lo_hz,
        f_hi_hz=f_hi_hz,
        period_s=period_s,
        sample_rate_hz=sample_rate_hz,
    )
    return (signals, meta)


def build_strobe_waveform(
    best_pts,
    n_idle_slots=1,
    n_samples=N_SAMPLES,
    period_s=T_PERIOD_S,
    cycle_freq_hz=STROBE_CYCLE_FREQ_HZ,
    sample_rate_hz=AWG_SR,
):
    """Fast strobe: RED->BLUE->GREEN->IDLE[*n_idle_slots] at cycle_freq_hz.

    n_idle_slots: number of consecutive IDLE slots (default=1).
    With n_idle_slots > 1 each IDLE slot carries -(R+G+B)/n_idle_slots so
    the per-slot IDLE peak is reduced by factor 1/n_idle_slots.
    Use check_strobe_dc_balance() to find the minimum feasible n_idle_slots.

    DC balance: every channel integrates to exactly zero over one full cycle
    (equal-area slots, sum of slot voltages = 0 per channel).

    Raises ValueError if any voltage exceeds the AWG range.

    Validity requirements:
      - sample_rate_hz / cycle_freq_hz must be an integer
      - samples_per_cycle must be divisible by (3 + n_idle_slots)
      - n_samples must be divisible by samples_per_cycle

    Returns
    -------
    waveform   : (3, n_samples) float64, normalised to [-1, 1]
    states_v   : list of (3+n_idle_slots) (3,) arrays in physical Volts
    state_names: ['RED', 'BLUE', 'GREEN', 'IDLE', ...] length=3+n_idle_slots
    """
    if not isinstance(n_idle_slots, (int, np.integer)) or n_idle_slots < 1:
        raise ValueError("n_idle_slots must be a positive integer.")
    if not np.isfinite(cycle_freq_hz) or cycle_freq_hz <= 0:
        raise ValueError("cycle_freq_hz must be positive and finite.")
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive and finite.")
    if not isinstance(n_samples, (int, np.integer)) or n_samples < 1:
        raise ValueError("n_samples must be a positive integer.")
    spc_f = sample_rate_hz / cycle_freq_hz
    spc = int(round(spc_f))
    if abs(spc - spc_f) > 1e-09:
        raise ValueError(
            f"AWG_SR / cycle_freq_hz = {spc_f:.3f} is not an integer. Choose f_cycle that divides {sample_rate_hz / 1000000000.0:.3f} GSa/s exactly (e.g. 10, 20, 30, 40, 60, 120 MHz)."
        )
    if spc < 1:
        raise ValueError("The cycle must contain at least one sample.")
    n_total_slots = 3 + n_idle_slots
    if spc % n_total_slots != 0:
        raise ValueError(
            f"samples_per_cycle={spc} is not divisible by n_total_slots={n_total_slots} (n_idle_slots={n_idle_slots}). Try a different n_idle_slots or cycle frequency."
        )
    if n_samples % spc != 0:
        raise ValueError(
            f"n_samples={n_samples} is not divisible by samples_per_cycle={spc}."
        )
    samples_per_state = spc // n_total_slots
    N_cycles = n_samples // spc
    red_v = np.asarray(best_pts["red"]["awg_v"], dtype=np.float64)
    blue_v = np.asarray(best_pts["blue"]["awg_v"], dtype=np.float64)
    green_v = np.asarray(best_pts["green"]["awg_v"], dtype=np.float64)
    idle_per_slot = -(red_v + blue_v + green_v) / float(n_idle_slots)
    state_names = ["RED", "BLUE", "GREEN"] + ["IDLE"] * n_idle_slots
    states_v = [red_v, blue_v, green_v] + [idle_per_slot] * n_idle_slots
    max_abs = float(np.max(np.abs(np.stack(states_v))))
    if max_abs > AWG_OUTPUT_RANGE_V + 1e-09:
        raise ValueError(
            f"Peak AWG voltage {max_abs:.3f} V exceeds range +-{AWG_OUTPUT_RANGE_V:.1f} V with n_idle_slots={n_idle_slots}. Call check_strobe_dc_balance(best_pts) to find a valid n_idle_slots."
        )
    alpha = STROBE_TUKEY_ALPHA
    n_trans = max(1, int(round(alpha * samples_per_state)))
    n_flat = samples_per_state - n_trans
    t_ramp = np.linspace(0.0, 1.0, n_trans, dtype=np.float64)
    crossfade = 0.5 * (1.0 - np.cos(np.pi * t_ramp))
    one_cycle = np.zeros((3, spc), dtype=np.float64)
    for k, v_curr in enumerate(states_v):
        v_next = states_v[(k + 1) % n_total_slots]
        i0 = k * samples_per_state
        for ch in range(3):
            one_cycle[ch, i0 : i0 + n_flat] = v_curr[ch]
            one_cycle[ch, i0 + n_flat : i0 + samples_per_state] = (
                v_curr[ch] + (v_next[ch] - v_curr[ch]) * crossfade
            )
    waveform = np.tile(one_cycle, (1, N_cycles))
    waveform_norm = waveform / AWG_OUTPUT_RANGE_V
    return (waveform_norm, states_v, state_names)


def check_strobe_dc_balance(
    best_pts,
    awg_range=AWG_OUTPUT_RANGE_V,
    cycle_freq_hz=STROBE_CYCLE_FREQ_HZ,
    sample_rate_hz=AWG_SR,
):
    """Find minimum n_idle_slots so IDLE fits within +-awg_range V.

    Returns (n_idle_slots, idle_per_slot, max_abs, message).
    """
    red_v = np.asarray(best_pts["red"]["awg_v"], dtype=np.float64)
    blue_v = np.asarray(best_pts["blue"]["awg_v"], dtype=np.float64)
    green_v = np.asarray(best_pts["green"]["awg_v"], dtype=np.float64)
    rgb_sum = red_v + blue_v + green_v
    spc_f = sample_rate_hz / cycle_freq_hz
    spc = int(round(spc_f))
    if not np.isclose(spc_f, spc, rtol=0, atol=1e-09):
        raise ValueError("Sample rate divided by cycle frequency must be an integer.")
    for n_idle in range(1, STROBE_MAX_IDLE_SLOTS + 1):
        total_slots = 3 + n_idle
        if spc % total_slots != 0:
            continue
        idle_per_slot = -rgb_sum / float(n_idle)
        all_v = np.concatenate([red_v, blue_v, green_v, idle_per_slot])
        max_abs = float(np.max(np.abs(all_v)))
        if max_abs <= awg_range + 1e-09:
            msg = f"{n_idle} idle slot(s); maximum excursion {max_abs:.3f} V."
            return (n_idle, idle_per_slot, max_abs, msg)
    raise RuntimeError(
        "Cannot find a valid n_idle_slots in 1..%d such that IDLE fits within +/-%.1f V. RGB sum max = %.3f V. Consider reducing AWG voltage targets."
        % (STROBE_MAX_IDLE_SLOTS, awg_range, float(np.max(np.abs(rgb_sum))))
    )


def get_input_trajectory(waveforms, t_array, delays_samples=None):
    n_ch, N = waveforms.shape
    t_array = np.asarray(t_array, dtype=np.float64)
    if delays_samples is None:
        delays_samples = np.zeros(n_ch)
    out = np.zeros((n_ch, len(t_array)))
    for ch in range(n_ch):
        te = (t_array - delays_samples[ch]) % N
        i0 = np.floor(te).astype(int) % N
        i1 = (i0 + 1) % N
        f = te - np.floor(te)
        out[ch] = (1 - f) * waveforms[ch, i0] + f * waveforms[ch, i1]
    return out
