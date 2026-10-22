#!/usr/bin/env python3
"""Independently extract all 15 forward A46/K3 resonances from raw spectra.

Usage: python extract_corrected_forward_K3.py --source RAW.npz --output DATA.csv
The exported frequency detunings are already in GHz and plot directly.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import median_abs_deviation
LAMBDA0 = {"K1": 1541.59, "K2": 1541.87, "K3": 1541.56}

def spectral_fit(w, y, seed, half_pm=90, degree=1, start=None, starts=2,
                 weight_mode="noise"):
    sel = np.abs(w-seed) <= half_pm / 1000
    x = (w[sel]-seed) * 1000
    yy = y[:, sel].T
    if len(x) < 9 or not np.isfinite(yy).all():
        return {"valid": False, "reason": "insufficient or nonfinite spectral samples"}
    sigma = median_abs_deviation(np.diff(yy, n=2, axis=0), axis=0, scale="normal") / np.sqrt(6)
    sigma = np.maximum(sigma, np.maximum(np.ptp(yy, axis=0) * 1e-6, 1e-12))
    noise_sigma = sigma.copy()
    if weight_mode == "ptp":
        sigma = np.maximum(np.ptp(yy, axis=0), 1e-12)
    elif weight_mode == "equal":
        sigma = np.full(yy.shape[1], np.median(sigma))
    scaled = yy / sigma

    def model(par, full=False):
        center_pm, width_pm = par
        u = 2*(x-center_pm)/width_pm
        columns = [(x/half_pm)**k for k in range(degree+1)] + [1/(1+u*u), u/(1+u*u)]
        a = np.column_stack(columns)
        coeff = np.linalg.lstsq(a, scaled, rcond=None)[0]
        prediction = a @ coeff
        return (prediction, coeff, a) if full else prediction

    initial = [0., 32.] if start is None else [(start[0]-seed)*1000, start[1]*1000]
    initial = np.clip(initial, [-64., 4.], [64., 140.])
    guesses = [initial]
    if starts > 1:
        guesses += [[np.clip(initial[0]+18, -64, 64), 25.],
                    [np.clip(initial[0]-18, -64, 64), 50.]]
    fits = [least_squares(lambda p: (model(p)-scaled).ravel(), guess,
                          bounds=([-65., 3.], [65., 150.]),
                          x_scale=[20., 30.], max_nfev=100,
                          ftol=1e-9, xtol=1e-9, gtol=1e-9) for guess in guesses]
    best = min(fits, key=lambda r: np.sum(r.fun*r.fun))
    pred, coeff, a = model(best.x, full=True)
    residual = (pred-scaled)*sigma
    ptp = np.maximum(np.ptp(yy, axis=0), 1e-12)
    nrmse = np.sqrt(np.mean(residual**2, axis=0))/ptp
    dof = max(1, yy.size - (2 + yy.shape[1]*(degree+3)))
    cov = np.linalg.pinv(best.jac.T @ best.jac) * np.sum(best.fun**2)/dof
    resonant = a[:, -2:] @ coeff[-2:]
    snr = np.ptp(resonant, axis=0)
    center = seed+best.x[0]/1000
    width = best.x[1]/1000
    return {"valid": bool(best.success and not np.any(best.active_mask)),
            "center_nm": float(center), "width_nm": float(width),
            "center_formal_se_pm": float(np.sqrt(max(cov[0, 0], 0))),
            "median_channel_nrmse": float(np.median(nrmse)),
            "channel_nrmse": nrmse.tolist(), "channel_feature_snr": snr.tolist(),
            "edge_distance_in_fwhm": float(min(center-w[0], w[-1]-center)/width),
            "samples_per_fwhm": float(width/np.median(np.diff(w))),
            "active_bounds": best.active_mask.tolist(), "cost": float(np.mean(best.fun**2)),
            "noise_sigma": noise_sigma.tolist(), "weight_scale": sigma.tolist(),
            "coefficients_raw": (coeff*sigma).tolist(),
            "seed_nm": float(seed), "half_window_pm": half_pm, "background_degree": degree,
            "weight_mode": weight_mode, "nfev": best.nfev}

def choose_seed(w, y, cavity, voltage):
    expected = LAMBDA0[cavity]+0.015*voltage
    sel = np.abs(w-expected) < 0.24
    idx = np.flatnonzero(sel)
    if len(idx) < 12:
        return float(np.clip(expected, w[0]+0.02, w[-1]-0.02))
    local = y[:, sel]
    spans = np.maximum(np.percentile(local, 98, axis=1)-np.percentile(local, 2, axis=1), 1e-12)
    window = min(11, len(idx)//2*2-1)
    gradient = savgol_filter(local/spans[:, None], window, 2, deriv=1, axis=1)
    score = np.sqrt(np.sum(gradient**2, axis=0))
    peaks, _ = find_peaks(score, distance=max(2, int(.012/np.median(np.diff(w)))))
    if len(peaks) == 0:
        p = int(np.argmax(score))
    else:
        merit = score[peaks]*np.exp(-0.5*((w[idx[peaks]]-expected)/0.16)**2)
        p = int(peaks[np.argmax(merit)])
    return float(w[idx[p]])

def linear_metrics(v, lam, se_pm):
    if not np.isfinite(lam).all() or len(v) < 2 or np.ptp(v) == 0:
        return {}
    y = 299792458/lam
    vmean = v.mean(); x = v-vmean
    a = np.column_stack([np.ones(len(v)), x])
    par = np.linalg.lstsq(a, y, rcond=None)[0]
    pred = a @ par; residual = y-pred
    n, p = a.shape; df = n-p
    inv = np.linalg.pinv(a.T @ a)
    standard = np.sqrt(max(0, inv[1, 1]*np.sum(residual**2)/df)) if df else np.nan
    hat = np.sum((a @ inv)*a, axis=1)
    hc3 = inv @ (a.T @ (a*(residual/np.maximum(1-hat, 1e-8))[:, None]**2)) @ inv
    rmse = np.sqrt(np.mean(residual**2))
    span = np.ptp(y)
    quadratic = np.linalg.lstsq(np.column_stack([a, x*x]), y, rcond=None)[0]
    cubic = np.linalg.lstsq(np.column_stack([a, x*x, x*x*x]), y, rcond=None)[0]
    adjacent = adjacent_cluster_sensitivity(a, residual, v)
    return {"slope_magnitude_GHz_V": float(abs(par[1])), "signed_slope_GHz_V": float(par[1]),
            "formal_OLS_se_GHz_V": float(standard),
            "HC3_se_GHz_V": float(np.sqrt(max(0, hc3[1, 1]))) if df else None,
            "R2": float(1-np.sum(residual**2)/np.sum((y-y.mean())**2)) if span else 0.,
            "frequency_residual_rms_GHz": float(rmse),
            "frequency_residual_max_GHz": float(max(abs(residual))),
            "relative_residual_rms_percent": float(100*rmse/span) if span else None,
            "quadratic_coefficient_GHz_V2": float(quadratic[-1]),
            "cubic_coefficient_GHz_V3": float(cubic[-1]),
            "residual_GHz": residual.tolist(), "frequency_GHz": y.tolist(),
            "predicted_frequency_GHz": pred.tolist(),
            "median_center_formal_se_pm": float(np.nanmedian(se_pm)),
            "point_count": len(v), "unique_voltage_count": len(np.unique(np.round(v, 7))),
            "adjacent_voltage_cluster_sensitivity": adjacent}

def adjacent_cluster_sensitivity(a, residual, voltage):
    """Sensitivity to covariance shared by 2-4 neighboring voltage settings.

    These descriptive SEs are not calibrated confidence intervals for a
    deterministic nonlinear response. All points enter every estimate.
    """
    unique = np.unique(np.round(voltage, 7))
    if len(unique) < 8:
        return {}
    rank = np.searchsorted(unique, np.round(voltage, 7))
    inv = np.linalg.pinv(a.T @ a)
    n, p = a.shape
    result = {}
    for block in [1, 2, 3, 4]:
        values = []
        for offset in range(block):
            groups = (rank+offset)//block
            scores = np.array([a[groups==g].T @ residual[groups==g]
                               for g in np.unique(groups)])
            ng = len(scores)
            cov = inv @ scores.T @ scores @ inv * ng/(ng-1)*(n-1)/(n-p)
            values.append(float(np.sqrt(max(0, cov[-1, -1]))))
        result[str(block)] = values
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    z=np.load(args.source,allow_pickle=True)
    w=z['wavelengths'];v=z['Pm14_K3_voltage_combinations'][:15,2];spectra=z['Pm14_K3_spectra'][:15]
    fits=[spectral_fit(w,spectra[i],choose_seed(w,spectra[i],'K3',voltage)) for i,voltage in enumerate(v)]
    assert len(fits)==15 and all(f['valid'] for f in fits)
    center=np.array([f['center_nm'] for f in fits]);se=np.array([f['center_formal_se_pm'] for f in fits])
    metrics=linear_metrics(v,center,se);frequency=299792458/center;shift=frequency-frequency[np.flatnonzero(np.isclose(v,0))[0]]
    slope,intercept=np.polyfit(v,shift,1)
    rows=[dict(voltage_v=float(v[i]),measured_resonance_wavelength_nm=float(center[i]),measured_frequency_detuning_ghz=float(shift[i]),linear_fit_frequency_detuning_ghz=float(intercept+slope*v[i]),spectral_fit_valid=True,center_standard_error_pm=float(se[i])) for i in range(15)]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    metadata=dict(device='A46',cavity='K3',laser_output_dbm=-14,voltage_direction='increasing',point_count=15,valid_points=15,source_filename=args.source.name,source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),extraction='Independent shared-center/linewidth fits of three simultaneous detector spectra at each voltage, linear backgrounds and Lorentzian/dispersive terms; wavelength-based initialization; no voltage-trajectory smoothing or substitutions.',frequency_units='GHz',frequency_reference='Independently fitted resonance at 0 V',tuning_GHz_V=abs(float(slope)),formal_OLS_slope_SE_GHz_V=metrics['formal_OLS_se_GHz_V'],reported_practical_uncertainty_GHz_V=0.06,uncertainty_scope='Descriptive full-range fit uncertainty allowing correlation across 2-4 adjacent voltage settings; not a calibrated confidence interval.',adjacent_voltage_cluster_sensitivity=metrics['adjacent_voltage_cluster_sensitivity'],last_two_wavelengths_nm=center[-2:].tolist())
    args.output.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(f"15/15 independent spectral fits; slope {abs(slope):.9f} GHz/V; formal SE {metrics['formal_OLS_se_GHz_V']:.9f} GHz/V")

if __name__=='__main__':main()
