# Offline voltage-cube workflow

Code for generating the Fig. 4 exploration waveforms, reconstructing saved
measurements, fitting the reciprocal model, selecting measured routing states,
and reconstructing the programmed strobe.

## Files

| File | Purpose |
|---|---|
| `waveforms.py` | Multisine, targeted-trajectory and DC-balanced strobe generation. |
| `delay_calibration.py` | Cable-delay estimation from a saved multi-drive calibration capture. |
| `routing.py` | Routing-state selection from the reconstructed measured cube. |
| `model_v1.py` | Reconstruction, reciprocal model, calibration and numerical fit. |
| `run_saved_result_with_3d_v2.py` | Shared multistart refinement helpers. |
| `run_saved_result_with_3d_v3.py` | Model operating-point and balance optimization with a CLI. |
| `run_saved_result_with_3d_v4.py` | Voltage/detector permutation search and fit orchestration. |
| `run_workflow.py` | Offline command-line entry point. |
| `example_paths.py` | Clearly marked input/output path placeholders. |
| `provenance.json` | Original repository, commit and source checksums. |

Requires Python 3.11 or newer, NumPy, SciPy and Matplotlib. The repository root includes the pinned environment in `requirements.txt`.

## Paths

Replace each `INSERT PATH HERE` string with the corresponding file or folder.
Paths containing spaces must be quoted. Output paths must be new; existing
files are not overwritten. No instrument connection is required.

Run these commands from this directory:

```sh
python run_workflow.py multisine --output "INSERT PATH HERE FOR GENERATED MULTISINE.npz"

python run_workflow.py reconstruct \
  --raw "INSERT PATH HERE FOR THE RAW CUBE CAPTURE.npz" \
  --output "INSERT PATH HERE FOR THE RECONSTRUCTED CUBE.npz"

python run_workflow.py targets \
  --reconstructed "INSERT PATH HERE FOR THE RECONSTRUCTED CUBE.npz" \
  --output "INSERT PATH HERE FOR SELECTED ROUTING STATES.json"

python run_workflow.py evaluate \
  --fit "INSERT PATH HERE FOR THE SAVED FIT PARAMETERS.npz" \
  --reconstructed "INSERT PATH HERE FOR THE RECONSTRUCTED CUBE.npz" \
  --output "INSERT PATH HERE FOR EVALUATED MODEL CUBES.npz"

python run_workflow.py strobe \
  --capture "INSERT PATH HERE FOR THE STROBE CAPTURE.npz" \
  --output "INSERT PATH HERE FOR THE PROGRAMMED STROBE.npz"

python run_workflow.py fit \
  --reconstructed "INSERT PATH HERE FOR THE RECONSTRUCTED CUBE.npz" \
  --initial-fit "INSERT PATH HERE FOR INITIAL FIT PARAMETERS.npz" \
  --output "INSERT PATH HERE FOR A NEW FIT OUTPUT DIRECTORY"
```

For fitting directly from the raw scan, replace `--reconstructed` with `--raw`.
`--initial-fit` is optional and can be repeated. With no supplied initial fit,
the original built-in reference and permutation-search starts are used. The
code does not search lab directories for previous fits.

## Measurement settings

The Fig. 4 multisine defaults are a 25-us period, 2.4-GSa/s sampling,
10–50-MHz synthesis bins and seed 42. Orthogonalization is followed by uniform
rank remapping. The remapping slightly changes the spectrum and interchannel
correlation; `compute_spectral_quality` evaluates the final waveform.

Raw reconstruction requires the acquisition's saved frequency, seed, bias,
amplitude and delay metadata.

The strobe is reconstructed from its saved voltage tuples, cycle frequency
and number of idle slots. The paper uses two idle slots and a 0.7 crossfade
fraction. It is a programmed waveform reconstruction, not a direct electrical
measurement at the device.

## Channel conventions

Measured CH2, CH3 and CH4 are manuscript P1, P2 and P3, shown in red, green
and blue. Routing-state selection uses this order.

The reciprocal fit retains the original internal port labels. Its saved
voltage permutation maps physical electrode order into model order. For the
paper fit, the model axes are K3/K2/K1 and the fitted detector order is
CH2/CH4/CH3. The manuscript output order is obtained from model components
`[0, 2, 1]`. `evaluate` retains the stored internal ordering so its arrays can
be compared directly with the archived fitted cubes.

## Verification

The cleaned model reproduces the archived fitted cubes exactly when evaluated
with the saved parameters. Measured-state selection reproduces the three
saved voltage tuples exactly. Multisine and valid-setting strobe generation
are compared with the original source. Integer samples per cycle are checked
before strobe generation.

A new nonlinear fit can converge to a different optimum when its initial
parameters or configuration differ. The archived fit is the reference for
reproducing the published model; a cold refit is not asserted to be identical.

Fit outputs store `applied_voltage_perm` and `applied_detector_perm` in both
`fit_params.npz` and `full_cube_fit.npz`. Reports label the physical electrodes
and manuscript outputs in that stored order. New fits do not change the
archived paper fit. The `reconstruct` command accepts uniform multisine scans;
targeted captures use the separate supplementary analysis.

To calculate model-based channel maxima, balanced operating points and bias
centering suggestions, run from this directory:

```sh
python run_saved_result_with_3d_v3.py \
  --fit "INSERT PATH HERE FOR THE FULL SAVED FIT.npz" \
  --reconstructed "INSERT PATH HERE FOR THE RECONSTRUCTED CUBE.npz" \
  --output "INSERT PATH HERE FOR A NEW OPERATING POINT REPORT.txt"
```

These modeled optima are separate from the measured-extinction routing states
selected by `run_workflow.py targets`. Report axes are explicitly labeled with
their physical electrode order.
