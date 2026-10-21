# Multiport photonic routing with a single cavity-based electro-optic phase shifter per port

Analysis software and formal proofs accompanying the manuscript.

The repository is private during manuscript preparation. It contains software,
formal proofs and documentation. Measurement data and saved numerical results
are obtained from [Zenodo, DOI 10.5281/zenodo.23194529](https://doi.org/10.5281/zenodo.23194529).
Instrument control is outside the scope of this offline analysis repository.

## Contents

| Directory | Contents |
|---|---|
| `workflow/` | Multisine exploration, targeted trajectories, programmed strobe, delay alignment, cube reconstruction, reciprocal transfer-matrix fitting and routing-state selection. |
| `figures/main/` | Numerical panels for Figs. 1, 3 and 4, including the voltage cubes, routing plane and switching traces. |
| `figures/supplementary/` | Cavity-state reconstruction, reference spectrum, Lorentzian fit, insertion-loss budget and targeted extinction analysis. |
| `simulations/` | Ideal reachable regions, four-port comparisons, loss imbalance, fitted-device variations and joint-drive geometry. |
| `formalization/` | Lean 4/Mathlib sources for the routing theorems and supplementary geometry. |
| `notebooks/` | Main-figure source-data checks; outputs are cleared. |
| `viewer/` | Interactive voltage-cube viewer; data are generated locally. |
| `scripts/` | Zenodo downloader, model-parameter preparation and viewer-data export. |
| `config/` | Required data paths and verified archive coverage. |
| `tests/` | Numerical invariants, waveform validation and safe archive extraction. |

## Setup

Use Python 3.11 or newer. From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Download the data before running figure or fitted-model analyses. For a public
record, use `python scripts/download_data.py`.

The reserved Zenodo record is currently a draft. Until publication, an authorized
reviewer can download its ZIP from the review page and import it locally:

```sh
python scripts/download_data.py --archive "INSERT PATH HERE FOR THE ZENODO DATA ZIP"
```

Alternatively set `ZENODO_REVIEW_URL` locally to the authorized review link and
run the downloader. Never commit or share that environment value. Once the record
is public, the default downloader uses the public record API and verifies its
checksum. The ZIP importer extracts data files only, so historical code in a
Zenodo archive does not replace this repository's code.

After importing the data, run `python scripts/prepare_model_parameters.py`.
Data go into ignored `data/`; generated results go into ignored `outputs/`.
Set `TFLN_ROUTER_DATA` or `TFLN_ROUTER_OUTPUT` to use other directories.
Only open NPZ files from a trusted source: legacy acquisition metadata use NumPy
object arrays.

## Reproduce figures and the saved fit

Run commands from the repository root. Common examples:

```sh
python figures/main/plot_cavity_reflection.py
python figures/main/plot_reachable_region.py
python figures/main/plot_reflection_transmission.py
python figures/main/plot_resonance_tuning.py
python figures/main/fit_bandwidth.py \
  data/fig3/fig3c/a46_bandwidth_13dbm_2605111807.s2p \
  data/fig3/fig3c/through_meas_2605111715.s2p \
  --output-prefix outputs/figures/main/bandwidth_fit
python figures/main/plot_voltage_cubes.py
python figures/main/plot_voltage_plane.py
python figures/main/plot_routing_traces.py
python figures/supplementary/plot_cavity_states.py
python figures/supplementary/plot_reference_lorentzian.py
```

The saved-fit and waveform commands are described in [workflow/README.md](workflow/README.md).
The archive contains the reconstructed exploration cube, saved reciprocal fit and
strobe capture. Evaluating the saved parameters reproduces the archived model;
a new nonlinear fit may converge to a different optimum. No hardware is required
for the supplied offline workflows.

The Fig. 4 output convention is **P1/CH2 red, P2/CH3 green, P3/CH4 blue**.
The original model's internal output order is CH2/CH4/CH3, converted to the
manuscript order using `[0, 2, 1]`. Programmed voltage traces are reconstructed
from the acquisition settings; they are not electrical measurements at the device.
The figure scripts export numerical panels; the final Illustrator composites,
microscopy and design schematics are not assembled by these scripts. Rendering
can vary with installed fonts; no proprietary font is distributed.

### Data inputs

The accompanying Zenodo archive supplies the inputs listed in
[config/data_manifest.json](config/data_manifest.json). Fig. 3a and the
supplementary reference-cavity plots use final processed coordinates.
These plotting scripts read the supplied values directly. Original measurements and
preprocessing are retained in the private research archive.

The phase-sensitivity script plots a stored rounded tolerance reference;
it does not recompute that separate optimization. Model parameters are
reconstructed locally from saved fits by `prepare_model_parameters.py`.

## Simulations, proofs and viewer

Analytic and numerical simulation entry points are in `simulations/`. Some
Monte Carlo and nonlinear optimization runs can take substantial time. They
write results under `outputs/`; rerunning them is a software calculation and
requires no new measurements. Fitted-device simulations require the prepared
model parameters.

For the Lean proofs, install Lean via elan and run `lake build` inside
`formalization/`. The toolchain and Mathlib release are pinned to v4.19.0;
dependencies are downloaded by Lake. See [formalization/README.md](formalization/README.md).

For the interactive viewer:

```sh
python scripts/export_cube_viewer_data.py
python -m http.server 8000 --bind 127.0.0.1
```

Then open `http://localhost:8000/viewer/` in a browser. The viewer loads Plotly 2.35.2 from its public CDN.
All Python analyses run independently of the viewer. The notebook in `notebooks/` redraws
source data for checking rather than reproducing the publication layout.

## Provenance, license and citation

The acquisition/model workflow was extracted from
[abuzzimit/AutomaticGratingSetup](https://github.com/abuzzimit/AutomaticGratingSetup),
commit `fd57583980fc8134d7ce966fe45b1883bd50f261`.
Source checksums and adaptation notes are in `workflow/provenance.json`.
Saved model conventions are retained, with the display conversion documented
above. Fit exports include their voltage and detector permutations. The global
fitting fallback uses the same ordering as the preceding local fit.

The code is distributed under CC BY 4.0, consistent with the existing manuscript
code deposit; see [LICENSE](LICENSE). Please cite the manuscript and the Zenodo
data record. Machine-readable citation metadata are in [CITATION.cff](CITATION.cff).

For development checks, install `requirements-dev.txt` and run
`python -m ruff check . --select F`. Viewer numerical checks use
`node --test tests/test_viewer.cjs` (Node.js 22 in CI).
