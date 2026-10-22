# Verification

## Numerical and source checks

- The originally downloaded Zenodo draft archive matched its advertised MD5
  `eb1ab65f55cde61bedbbd477f5dcbc4f`. A replacement archive with corrected
  Fig. 3b data and supporting Supplementary inputs has been prepared and
  checked locally; upload verification remains pending.
- Main-figure scripts, the bandwidth fit, source-data notebook, cavity-state
  reconstruction, reference Lorentzian, joint-drive geometry and four-port
  regions run against the downloaded data.
- Evaluating the saved reciprocal model reproduces its physical-power and
  calibrated-detector cubes exactly. The measured routing-state selection
  reproduces the saved strobe voltage tuples.
- Multisine, targeted and valid-setting strobe arrays and delay-aware
  interpolation match the source acquisition implementation exactly.
- The fitting objective at the saved parameters matches the source code.
  A bounded integration check starting from those parameters completes, and
  its newly exported fit can be evaluated again with identical cube values.
- The cable-delay optimizer recovers known synthetic input/output delays to
  within 1 ps. The operating-point reporting CLI runs on the archived fit.
- Ten Python tests check numerical invariants, input validation, data-only ZIP
  extraction, permutation-aware fallback and fit-export round trips.
- Two JavaScript tests check invariant colors under viewer subsampling and
  finite colors for constant channels. JavaScript syntax also passes.
- Python static checks report no undefined names, unused imports or unused
  local assignments. Dead diagnostic branches and duplicate implementations
  have been removed.
- All five Lean modules build using pinned Lean/Mathlib v4.19.0. Existing Lean
  style and unused-variable linter warnings do not prevent the proof build.
- Supplementary analyses requiring the missing measurement files run against
  a separately staged data addition. The code repository contains no datasets.

## Fresh replacement-archive check (21 October 2026)

- All 14 figure/model-analysis entry points ran successfully against a fresh
  extraction of the prepared replacement ZIP.
- Both the code notebook and the archive-root notebook executed every code cell.
- All four device-statistics scripts ran against the included tables.
- The final processed Fig. 3a columns plot directly; the Fig. 3b notebook uses
  the independently extracted points and linear fit.
- The standalone Git history contains no measurement datasets, private paths,
  credentials, tokenized review links or the superseded Fig. 3a scaling code.
- Remote Zenodo attachment verification remains pending.

## Reproduction limits

- Use the accompanying data archive for the supplementary inputs listed in
  the data manifest. Reference-cavity plotting uses final processed values;
  it does not reproduce private measurement preprocessing.
- The phase-sensitivity plot reads its rounded tolerance-optimized marker
  from stored reference data. The separate tolerance optimization is not
  recomputed by the supplied script.
- A full cold multistart refit and all long Monte Carlo/sensitivity
  optimizations have not been rerun for this release. New nonlinear fits
  are not asserted to recover identical parameters.
- Instrument control and the browser interface are outside this numerical
  audit. The viewer's export format and color calculations are checked.
