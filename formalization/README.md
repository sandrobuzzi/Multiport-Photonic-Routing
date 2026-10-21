# Lean formalization of the ideal routing theorems

`TFLNRouter.lean` formalizes the ideal-case results in Supplementary Notes 1
and 2. Fixed arm phases and reflector phases are complex numbers on the unit
circle, and the round trip uses the transpose of the forward MMI matrix, as
required by reciprocity.

`ThreePortBalanced.lean` extends that development with the analytic
three-port power identities and the sharp balanced two-output limit.

`FourPortWalsh.lean` checks the simultaneous-routing, binary-Walsh,
four-port-family, and planar balanced-MMI claims added in Supplementary
Notes 2 and 7.

`SupplementaryGeometry.lean` formalizes the remaining unnumbered algebraic
proofs: the power-of-two routing-group obstruction, the exact three-port
Jacobian and critical curve, and the four-port section geometry and edge
phase-matching condition.

`ThreePortReachableRegion.lean` closes the global converse for the analytic
three-port region. It constructs physical reflector phases from every power
vector satisfying the exact triangle inequalities, including all degenerate
zero-power cases, and proves that saturation of those inequalities is exactly
the full closed critical-value curve printed in Supplementary Note 2.

The principal results are:

- `armPhase_controlFor`: the proposed reflector setting exactly cancels both
  traversals of each arbitrary fixed arm phase;
- `pairwise_routing`: for every nonzero port count, every fixed phase vector,
  and every input/output pair, the normalized round trip routes the input basis
  field to the selected output basis field, up to an arbitrary global phase;
- `reference_arm_control_is_one`: the global phase can be chosen so that arm
  zero has unit reflector phase, leaving only `N - 1` independent controls.
- `fixedPhase_reachableMatrices`, `fixedPhase_reachableFields`, and
  `fixedPhase_reachablePowers`: for every finite dimension and every MMI
  matrix, arbitrary fixed reciprocal diagonal phases preserve the complete
  reachable round-trip-matrix, field, and power sets;
- `reachablePowers_eq_referenceReachablePowers`: after quotienting the
  physically irrelevant common reflector phase, fixing any chosen reflector
  to one leaves the complete reachable power set unchanged.  Thus only
  `N - 1` relative phases are needed, not just at the routing vertices but for
  the whole power region;
- `general_pairwise_routing`: every column-unitary MMI whose relevant columns
  have equal entrywise magnitudes supports exact phase-only routing between
  every input/output pair, including arbitrary fixed arm phases;
- `pairMagnitude_pairwise_routing`: the corresponding route-specific theorem
  needs equal entrywise magnitudes only for the selected input and output
  columns.  It also covers rows in which both entries vanish and therefore
  does not impose the global nonzero-entry hypothesis used by the simpler
  ratio formula;
- `generalRoutingControl_normalizedDFT`: for the normalized `N`-point DFT, the
  arbitrary-MMI conjugated-column control law reduces exactly to the Fourier
  phase ramp in Supplementary Note 1;
- `dephased_three_classification`: an elementary formal classification of
  dephased `3 × 3` complex Hadamard matrices as three-point Fourier phase
  matrices;
- `equalSplitUnitary3_factorization`: every normalized equal-split unitary of
  order three factors into a three-point DFT and diagonal row/column phases;
- `threePort_reachable_region`: every such `3 × 3` MMI, with arbitrary fixed
  reciprocal phases, has exactly the complete reachable power region of a
  three-point DFT, up to a fixed output permutation;
- `threePort_reachable_region_of_mem_unitary`: the same result stated using
  Mathlib's standard unitary-group predicate; and
- `threePhaseOutput_zero`, `threePhaseOutput_one`, and
  `threePhaseOutput_two`: the explicit componentwise three-point DFT output
  formulas;
- `threePhaseOutput_correlation` and `threePhaseOutput_power_sum`: the exact
  cyclic field identity and power-conservation identity used to bound the
  reachable region;
- `threeDFT_reachable_balanced_leakage`: every reachable DFT state with two
  equal designated output powers has at least `1/9` power in the third output
  and at most `4/9` in either selected output;
- `threeDFT_balanced_witness_reachable`: the vector `(4/9, 4/9, 1/9)` is
  reachable for every input and every arbitrary fixed reciprocal phase
  vector, so the preceding bound is sharp;
- `threePort_balanced_leakage` and
  `threePort_balanced_witness_reachable`: the same sharp statements for every
  ideal equal-split unitary `3 × 3` MMI, obtained from full reachable-region
  universality; and
- `fourPort_reachable_regions_can_differ`: an explicit formal counterexample
  showing that full reachable-region universality fails already at `N = 4`.
  The normalized Walsh matrix reaches a two-output power vector that the
  normalized four-point DFT cannot reach, even though both matrices are
  equal-split and unitary.  `walshWitnessPower4_eq` verifies that the witness
  is exactly `(1/2, 1/2, 0, 0)`, and
  `fourPort_reachable_regions_can_differ_with_fixed` proves that the separation
  persists under arbitrary fixed reciprocal arm phases on either device.
- `generalRoundTrip_symmetric`: every reciprocal round trip
  `Mᵀ diag(r) M` is symmetric;
- `reachablePowers_input_independent`: for every equal-split MMI, the complete
  reachable power set is independent of the illuminated input, including
  arbitrary fixed reciprocal arm phases;
- `dft_simultaneous_routing`, `dftSimultaneousRoute_involution`, and
  `dftSimultaneousRoute_fixed_iff`: a single DFT phase ramp acts on all ports
  as the involution `n ↦ l - n`, with self-reflections exactly at `2n = l`;
- `walsh_simultaneous_routing` and `walsh_simultaneous_complete_matching`:
  for arbitrary binary dimension `k`, the Walsh control indexed by nonzero
  `a` implements `x ↦ x ⊕ a` simultaneously on every input, giving a complete
  reflection-free matching using only binary reflector phases;
- `walsh4_simultaneous_01_23`, `walsh4_simultaneous_02_13`, and
  `walsh4_simultaneous_03_12`: explicit matrix checks of all three complete
  four-port Walsh pairings;
- `fourPortFamily_columnUnitary`, `fourPortFamily_equalSplitNonzero`,
  `fourPortFamily_at_one`, and `fourPortFamily_at_i`: the full continuous
  four-port family is equal-split and unitary, with Walsh and DFT endpoints;
- `fourPortFamily_pairwise_routing` and
  `fourPortFamily_reachablePowers_input_independent`: every member supports
  every selected route and has an input-independent reachable power set; and
- `planar_balancedMMI_factorization`: exact verification of the stated
  two-stage balanced-`2 × 2`-MMI, shuffle, interstage-phase factorization of
  the continuous four-port family;
- `port_card_eq_two_pow_of_regular_involutive_action`: a finite regular
  abelian routing group whose elements are all involutions can act only on
  `2^k` ports;
- `three_jacobian_determinant`, `three_jacobian_factorization`, and
  `three_jacobian_zero_iff_critical_families`: the exact determinant,
  factorization, and complete three-family critical set quoted for the
  three-port DFT power map;
- `threePowerVector_on_critical_family`: substitution of a critical family
  gives the manuscript's full three-coordinate parametric curve;
- `threeDFT_reachablePowers_iff_triangleRegion`: for every input and every
  fixed reciprocal phase vector, the complete physical three-port reachable
  set is exactly the nonnegative unit-power region defined by the three
  magnitude-product triangle inequalities; the reverse implication includes
  an explicit complex-field and inverse-DFT phase construction;
- `triangleBoundary_iff_threeBoundaryVector`: saturation of at least one of
  those three exact inequalities is equivalent to membership in the full
  analytic cosine-parametrized boundary curve, proving the global boundary
  converse rather than only the Jacobian/critical-point necessity;
- `fourSectionK_bounds`, `fourSection_dft_full_region_bound`,
  `fourSection_walsh_lower_bounds`, and `central_section_family_bound`: the
  auxiliary-product interval and the DFT, Walsh, and central-section
  inequalities in Supplementary Note 7;
- `fourSection_reachable_iff_curveUnion`: the exact converse between the
  physical four-port phases and the fixed-height union of analytic section
  curves, including explicit reconstruction of the physical phases;
- `fourSection_reachable_iff_filledEllipseUnion`: for every family angle, the
  physical phase image is exactly the union of the filled elliptical disks,
  with the lower, upper, and central degenerate sections handled explicitly
  and the zero-sine family angles reduced to their limiting line segments;
- `fourPhasePower_section_coordinates`: the section formulas are derived
  from the actual complex round-trip transfer matrix, rather than assumed as
  an independent real parametrization; and
- `adjacent_edge_phase_matching_forces_sin_zero`: a nontrivial split on a
  non-opposite output edge forces `sin α = 0`, the obstruction distinguishing
  the Walsh endpoint from the rest of the continuous family; and
- `opposite_edge_thirteen_reachable`, `opposite_edge_twofour_reachable`,
  `walsh_adjacent_edge_reachable`, and `adjacent_edge_no_phase_match`:
  constructive witnesses for the two full opposite edges and the Walsh
  adjacent edges, together with the strict non-Walsh obstruction.

The three-port classification result does not assume the Hadamard/DFT
factorization as a premise:
the factorization is derived in Lean from the `1/√3` entry magnitudes and row
unitarity.  All five proof modules contain no `sorry`, `admit`, or additional
axioms.

Build the proof with Lean 4.19.0 and Mathlib 4.19.0:

```sh
lake update
lake build
```
