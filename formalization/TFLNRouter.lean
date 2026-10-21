import Mathlib

/-!
# Pair-wise routing through a DFT MMI with fixed arm phases

This file formalizes the ideal-case routing theorem used in Supplementary
Note 1.  Ports and arms are indexed by `ZMod N`, so all index arithmetic is
automatically modulo `N`.
-/

namespace TFLNRouter

open Complex
open ComplexConjugate

open scoped ZMod

variable {N : ℕ} [NeZero N]

/-- The unit-amplitude field concentrated at port `n`. -/
def basis (n : ZMod N) : ZMod N → ℂ := fun j ↦ if j = n then 1 else 0

/-- The phase accumulated in one arm over the outward fixed section, the
reflector, and the return through the same fixed section. -/
noncomputable def armPhase (fixed control : ZMod N → Circle) (k : ZMod N) : Circle :=
  fixed k * control k * fixed k

/-- Reflector settings which cancel both passes through the fixed phases and
leave the DFT phase ramp indexed by `l`. `global` is a physically irrelevant
common output phase. -/
noncomputable def controlFor (fixed : ZMod N → Circle) (global : Circle)
    (l : ZMod N) : ZMod N → Circle := fun k ↦
  (fixed k)⁻¹ * (fixed k)⁻¹ * global * ZMod.toCircle (k * l)

/-- The normalized round-trip field: an unnormalized DFT, diagonal arm
phases, a second unnormalized DFT, and the normalization `1/N`. -/
noncomputable def roundTrip (fixed control : ZMod N → Circle)
    (input : ZMod N → ℂ) : ZMod N → ℂ := fun q ↦
  (N : ℂ)⁻¹ * ZMod.dft
    (fun k ↦ (armPhase fixed control k : ℂ) * ZMod.dft input k) q

lemma armPhase_controlFor (fixed : ZMod N → Circle) (global : Circle)
    (l k : ZMod N) :
    armPhase fixed (controlFor fixed global l) k =
      global * ZMod.toCircle (k * l) := by
  simp only [armPhase, controlFor]
  calc
    fixed k * ((fixed k)⁻¹ * (fixed k)⁻¹ * global * ZMod.toCircle (k * l)) * fixed k =
        (fixed k * (fixed k)⁻¹) * ((fixed k)⁻¹ * fixed k) *
          (global * ZMod.toCircle (k * l)) := by ac_rfl
    _ = global * ZMod.toCircle (k * l) := by simp

lemma dft_basis (n k : ZMod N) :
    ZMod.dft (basis n) k = ZMod.stdAddChar (-(n * k)) := by
  simp [ZMod.dft_apply, basis]

lemma compensated_dft_basis (fixed : ZMod N → Circle) (global : Circle)
    (n l : ZMod N) :
    (fun k ↦ (armPhase fixed (controlFor fixed global l) k : ℂ) *
      ZMod.dft (basis n) k) =
    (fun k ↦ (global : ℂ) * ZMod.dft (basis (n - l)) k) := by
  funext k
  rw [armPhase_controlFor, dft_basis, dft_basis]
  simp only [Circle.coe_mul, ZMod.stdAddChar_apply]
  norm_cast
  rw [mul_assoc]
  congr 1
  rw [← ZMod.toCircle.map_add_eq_mul]
  congr 1
  ring

/-- **Complete pair-wise routing.** For every number of ports, every set of
fixed unit-modulus arm phases, and every input/output pair `(n,m)`, the
constructed pure-phase reflector settings route the input exactly to `m`, up
to the freely chosen common phase `global`. -/
theorem pairwise_routing (fixed : ZMod N → Circle) (global : Circle)
    (n m : ZMod N) :
    roundTrip fixed (controlFor fixed global (m + n)) (basis n) =
      fun q ↦ (global : ℂ) * basis m q := by
  funext q
  rw [roundTrip]
  rw [compensated_dft_basis]
  rw [ZMod.dft_const_mul]
  rw [show n - (m + n) = -m by ring]
  simp only
  rw [congrFun (ZMod.dft_dft (basis (-m))) q]
  by_cases hq : q = m
  · subst q
    simp only [basis, if_pos, smul_eq_mul, mul_one]
    calc
      (N : ℂ)⁻¹ * ((global : ℂ) * N) =
          (global : ℂ) * ((N : ℂ)⁻¹ * N) := by ring
      _ = (global : ℂ) := by rw [inv_mul_cancel₀ (NeZero.ne (N : ℂ)), mul_one]
  · simp [basis, hq, NeZero.ne (N : ℂ)]

/-- The common phase may be selected so that arm zero needs no tunable phase.
Thus only `N - 1` independently tunable reflectors are required. -/
theorem reference_arm_control_is_one (fixed : ZMod N → Circle) (l : ZMod N) :
    controlFor fixed (fixed 0 * fixed 0) l 0 = 1 := by
  simp only [controlFor, zero_mul]
  rw [AddChar.map_zero_eq_one]
  change (fixed 0)⁻¹ * (fixed 0)⁻¹ * (fixed 0 * fixed 0) * 1 = 1
  rw [mul_one]
  calc
    (fixed 0)⁻¹ * (fixed 0)⁻¹ * (fixed 0 * fixed 0) =
        ((fixed 0)⁻¹ * fixed 0) * ((fixed 0)⁻¹ * fixed 0) := by ac_rfl
    _ = 1 := by simp

/-! ## General finite-dimensional results -/

section General

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- Generic two-pass diagonal arm phase. -/
noncomputable def generalArmPhase (fixed control : ι → Circle) (j : ι) : Circle :=
  fixed j * control j * fixed j

/-- The round-trip output amplitude written componentwise.  `M j q` is the
MMI amplitude between arm `j` and external port `q`; `fixed` occurs on both
passes and `control` is the diagonal reflector phase. -/
noncomputable def generalOutput (M : Matrix ι ι ℂ) (fixed control : ι → Circle)
    (n q : ι) : ℂ :=
  ∑ j, M j q * (generalArmPhase fixed control j : ℂ) * M j n

/-- The complete round-trip matrix, with external output index first and
external input index second. -/
noncomputable def generalRoundTrip (M : Matrix ι ι ℂ)
    (fixed control : ι → Circle) : Matrix ι ι ℂ :=
  fun q n ↦ generalOutput M fixed control n q

/-- The complete set of round-trip matrices reachable by varying all
phase-only reflectors. -/
noncomputable def reachableMatrices (M : Matrix ι ι ℂ) (fixed : ι → Circle) :
    Set (Matrix ι ι ℂ) :=
  {T | ∃ control : ι → Circle, T = generalRoundTrip M fixed control}

/-- The complete set of reachable complex output fields from input `n`. -/
noncomputable def reachableFields (M : Matrix ι ι ℂ) (fixed : ι → Circle)
    (n : ι) : Set (ι → ℂ) :=
  {z | ∃ control : ι → Circle, z = fun q ↦ generalOutput M fixed control n q}

/-- Componentwise output power. -/
noncomputable def fieldPower (z : ι → ℂ) : ι → ℝ := fun q ↦ ‖z q‖ ^ 2

/-- The complete set of reachable output-power vectors from input `n`. -/
noncomputable def reachablePowers (M : Matrix ι ι ℂ) (fixed : ι → Circle)
    (n : ι) : Set (ι → ℝ) :=
  fieldPower '' reachableFields M fixed n

/-- Absorb both traversals of the fixed phase into the reflector control. -/
noncomputable def effectiveControl (fixed control : ι → Circle) : ι → Circle :=
  fun j ↦ fixed j * fixed j * control j

/-- Recover the physical reflector setting from a desired effective phase. -/
noncomputable def physicalControl (fixed effective : ι → Circle) : ι → Circle :=
  fun j ↦ (fixed j)⁻¹ * (fixed j)⁻¹ * effective j

lemma armPhase_eq_effectiveControl (fixed control : ι → Circle) (j : ι) :
    generalArmPhase fixed control j = effectiveControl fixed control j := by
  simp only [generalArmPhase, effectiveControl]
  ac_rfl

lemma armPhase_physicalControl (fixed effective : ι → Circle) (j : ι) :
    generalArmPhase fixed (physicalControl fixed effective) j = effective j := by
  simp only [generalArmPhase, physicalControl]
  calc
    fixed j * ((fixed j)⁻¹ * (fixed j)⁻¹ * effective j) * fixed j =
        (fixed j * (fixed j)⁻¹) * ((fixed j)⁻¹ * fixed j) * effective j := by ac_rfl
    _ = effective j := by simp

lemma generalOutput_fixed_eq_effective (M : Matrix ι ι ℂ)
    (fixed control : ι → Circle) (n : ι) :
    (fun q ↦ generalOutput M fixed control n q) =
      (fun q ↦ generalOutput M (fun _ ↦ 1) (effectiveControl fixed control) n q) := by
  funext q
  simp only [generalOutput]
  apply Finset.sum_congr rfl
  intro j _
  rw [armPhase_eq_effectiveControl]
  simp [generalArmPhase]

lemma generalOutput_physical_eq_effective (M : Matrix ι ι ℂ)
    (fixed effective : ι → Circle) (n : ι) :
    (fun q ↦ generalOutput M fixed (physicalControl fixed effective) n q) =
      (fun q ↦ generalOutput M (fun _ ↦ 1) effective n q) := by
  funext q
  simp only [generalOutput]
  apply Finset.sum_congr rfl
  intro j _
  rw [armPhase_physicalControl]
  simp [generalArmPhase]

/-- **Fixed-phase invariance.** Arbitrary reciprocal diagonal phases preserve
the entire set of reachable complex fields, in every finite dimension and for
every MMI matrix. -/
theorem fixedPhase_reachableFields (M : Matrix ι ι ℂ) (fixed : ι → Circle)
    (n : ι) :
    reachableFields M fixed n = reachableFields M (fun _ ↦ 1) n := by
  ext z
  constructor
  · rintro ⟨control, rfl⟩
    exact ⟨effectiveControl fixed control,
      generalOutput_fixed_eq_effective M fixed control n⟩
  · rintro ⟨effective, rfl⟩
    exact ⟨physicalControl fixed effective,
      (generalOutput_physical_eq_effective M fixed effective n).symm⟩

/-- Fixed phases therefore preserve the complete reachable power region. -/
theorem fixedPhase_reachablePowers (M : Matrix ι ι ℂ) (fixed : ι → Circle)
    (n : ι) :
    reachablePowers M fixed n = reachablePowers M (fun _ ↦ 1) n := by
  simp only [reachablePowers, fixedPhase_reachableFields]

/-- Fixed reciprocal phases preserve not only the outputs of one input, but
the entire set of reachable round-trip matrices. -/
theorem fixedPhase_reachableMatrices (M : Matrix ι ι ℂ) (fixed : ι → Circle) :
    reachableMatrices M fixed = reachableMatrices M (fun _ ↦ 1) := by
  ext T
  constructor
  · rintro ⟨control, rfl⟩
    refine ⟨effectiveControl fixed control, ?_⟩
    ext q n
    exact congrFun (generalOutput_fixed_eq_effective M fixed control n) q
  · rintro ⟨effective, rfl⟩
    refine ⟨physicalControl fixed effective, ?_⟩
    ext q n
    exact (congrFun (generalOutput_physical_eq_effective M fixed effective n) q).symm

/-! ### Removing the physically irrelevant common reflector phase -/

/-- Normalize a reflector vector so that the selected reference arm is one. -/
noncomputable def referenceNormalizedControl (ref : ι)
    (control : ι → Circle) : ι → Circle :=
  fun j ↦ (control ref)⁻¹ * control j

lemma referenceNormalizedControl_at_ref (ref : ι) (control : ι → Circle) :
    referenceNormalizedControl ref control ref = 1 := by
  simp [referenceNormalizedControl]

/-- Restoring the common phase recovers the original reflector vector. -/
lemma control_eq_common_mul_normalized (ref : ι) (control : ι → Circle) :
    control = fun j ↦ control ref * referenceNormalizedControl ref control j := by
  funext j
  simp [referenceNormalizedControl]

/-- Reference normalization changes the complete output field only by the
common unit-modulus factor `(control ref)⁻¹`. -/
lemma generalOutput_referenceNormalized (M : Matrix ι ι ℂ)
    (fixed control : ι → Circle) (ref n : ι) :
    (fun q ↦ generalOutput M fixed (referenceNormalizedControl ref control) n q) =
      fun q ↦ ((control ref)⁻¹ : Circle) * generalOutput M fixed control n q := by
  funext q
  simp only [generalOutput]
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro j _
  simp only [generalArmPhase, referenceNormalizedControl, Circle.coe_mul,
    Circle.coe_inv]
  ring

/-- Output powers are exactly unchanged by reference normalization. -/
lemma fieldPower_referenceNormalized (M : Matrix ι ι ℂ)
    (fixed control : ι → Circle) (ref n : ι) :
    fieldPower (fun q ↦ generalOutput M fixed
      (referenceNormalizedControl ref control) n q) =
      fieldPower (fun q ↦ generalOutput M fixed control n q) := by
  rw [generalOutput_referenceNormalized]
  funext q
  simp [fieldPower, norm_mul, Circle.norm_coe]

/-- Reachable powers when the chosen reference reflector is fixed to one. -/
noncomputable def referenceReachablePowers (M : Matrix ι ι ℂ)
    (fixed : ι → Circle) (ref n : ι) : Set (ι → ℝ) :=
  {p | ∃ control : ι → Circle, control ref = 1 ∧
    p = fieldPower (fun q ↦ generalOutput M fixed control n q)}

/-- **`N - 1` controls suffice for powers.** In every finite dimension and
for every MMI, fixing one reflector phase to one leaves the complete reachable
power set unchanged. -/
theorem reachablePowers_eq_referenceReachablePowers (M : Matrix ι ι ℂ)
    (fixed : ι → Circle) (ref n : ι) :
    reachablePowers M fixed n = referenceReachablePowers M fixed ref n := by
  ext p
  constructor
  · rintro ⟨z, ⟨control, rfl⟩, rfl⟩
    refine ⟨referenceNormalizedControl ref control,
      referenceNormalizedControl_at_ref ref control, ?_⟩
    exact (fieldPower_referenceNormalized M fixed control ref n).symm
  · rintro ⟨control, _, rfl⟩
    exact ⟨_, ⟨control, rfl⟩, rfl⟩

/-- Column orthogonality, written in exactly the form used by the optical
routing calculation.  For a square complex matrix this is equivalent to
unitarity. -/
def ColumnUnitary (M : Matrix ι ι ℂ) : Prop :=
  ∀ q m, ∑ j, M j q * conj (M j m) = if q = m then 1 else 0

/-- The entrywise magnitude condition needed for every pair of columns, plus
the non-vanishing guaranteed by an ideal equal-split MMI. -/
def EqualSplitNonzero (M : Matrix ι ι ℂ) : Prop :=
  (∀ j a b, ‖M j a‖ = ‖M j b‖) ∧ (∀ j a, M j a ≠ 0)

/-- The ratio between the conjugated target-column phase and the input-column
phase is itself a unit-modulus complex number. -/
noncomputable def columnRatio (M : Matrix ι ι ℂ) (hM : EqualSplitNonzero M)
    (n m j : ι) : Circle :=
  ⟨conj (M j m) / M j n, mem_sphere_zero_iff_norm.2 <| by
    rw [norm_div, norm_conj, hM.1 j m n]
    exact div_self (norm_ne_zero_iff.mpr (hM.2 j n))⟩

/-- General reflector settings: cancel both fixed-phase traversals and convert
the input-column phase signature into the conjugated target-column signature. -/
noncomputable def generalRoutingControl (M : Matrix ι ι ℂ)
    (hM : EqualSplitNonzero M) (fixed : ι → Circle) (global : Circle)
    (n m : ι) : ι → Circle := fun j ↦
  (fixed j)⁻¹ * (fixed j)⁻¹ * global * columnRatio M hM n m j

lemma armPhase_generalRoutingControl (M : Matrix ι ι ℂ)
    (hM : EqualSplitNonzero M) (fixed : ι → Circle) (global : Circle)
    (n m j : ι) :
    generalArmPhase fixed (generalRoutingControl M hM fixed global n m) j =
      global * columnRatio M hM n m j := by
  simp only [generalArmPhase, generalRoutingControl]
  calc
    fixed j * ((fixed j)⁻¹ * (fixed j)⁻¹ * global * columnRatio M hM n m j) * fixed j =
        (fixed j * (fixed j)⁻¹) * ((fixed j)⁻¹ * fixed j) *
          (global * columnRatio M hM n m j) := by ac_rfl
    _ = global * columnRatio M hM n m j := by simp

/-- **General pair-wise routing.** Column unitarity supplies destructive
interference at unwanted ports, while equal entrywise column magnitudes make
the required diagonal reflector coefficients phase-only. -/
theorem general_pairwise_routing (M : Matrix ι ι ℂ)
    (hU : ColumnUnitary M) (hM : EqualSplitNonzero M)
    (fixed : ι → Circle) (global : Circle) (n m : ι) :
    (fun q ↦ generalOutput M fixed
      (generalRoutingControl M hM fixed global n m) n q) =
      fun q ↦ (global : ℂ) * if q = m then 1 else 0 := by
  funext q
  simp only [generalOutput]
  calc
    ∑ j, M j q *
        (generalArmPhase fixed (generalRoutingControl M hM fixed global n m) j : ℂ) * M j n =
        ∑ j, (global : ℂ) * (M j q * conj (M j m)) := by
          apply Finset.sum_congr rfl
          intro j _
          rw [armPhase_generalRoutingControl]
          simp only [Circle.coe_mul, columnRatio]
          change M j q * ((global : ℂ) * (conj (M j m) / M j n)) * M j n =
            (global : ℂ) * (M j q * conj (M j m))
          field_simp [hM.2 j n]
          <;> ring
    _ = (global : ℂ) * ∑ j, M j q * conj (M j m) := by
          rw [Finset.mul_sum]
    _ = (global : ℂ) * if q = m then 1 else 0 := by rw [hU q m]

/-! ### Route-specific magnitude condition -/

/-- The exact weaker condition needed for a selected route: the input and
target columns have equal magnitude arm by arm.  Zero entries are allowed. -/
def PairColumnMagnitude (M : Matrix ι ι ℂ) (n m : ι) : Prop :=
  ∀ j, ‖M j m‖ = ‖M j n‖

/-- Phase ratio for a selected pair of columns.  If both entries vanish, the
arm is irrelevant and the ratio is chosen to be one. -/
noncomputable def pairColumnRatio (M : Matrix ι ι ℂ)
    (n m : ι) (hmag : PairColumnMagnitude M n m) (j : ι) : Circle :=
  if h : M j n = 0 then 1 else
    ⟨conj (M j m) / M j n, mem_sphere_zero_iff_norm.2 <| by
      rw [norm_div, norm_conj, hmag j]
      exact div_self (norm_ne_zero_iff.mpr h)⟩

lemma pairColumnRatio_mul_input (M : Matrix ι ι ℂ)
    (n m : ι) (hmag : PairColumnMagnitude M n m) (j : ι) :
    (pairColumnRatio M n m hmag j : ℂ) * M j n = conj (M j m) := by
  by_cases h : M j n = 0
  · have hm : M j m = 0 := by
      apply norm_eq_zero.mp
      rw [hmag j, norm_eq_zero]
      exact h
    simp [pairColumnRatio, h, hm]
  · simp only [pairColumnRatio, h, ↓reduceDIte]
    change (conj (M j m) / M j n) * M j n = conj (M j m)
    field_simp [h]

/-- Reflector settings for one compatible pair of columns. -/
noncomputable def pairMagnitudeRoutingControl (M : Matrix ι ι ℂ)
    (n m : ι) (hmag : PairColumnMagnitude M n m) (fixed : ι → Circle)
    (global : Circle) : ι → Circle := fun j ↦
  (fixed j)⁻¹ * (fixed j)⁻¹ * global * pairColumnRatio M n m hmag j

lemma armPhase_pairMagnitudeRoutingControl (M : Matrix ι ι ℂ)
    (n m : ι) (hmag : PairColumnMagnitude M n m) (fixed : ι → Circle)
    (global : Circle) (j : ι) :
    generalArmPhase fixed (pairMagnitudeRoutingControl M n m hmag fixed global) j =
      global * pairColumnRatio M n m hmag j := by
  simp only [generalArmPhase, pairMagnitudeRoutingControl]
  calc
    fixed j * ((fixed j)⁻¹ * (fixed j)⁻¹ * global *
        pairColumnRatio M n m hmag j) * fixed j =
        (fixed j * (fixed j)⁻¹) * ((fixed j)⁻¹ * fixed j) *
          (global * pairColumnRatio M n m hmag j) := by ac_rfl
    _ = global * pairColumnRatio M n m hmag j := by simp

/-- **Exact route-specific theorem.** Full equal splitting is unnecessary:
column unitarity and equal arm-by-arm magnitudes of just the selected input and
target columns suffice, even if their common entries vanish on some arms. -/
theorem pairMagnitude_pairwise_routing (M : Matrix ι ι ℂ)
    (hU : ColumnUnitary M) (n m : ι) (hmag : PairColumnMagnitude M n m)
    (fixed : ι → Circle) (global : Circle) :
    (fun q ↦ generalOutput M fixed
      (pairMagnitudeRoutingControl M n m hmag fixed global) n q) =
      fun q ↦ (global : ℂ) * if q = m then 1 else 0 := by
  funext q
  simp only [generalOutput]
  calc
    ∑ j, M j q *
        (generalArmPhase fixed
          (pairMagnitudeRoutingControl M n m hmag fixed global) j : ℂ) * M j n =
        ∑ j, (global : ℂ) * (M j q * conj (M j m)) := by
          apply Finset.sum_congr rfl
          intro j _
          rw [armPhase_pairMagnitudeRoutingControl]
          simp only [Circle.coe_mul]
          calc
            M j q * ((global : ℂ) * (pairColumnRatio M n m hmag j : ℂ)) * M j n =
                (global : ℂ) * M j q *
                  ((pairColumnRatio M n m hmag j : ℂ) * M j n) := by ring
            _ = (global : ℂ) * (M j q * conj (M j m)) := by
                  rw [pairColumnRatio_mul_input M n m hmag j]
                  ring
    _ = (global : ℂ) * ∑ j, M j q * conj (M j m) := by
          rw [Finset.mul_sum]
    _ = (global : ℂ) * if q = m then 1 else 0 := by rw [hU q m]

/-! ### Reachable-region invariance under Hadamard/DFT factorization -/

/-- Row and column rephasings and permutations of a reference MMI.  This is
the entrywise form of `Dout * Pout * F * Pin * Din`. -/
noncomputable def monomialTransform (F : Matrix ι ι ℂ)
    (rowPhase colPhase : ι → Circle) (rowPerm colPerm : ι ≃ ι) :
    Matrix ι ι ℂ := fun j k ↦
  (rowPhase j : ℂ) * F (rowPerm j) (colPerm k) * (colPhase k : ℂ)

/-- Translate physical reflector settings to the settings of the reference
matrix by absorbing the row phases and row permutation. -/
noncomputable def factorControlToBase (rowPhase : ι → Circle)
    (rowPerm : ι ≃ ι) (control : ι → Circle) : ι → Circle := fun k ↦
  rowPhase (rowPerm.symm k) * rowPhase (rowPerm.symm k) *
    control (rowPerm.symm k)

/-- Inverse translation from reference reflector settings to the physical
matrix settings. -/
noncomputable def factorControlFromBase (rowPhase : ι → Circle)
    (rowPerm : ι ≃ ι) (baseControl : ι → Circle) : ι → Circle := fun j ↦
  (rowPhase j)⁻¹ * (rowPhase j)⁻¹ * baseControl (rowPerm j)

lemma factorControl_inverse (rowPhase : ι → Circle) (rowPerm : ι ≃ ι)
    (baseControl : ι → Circle) :
    factorControlToBase rowPhase rowPerm
      (factorControlFromBase rowPhase rowPerm baseControl) = baseControl := by
  funext k
  simp only [factorControlToBase, factorControlFromBase, Equiv.apply_symm_apply]
  calc
    rowPhase (rowPerm.symm k) * rowPhase (rowPerm.symm k) *
        ((rowPhase (rowPerm.symm k))⁻¹ * (rowPhase (rowPerm.symm k))⁻¹ *
          baseControl k) =
        (rowPhase (rowPerm.symm k) * (rowPhase (rowPerm.symm k))⁻¹) *
        (rowPhase (rowPerm.symm k) * (rowPhase (rowPerm.symm k))⁻¹) *
        baseControl k := by ac_rfl
    _ = baseControl k := by simp

lemma monomial_output_relation (F : Matrix ι ι ℂ)
    (rowPhase colPhase : ι → Circle) (rowPerm colPerm : ι ≃ ι)
    (control : ι → Circle) (n q : ι) :
    generalOutput (monomialTransform F rowPhase colPhase rowPerm colPerm)
      (fun _ ↦ 1) control n q =
      ((colPhase q * colPhase n : Circle) : ℂ) *
        generalOutput F (fun _ ↦ 1)
          (factorControlToBase rowPhase rowPerm control) (colPerm n) (colPerm q) := by
  simp only [generalOutput, monomialTransform, generalArmPhase, one_mul, mul_one]
  calc
    (∑ j, ((rowPhase j : ℂ) * F (rowPerm j) (colPerm q) *
        (colPhase q : ℂ)) * (control j : ℂ) *
        ((rowPhase j : ℂ) * F (rowPerm j) (colPerm n) *
          (colPhase n : ℂ))) =
        ∑ j, ((colPhase q * colPhase n : Circle) : ℂ) *
          (F (rowPerm j) (colPerm q) *
            ((rowPhase j * rowPhase j * control j : Circle) : ℂ) *
            F (rowPerm j) (colPerm n)) := by
          apply Finset.sum_congr rfl
          intro j _
          simp only [Circle.coe_mul]
          ring
    _ = ((colPhase q * colPhase n : Circle) : ℂ) *
        ∑ j, F (rowPerm j) (colPerm q) *
          ((rowPhase j * rowPhase j * control j : Circle) : ℂ) *
          F (rowPerm j) (colPerm n) := by
          rw [Finset.mul_sum]
    _ = ((colPhase q * colPhase n : Circle) : ℂ) *
        ∑ k, F k (colPerm q) *
          (factorControlToBase rowPhase rowPerm control k : ℂ) *
          F k (colPerm n) := by
          congr 1
          exact Fintype.sum_equiv rowPerm
            (fun j ↦ F (rowPerm j) (colPerm q) *
              ((rowPhase j * rowPhase j * control j : Circle) : ℂ) *
              F (rowPerm j) (colPerm n))
            (fun k ↦ F k (colPerm q) *
              (factorControlToBase rowPhase rowPerm control k : ℂ) *
              F k (colPerm n))
            (fun j ↦ by
              simp only [factorControlToBase, Equiv.symm_apply_apply])

/-- Reindex a power vector by the external-port permutation. -/
def reindexPower (colPerm : ι ≃ ι) (p : ι → ℝ) : ι → ℝ :=
  fun q ↦ p (colPerm q)

lemma monomial_power_relation (F : Matrix ι ι ℂ)
    (rowPhase colPhase : ι → Circle) (rowPerm colPerm : ι ≃ ι)
    (control : ι → Circle) (n : ι) :
    fieldPower (fun q ↦ generalOutput
      (monomialTransform F rowPhase colPhase rowPerm colPerm)
      (fun _ ↦ 1) control n q) =
      reindexPower colPerm (fieldPower (fun q ↦ generalOutput F (fun _ ↦ 1)
        (factorControlToBase rowPhase rowPerm control) (colPerm n) q)) := by
  funext q
  rw [fieldPower, reindexPower, monomial_output_relation, norm_mul, Circle.norm_coe, one_mul]
  rfl

/-- **Reachable-region equivalence from Hadamard factorization.** Any MMI
obtained from a reference matrix by diagonal phases and row/column
permutations has exactly the reference reachable power region, up to the
fixed external-port permutation.  The `3 × 3` classification theorem supplies
this factorization for every equal-split unitary of order three. -/
theorem reachablePowers_monomialTransform (F : Matrix ι ι ℂ)
    (rowPhase colPhase : ι → Circle) (rowPerm colPerm : ι ≃ ι) (n : ι) :
    reachablePowers (monomialTransform F rowPhase colPhase rowPerm colPerm)
      (fun _ ↦ 1) n =
      reindexPower colPerm '' reachablePowers F (fun _ ↦ 1) (colPerm n) := by
  ext p
  constructor
  · rintro ⟨z, ⟨control, rfl⟩, rfl⟩
    refine ⟨fieldPower (fun q ↦ generalOutput F (fun _ ↦ 1)
      (factorControlToBase rowPhase rowPerm control) (colPerm n) q), ?_, ?_⟩
    · exact ⟨_, ⟨factorControlToBase rowPhase rowPerm control, rfl⟩, rfl⟩
    · exact (monomial_power_relation F rowPhase colPhase rowPerm colPerm control n).symm
  · rintro ⟨pBase, ⟨zBase, ⟨baseControl, rfl⟩, rfl⟩, rfl⟩
    let control := factorControlFromBase rowPhase rowPerm baseControl
    refine ⟨_, ⟨control, rfl⟩, ?_⟩
    rw [monomial_power_relation]
    simp only [control, factorControl_inverse]

/-- Reachable-region equivalence when an explicit Hadamard factorization is
available.  The theorem `threePort_reachable_region` below constructs this
certificate from equal splitting and unitarity in dimension three. -/
theorem reachable_region_of_factorization
    (F M : Matrix (Fin 3) (Fin 3) ℂ)
    (rowPhase colPhase : Fin 3 → Circle)
    (rowPerm colPerm : Fin 3 ≃ Fin 3)
    (hfactor : M = monomialTransform F rowPhase colPhase rowPerm colPerm)
    (fixed : Fin 3 → Circle) (n : Fin 3) :
    reachablePowers M fixed n =
      reindexPower colPerm '' reachablePowers F (fun _ ↦ 1) (colPerm n) := by
  rw [fixedPhase_reachablePowers, hfactor]
  exact reachablePowers_monomialTransform F rowPhase colPhase rowPerm colPerm n

/-! ### Complete classification and reachable-region theorem in dimension three -/

/-- Orthogonality of distinct rows of a matrix whose entries lie on the unit
circle.  This is the phase-only form obtained by multiplying an equal-split
`3 × 3` unitary by `√3`. -/
def CircleRowOrthogonal (U : Matrix (Fin 3) (Fin 3) Circle) : Prop :=
  ∀ j l, j ≠ l → ∑ k, (U j k : ℂ) * conj (U l k : ℂ) = 0

/-- Remove the phases in the zeroth row and zeroth column. -/
noncomputable def dephase3 (U : Matrix (Fin 3) (Fin 3) Circle) :
    Matrix (Fin 3) (Fin 3) Circle := fun j k ↦
  (U j 0)⁻¹ * U j k * ((U 0 0)⁻¹ * U 0 k)⁻¹

lemma dephase3_zero_row (U : Matrix (Fin 3) (Fin 3) Circle) (k : Fin 3) :
    dephase3 U 0 k = 1 := by
  simp only [dephase3]
  calc
    (U 0 0)⁻¹ * U 0 k * ((U 0 0)⁻¹ * U 0 k)⁻¹ =
        ((U 0 0)⁻¹ * U 0 k) * ((U 0 0)⁻¹ * U 0 k)⁻¹ := by rfl
    _ = 1 := by group

lemma dephase3_zero_col (U : Matrix (Fin 3) (Fin 3) Circle) (j : Fin 3) :
    dephase3 U j 0 = 1 := by
  simp [dephase3]

lemma dephase3_rowOrthogonal (U : Matrix (Fin 3) (Fin 3) Circle)
    (hU : CircleRowOrthogonal U) : CircleRowOrthogonal (dephase3 U) := by
  intro j l hjl
  rw [show (∑ k, (dephase3 U j k : ℂ) * conj (dephase3 U l k : ℂ)) =
      ((U j 0 : ℂ)⁻¹ * conj ((U l 0 : ℂ)⁻¹)) *
      ∑ k, (U j k : ℂ) * conj (U l k : ℂ) by
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro k _
    rw [← Circle.coe_inv_eq_conj (dephase3 U l k), map_inv₀,
      ← Circle.coe_inv_eq_conj (U l 0), ← Circle.coe_inv_eq_conj (U l k)]
    norm_cast
    simp only [dephase3]
    group
    ac_rfl]
  rw [hU j l hjl, mul_zero]

/-- If three unit phasors `1, a, b` sum to zero, the two nontrivial
phasors have product one. -/
lemma circle_pair_product_one (a b : Circle)
    (h : (1 : ℂ) + (a : ℂ) + (b : ℂ) = 0) :
    (a : ℂ) * (b : ℂ) = 1 := by
  have hc := congrArg conj h
  simp only [map_add, map_one, map_zero] at hc
  rw [← Circle.coe_inv_eq_conj, ← Circle.coe_inv_eq_conj] at hc
  have ha : (a : ℂ) ≠ 0 := Circle.coe_ne_zero a
  have hb : (b : ℂ) ≠ 0 := Circle.coe_ne_zero b
  field_simp [ha, hb] at hc
  linear_combination hc - h

/-- The same phasor condition implies the primitive-cube-root equation. -/
lemma circle_pair_poly (a b : Circle)
    (h : (1 : ℂ) + (a : ℂ) + (b : ℂ) = 0) :
    (a : ℂ)^2 + (a : ℂ) + 1 = 0 := by
  have hp := circle_pair_product_one a b h
  linear_combination (a : ℂ) * h - hp

/-- Two orthogonal dephased rows must use opposite orderings of the two
nontrivial cube-root phases. -/
lemma dephased_four_entries (a b c d : Circle)
    (hab : (1 : ℂ) + (a : ℂ) + (b : ℂ) = 0)
    (hcd : (1 : ℂ) + (c : ℂ) + (d : ℂ) = 0)
    (hacbd : (1 : ℂ) + (a : ℂ) * conj (c : ℂ) +
      (b : ℂ) * conj (d : ℂ) = 0) :
    b = c ∧ d = a := by
  have ha := circle_pair_poly a b hab
  have hc := circle_pair_poly c d hcd
  have hne : a ≠ c := by
    intro heq
    have hbd : b = d := by
      apply Circle.coe_injective
      rw [heq] at hab
      linear_combination hab - hcd
    rw [heq, hbd] at hacbd
    rw [← Circle.coe_inv_eq_conj, ← Circle.coe_inv_eq_conj] at hacbd
    norm_cast at hacbd
    norm_num at hacbd
  have hac : (a : ℂ) + (c : ℂ) + 1 = 0 := by
    have hfac : ((a : ℂ) - (c : ℂ)) *
        ((a : ℂ) + (c : ℂ) + 1) = 0 := by
      linear_combination ha - hc
    rcases mul_eq_zero.mp hfac with h | h
    · exact (hne (Circle.coe_injective (sub_eq_zero.mp h))).elim
    · exact h
  constructor
  · apply Circle.coe_injective
    linear_combination hab - hac
  · apply Circle.coe_injective
    linear_combination hcd - hac

/-- The unnormalized three-point Fourier phase pattern. -/
noncomputable def FourierPhase3 (w : Circle) : Matrix (Fin 3) (Fin 3) Circle :=
  fun j k ↦ w ^ (j.val * k.val)

/-- **Three-dimensional complex-Hadamard classification.** After dephasing,
every row-orthogonal `3 × 3` unit-circle matrix is the Fourier phase matrix
for a primitive cube root `w`. -/
theorem dephased_three_classification (U : Matrix (Fin 3) (Fin 3) Circle)
    (hU : CircleRowOrthogonal U) :
    let w := dephase3 U 1 1
    (w : ℂ)^2 + (w : ℂ) + 1 = 0 ∧ dephase3 U = FourierPhase3 w := by
  let H := dephase3 U
  have hH := dephase3_rowOrthogonal U hU
  have h01 := hH 0 1 (by decide)
  have h02 := hH 0 2 (by decide)
  have h12 := hH 1 2 (by decide)
  dsimp only [H] at h01 h02 h12 ⊢
  simp only [Fin.sum_univ_three] at h01 h02 h12
  simp only [dephase3_zero_row, dephase3_zero_col] at h01 h02 h12
  have h01c : (1 : ℂ) + (H 1 1 : ℂ) + (H 1 2 : ℂ) = 0 := by
    simpa only [map_add, map_mul, map_one, map_zero, conj_conj,
      Circle.coe_one, one_mul] using congrArg conj h01
  have h02c : (1 : ℂ) + (H 2 1 : ℂ) + (H 2 2 : ℂ) = 0 := by
    simpa only [map_add, map_mul, map_one, map_zero, conj_conj,
      Circle.coe_one, one_mul] using congrArg conj h02
  have h12c : (1 : ℂ) + (H 1 1 : ℂ) * conj (H 2 1 : ℂ) +
      (H 1 2 : ℂ) * conj (H 2 2 : ℂ) = 0 := by
    simpa only [Circle.coe_one, map_one, one_mul] using h12
  have hfour := dephased_four_entries (H 1 1) (H 1 2) (H 2 1) (H 2 2)
    h01c h02c h12c
  have hp := circle_pair_poly (H 1 1) (H 1 2) h01c
  have hpow : H 1 2 = (H 1 1)^2 := by
    apply Circle.coe_injective
    change (H 1 2 : ℂ) = (H 1 1 : ℂ)^2
    linear_combination h01c - hp
  have hcub : (H 1 1)^3 = 1 := by
    apply Circle.coe_injective
    change (H 1 1 : ℂ)^3 = 1
    linear_combination (H 1 1 : ℂ) * hp - hp
  have h21 : H 2 1 = (H 1 1)^2 := hfour.1.symm.trans hpow
  have h22 : H 2 2 = H 1 1 := hfour.2
  have hfourth : (H 1 1)^4 = H 1 1 := by
    rw [show (4 : ℕ) = 3 + 1 by decide, pow_add, hcub, one_mul, pow_one]
  refine ⟨hp, ?_⟩
  funext j k
  fin_cases j <;> fin_cases k
  all_goals simp [H, FourierPhase3, dephase3_zero_row, dephase3_zero_col,
    h21, h22, hpow, hfourth]

/-- Row-unitarity in the usual normalized complex-matrix representation. -/
def RowUnitary3 (M : Matrix (Fin 3) (Fin 3) ℂ) : Prop :=
  ∀ j l, ∑ k, M j k * conj (M l k) = if j = l then 1 else 0

/-- `RowUnitary3` is exactly Mathlib's standard unitary-group predicate. -/
lemma mem_unitaryGroup_iff_RowUnitary3 (M : Matrix (Fin 3) (Fin 3) ℂ) :
    M ∈ Matrix.unitaryGroup (Fin 3) ℂ ↔ RowUnitary3 M := by
  rw [Matrix.mem_unitaryGroup_iff]
  constructor
  · intro h j l
    have hjl := congrFun (congrFun h j) l
    simpa only [Matrix.mul_apply, star, Matrix.conjTranspose_apply,
      Matrix.one_apply] using hjl
  · intro h
    ext j l
    simpa only [Matrix.mul_apply, star, Matrix.conjTranspose_apply,
      Matrix.one_apply] using h j l

/-- Every entry of an ideal `3 × 3` equal-split MMI has magnitude `1/√3`. -/
def EqualSplit3 (M : Matrix (Fin 3) (Fin 3) ℂ) : Prop :=
  ∀ j k, ‖M j k‖ = (Real.sqrt 3)⁻¹

/-- Normalize each equal-split entry to a unit phasor. -/
noncomputable def phaseLift3 (M : Matrix (Fin 3) (Fin 3) ℂ)
    (hM : EqualSplit3 M) : Matrix (Fin 3) (Fin 3) Circle := fun j k ↦
  ⟨(Real.sqrt 3 : ℂ) * M j k, mem_sphere_zero_iff_norm.2 <| by
    rw [norm_mul, hM j k]
    norm_cast
    rw [Real.norm_of_nonneg (Real.sqrt_nonneg 3)]
    exact mul_inv_cancel₀ (ne_of_gt (Real.sqrt_pos.2 (by norm_num)))⟩

lemma phaseLift3_rowOrthogonal (M : Matrix (Fin 3) (Fin 3) ℂ)
    (hM : EqualSplit3 M) (hU : RowUnitary3 M) :
    CircleRowOrthogonal (phaseLift3 M hM) := by
  intro j l hjl
  simp only [phaseLift3]
  calc
    ∑ k, ((Real.sqrt 3 : ℂ) * M j k) *
        conj ((Real.sqrt 3 : ℂ) * M l k) =
        (Real.sqrt 3 : ℂ)^2 * ∑ k, M j k * conj (M l k) := by
          rw [Finset.mul_sum]
          apply Finset.sum_congr rfl
          intro k _
          rw [map_mul, conj_ofReal]
          ring
    _ = 0 := by rw [hU j l, if_neg hjl, mul_zero]

/-- The normalized three-point DFT associated with a primitive cube root. -/
noncomputable def threeDFT (w : Circle) : Matrix (Fin 3) (Fin 3) ℂ :=
  fun j k ↦ (Real.sqrt 3 : ℂ)⁻¹ * (FourierPhase3 w j k : ℂ)

/-- A matrix is a three-point DFT when its phase generator satisfies
`w² + w + 1 = 0`.  This equation also implies `w³ = 1` and `w ≠ 1`. -/
def IsThreeDFT (F : Matrix (Fin 3) (Fin 3) ℂ) : Prop :=
  ∃ w : Circle, (w : ℂ)^2 + (w : ℂ) + 1 = 0 ∧ F = threeDFT w

/-- Every ideal equal-split unitary of order three is obtained from a
three-point DFT by diagonal row and column phases. -/
theorem equalSplitUnitary3_factorization (M : Matrix (Fin 3) (Fin 3) ℂ)
    (hM : EqualSplit3 M) (hU : RowUnitary3 M) :
    ∃ (F : Matrix (Fin 3) (Fin 3) ℂ)
      (rowPhase colPhase : Fin 3 → Circle),
      IsThreeDFT F ∧
        M = monomialTransform F rowPhase colPhase (Equiv.refl _) (Equiv.refl _) := by
  let U := phaseLift3 M hM
  let w := dephase3 U 1 1
  have hclass := dephased_three_classification U
    (phaseLift3_rowOrthogonal M hM hU)
  dsimp only [w] at hclass
  let rowPhase : Fin 3 → Circle := fun j ↦ U j 0
  let colPhase : Fin 3 → Circle := fun k ↦ (U 0 0)⁻¹ * U 0 k
  refine ⟨threeDFT w, rowPhase, colPhase, ⟨w, hclass.1, rfl⟩, ?_⟩
  funext j k
  have hd := congrFun (congrFun hclass.2 j) k
  have hphase : U j k = rowPhase j * FourierPhase3 w j k * colPhase k := by
    dsimp only [rowPhase, colPhase, w]
    rw [← hd]
    simp only [dephase3]
    group
  change M j k = (rowPhase j : ℂ) *
    ((Real.sqrt 3 : ℂ)⁻¹ * (FourierPhase3 w j k : ℂ)) *
      (colPhase k : ℂ)
  have hlift : (U j k : ℂ) = (Real.sqrt 3 : ℂ) * M j k := rfl
  rw [hphase] at hlift
  simp only [Circle.coe_mul] at hlift
  have hs : (Real.sqrt 3 : ℂ) ≠ 0 := by
    exact_mod_cast ne_of_gt (Real.sqrt_pos.2 (by norm_num))
  calc
    M j k = (Real.sqrt 3 : ℂ)⁻¹ * ((Real.sqrt 3 : ℂ) * M j k) := by
      field_simp [hs]
    _ = (Real.sqrt 3 : ℂ)⁻¹ *
        ((rowPhase j : ℂ) * (FourierPhase3 w j k : ℂ) *
          (colPhase k : ℂ)) := by rw [hlift]
    _ = (rowPhase j : ℂ) *
        ((Real.sqrt 3 : ℂ)⁻¹ * (FourierPhase3 w j k : ℂ)) *
          (colPhase k : ℂ) := by ring

/-- **Full three-port reachable-region universality.** Every ideal equal-split
unitary `3 × 3` MMI, even with arbitrary fixed reciprocal arm phases, has
exactly the reachable power region of a three-point DFT. -/
theorem threePort_reachable_region (M : Matrix (Fin 3) (Fin 3) ℂ)
    (hM : EqualSplit3 M) (hU : RowUnitary3 M)
    (fixed : Fin 3 → Circle) (n : Fin 3) :
    ∃ F : Matrix (Fin 3) (Fin 3) ℂ, IsThreeDFT F ∧
      reachablePowers M fixed n = reachablePowers F (fun _ ↦ 1) n := by
  rcases equalSplitUnitary3_factorization M hM hU with
    ⟨F, rowPhase, colPhase, hF, hfactor⟩
  refine ⟨F, hF, ?_⟩
  rw [fixedPhase_reachablePowers, hfactor]
  rw [reachablePowers_monomialTransform]
  rw [show reindexPower (Equiv.refl (Fin 3)) = id by rfl, Set.image_id]
  rfl

/-- The three-port theorem stated directly with Mathlib's unitary-group
membership, matching the paper's notation `M ∈ U(3)`. -/
theorem threePort_reachable_region_of_mem_unitary
    (M : Matrix (Fin 3) (Fin 3) ℂ) (hM : EqualSplit3 M)
    (hU : M ∈ Matrix.unitaryGroup (Fin 3) ℂ)
    (fixed : Fin 3 → Circle) (n : Fin 3) :
    ∃ F : Matrix (Fin 3) (Fin 3) ℂ, IsThreeDFT F ∧
      reachablePowers M fixed n = reachablePowers F (fun _ ↦ 1) n :=
  threePort_reachable_region M hM
    ((mem_unitaryGroup_iff_RowUnitary3 M).mp hU) fixed n

end General

/-! ## The general control law specializes exactly to the DFT phase ramp -/

/-- The normalized DFT matrix in the same sign convention as `ZMod.dft`. -/
noncomputable def normalizedDFTMatrix : Matrix (ZMod N) (ZMod N) ℂ :=
  fun j k ↦ (Real.sqrt N : ℂ)⁻¹ * ZMod.stdAddChar (-(j * k))

lemma normalizedDFTMatrix_equalSplitNonzero :
    EqualSplitNonzero (normalizedDFTMatrix (N := N)) := by
  constructor
  · intro j a b
    simp only [normalizedDFTMatrix, norm_mul, norm_inv]
    rw [ZMod.stdAddChar_apply, ZMod.stdAddChar_apply,
      Circle.norm_coe, Circle.norm_coe]
  · intro j a hzero
    have hs : (Real.sqrt N : ℂ) ≠ 0 := by
      exact_mod_cast ne_of_gt (Real.sqrt_pos.2 (by
        exact_mod_cast Nat.pos_of_ne_zero (NeZero.ne N)))
    simp only [normalizedDFTMatrix] at hzero
    exact mul_ne_zero (inv_ne_zero hs) (Circle.coe_ne_zero _) hzero

/-- For a normalized DFT, the general conjugated-column ratio is precisely
the Fourier phase ramp indexed by `m + n`. -/
lemma normalizedDFT_columnRatio (n m j : ZMod N) :
    columnRatio (normalizedDFTMatrix (N := N))
      normalizedDFTMatrix_equalSplitNonzero n m j =
      ZMod.toCircle (j * (m + n)) := by
  apply Circle.coe_injective
  simp only [columnRatio, normalizedDFTMatrix, ZMod.stdAddChar_apply]
  rw [map_mul, map_inv₀, conj_ofReal,
    ← Circle.coe_inv_eq_conj (ZMod.toCircle (-(j * m)))]
  have hs : (Real.sqrt N : ℂ) ≠ 0 := by
    exact_mod_cast ne_of_gt (Real.sqrt_pos.2 (by
      exact_mod_cast Nat.pos_of_ne_zero (NeZero.ne N)))
  field_simp [hs]
  rw [show (ZMod.toCircle (j * (m + n)) : ℂ) *
      ((Real.sqrt N : ℂ) * (ZMod.toCircle (-(j * m)) : ℂ) *
        (ZMod.toCircle (-(j * n)) : ℂ)) =
      (Real.sqrt N : ℂ) *
        ((ZMod.toCircle (j * (m + n)) : ℂ) *
          (ZMod.toCircle (-(j * m)) : ℂ) *
          (ZMod.toCircle (-(j * n)) : ℂ)) by ring]
  conv_lhs => rw [← mul_one (Real.sqrt N : ℂ)]
  have hcircle : (1 : Circle) = ZMod.toCircle (j * (m + n)) *
      ZMod.toCircle (-(j * m)) * ZMod.toCircle (-(j * n)) := by
    rw [← ZMod.toCircle.map_add_eq_mul, ← ZMod.toCircle.map_add_eq_mul]
    rw [show j * (m + n) + -(j * m) + -(j * n) = 0 by ring]
    exact (ZMod.toCircle (N := N)).map_zero_eq_one.symm
  have hcomplex := congrArg (fun x : Circle ↦ (x : ℂ)) hcircle
  simp only [Circle.coe_one, Circle.coe_mul] at hcomplex
  rw [hcomplex]

/-- The arbitrary-MMI reflector formula reduces exactly—not merely up to
output power—to the fixed-phase-corrected DFT control law of Note 1. -/
theorem generalRoutingControl_normalizedDFT (fixed : ZMod N → Circle)
    (global : Circle) (n m : ZMod N) :
    generalRoutingControl (normalizedDFTMatrix (N := N))
      normalizedDFTMatrix_equalSplitNonzero fixed global n m =
      controlFor fixed global (m + n) := by
  funext j
  simp only [generalRoutingControl, controlFor, normalizedDFT_columnRatio]

/-! ## A four-port counterexample to full reachable-region universality -/

/-- The normalized real Walsh `4 × 4` complex Hadamard matrix. -/
noncomputable def walsh4 : Matrix (Fin 4) (Fin 4) ℂ :=
  fun j k ↦ (2 : ℂ)⁻¹ * (![![1, 1, 1, 1], ![1, 1, -1, -1],
    ![1, -1, 1, -1], ![1, -1, -1, 1]] : Matrix (Fin 4) (Fin 4) ℂ) j k

/-- The normalized four-point DFT. -/
noncomputable def fourier4 : Matrix (Fin 4) (Fin 4) ℂ :=
  fun j k ↦ (2 : ℂ)⁻¹ * (![![1, 1, 1, 1], ![1, I, -1, -I],
    ![1, -1, 1, -1], ![1, -I, -1, I]] : Matrix (Fin 4) (Fin 4) ℂ) j k

lemma walsh4_columnUnitary : ColumnUnitary walsh4 := by
  intro q m
  fin_cases q <;> fin_cases m
  all_goals simp [walsh4, Fin.sum_univ_succ, map_add, map_mul,
    Complex.conj_ofNat] <;> norm_num <;> ring

lemma fourier4_columnUnitary : ColumnUnitary fourier4 := by
  intro q m
  fin_cases q <;> fin_cases m
  all_goals simp [fourier4, Fin.sum_univ_succ, map_add, map_mul, conj_I,
    Complex.conj_ofNat] <;> norm_num <;> ring_nf
  all_goals simp [I_mul_I] <;> ring

lemma walsh4_equalSplitNonzero : EqualSplitNonzero walsh4 := by
  constructor
  · intro j a b
    fin_cases j <;> fin_cases a <;> fin_cases b <;> simp [walsh4]
  · intro j a
    fin_cases j <;> fin_cases a <;> norm_num [walsh4]

lemma fourier4_equalSplitNonzero : EqualSplitNonzero fourier4 := by
  constructor
  · intro j a b
    fin_cases j <;> fin_cases a <;> fin_cases b <;> simp [fourier4]
  · intro j a
    fin_cases j <;> fin_cases a <;> norm_num [fourier4]

/-- The unit phasor `i`. -/
noncomputable def circleI : Circle :=
  ⟨I, mem_sphere_zero_iff_norm.2 norm_I⟩

/-- A reflector setting producing a nontrivial two-output split for `walsh4`. -/
noncomputable def walshControl4 : Fin 4 → Circle := ![1, 1, circleI, circleI]

/-- The corresponding Walsh output-power vector. -/
noncomputable def walshWitnessPower4 : Fin 4 → ℝ :=
  fieldPower (fun q ↦ generalOutput walsh4 (fun _ ↦ 1) walshControl4 0 q)

lemma walsh4_output_zero :
    generalOutput walsh4 (fun _ ↦ 1) walshControl4 0 0 = (1 + I) / 2 := by
  simp [generalOutput, generalArmPhase, walsh4, walshControl4, circleI,
    Fin.sum_univ_four]
  ring

lemma walsh4_output_one :
    generalOutput walsh4 (fun _ ↦ 1) walshControl4 0 1 = (1 - I) / 2 := by
  simp [generalOutput, generalArmPhase, walsh4, walshControl4, circleI,
    Fin.sum_univ_four]
  ring

lemma walsh4_output_zero_two :
    generalOutput walsh4 (fun _ ↦ 1) walshControl4 0 2 = 0 := by
  simp [generalOutput, generalArmPhase, walsh4, walshControl4, circleI,
    Fin.sum_univ_four]

lemma walsh4_output_zero_three :
    generalOutput walsh4 (fun _ ↦ 1) walshControl4 0 3 = 0 := by
  simp [generalOutput, generalArmPhase, walsh4, walshControl4, circleI,
    Fin.sum_univ_four]

lemma walsh4_output_zero_nonzero :
    generalOutput walsh4 (fun _ ↦ 1) walshControl4 0 0 ≠ 0 := by
  simp [generalOutput, generalArmPhase, walsh4, walshControl4, circleI,
    Fin.sum_univ_four]
  intro h
  have hi := congrArg im h
  norm_num at hi

lemma walsh4_output_one_nonzero :
    generalOutput walsh4 (fun _ ↦ 1) walshControl4 0 1 ≠ 0 := by
  simp [generalOutput, generalArmPhase, walsh4, walshControl4, circleI,
    Fin.sum_univ_four]
  intro h
  have hi := congrArg im h
  norm_num at hi

lemma walshWitnessPower4_two : walshWitnessPower4 2 = 0 := by
  simp [walshWitnessPower4, fieldPower, walsh4_output_zero_two]

lemma walshWitnessPower4_three : walshWitnessPower4 3 = 0 := by
  simp [walshWitnessPower4, fieldPower, walsh4_output_zero_three]

lemma walshWitnessPower4_zero : walshWitnessPower4 0 = 1 / 2 := by
  rw [walshWitnessPower4, fieldPower, walsh4_output_zero, Complex.sq_norm]
  norm_num [Complex.normSq, div_re, div_im]

lemma walshWitnessPower4_one : walshWitnessPower4 1 = 1 / 2 := by
  rw [walshWitnessPower4, fieldPower, walsh4_output_one, Complex.sq_norm]
  norm_num [Complex.normSq, div_re, div_im]

/-- The Walsh witness is exactly the power vector `(1/2, 1/2, 0, 0)` used
in Supplementary Note 2. -/
theorem walshWitnessPower4_eq :
    walshWitnessPower4 = ![(1 : ℝ) / 2, (1 : ℝ) / 2, 0, 0] := by
  funext q
  fin_cases q
  · exact walshWitnessPower4_zero
  · exact walshWitnessPower4_one
  · exact walshWitnessPower4_two
  · exact walshWitnessPower4_three

lemma walshWitnessPower4_zero_ne : walshWitnessPower4 0 ≠ 0 := by
  simp only [walshWitnessPower4, fieldPower]
  exact pow_ne_zero 2 (norm_ne_zero_iff.mpr walsh4_output_zero_nonzero)

lemma walshWitnessPower4_one_ne : walshWitnessPower4 1 ≠ 0 := by
  simp only [walshWitnessPower4, fieldPower]
  exact pow_ne_zero 2 (norm_ne_zero_iff.mpr walsh4_output_one_nonzero)

theorem walshWitnessPower4_reachable :
    walshWitnessPower4 ∈ reachablePowers walsh4 (fun _ ↦ 1) 0 := by
  exact ⟨_, ⟨walshControl4, rfl⟩, rfl⟩

/-- The same nontrivial two-output power vector is not reachable with the
four-point DFT.  If its last two outputs vanished, inversion of the DFT would
force four unit-modulus controls of the forms `a ± b` and `a ± i b`; equality
of their moduli forces `a * conj b = 0`, contradicting two nonzero outputs. -/
theorem walshWitnessPower4_not_fourier_reachable :
    walshWitnessPower4 ∉ reachablePowers fourier4 (fun _ ↦ 1) 0 := by
  rintro ⟨z, ⟨control, rfl⟩, hp⟩
  let y : Fin 4 → ℂ := fun q ↦
    generalOutput fourier4 (fun _ ↦ 1) control 0 q
  change fieldPower y = walshWitnessPower4 at hp
  have hp0 := congrFun hp 0
  have hp1 := congrFun hp 1
  have hp2 := congrFun hp 2
  have hp3 := congrFun hp 3
  have hy0 : y 0 ≠ 0 := by
    intro h
    apply walshWitnessPower4_zero_ne
    rw [← hp0]
    simp [fieldPower, h]
  have hy1 : y 1 ≠ 0 := by
    intro h
    apply walshWitnessPower4_one_ne
    rw [← hp1]
    simp [fieldPower, h]
  have hy2 : y 2 = 0 := by
    rw [walshWitnessPower4_two] at hp2
    have hn : ‖y 2‖ = 0 := by
      simp only [fieldPower] at hp2
      nlinarith [norm_nonneg (y 2)]
    exact norm_eq_zero.mp hn
  have hy3 : y 3 = 0 := by
    rw [walshWitnessPower4_three] at hp3
    have hn : ‖y 3‖ = 0 := by
      simp only [fieldPower] at hp3
      nlinarith [norm_nonneg (y 3)]
    exact norm_eq_zero.mp hn
  have h0 : y 0 = ((control 0 : ℂ) + control 1 + control 2 + control 3) / 4 := by
    simp [y, generalOutput, generalArmPhase, fourier4, Fin.sum_univ_four]
    ring
  have h1 : y 1 = ((control 0 : ℂ) + I * control 1 - control 2 - I * control 3) / 4 := by
    simp [y, generalOutput, generalArmPhase, fourier4, Fin.sum_univ_four]
    ring
  have h2 : y 2 = ((control 0 : ℂ) - control 1 + control 2 - control 3) / 4 := by
    simp [y, generalOutput, generalArmPhase, fourier4, Fin.sum_univ_four]
    ring
  have h3 : y 3 = ((control 0 : ℂ) - I * control 1 - control 2 + I * control 3) / 4 := by
    simp [y, generalOutput, generalArmPhase, fourier4, Fin.sum_univ_four]
    ring
  simp only [hy2, hy3] at h0 h1 h2 h3
  field_simp at h0 h1 h2 h3
  have hi : I^2 + 1 = (0 : ℂ) := by rw [pow_two, I_mul_I]; ring
  have hr0 : (control 0 : ℂ) = y 0 + y 1 := by
    linear_combination -(h0 + h1 + h2 + h3) / 4
  have hr1 : (control 1 : ℂ) = y 0 - I * y 1 := by
    linear_combination -(h0 - I * h1 - h2 + I * h3) / 4 +
      (((control 1 : ℂ) - control 3) / 2) * hi
  have hr2 : (control 2 : ℂ) = y 0 - y 1 := by
    linear_combination -(h0 - h1 + h2 - h3) / 4
  have hr3 : (control 3 : ℂ) = y 0 + I * y 1 := by
    linear_combination -(h0 + I * h1 - h2 - I * h3) / 4 +
      (((control 3 : ℂ) - control 1) / 2) * hi
  have hu0 : (control 0 : ℂ) * conj (control 0 : ℂ) = 1 := by
    rw [← Circle.coe_inv_eq_conj]
    norm_cast
    simp
  have hu1 : (control 1 : ℂ) * conj (control 1 : ℂ) = 1 := by
    rw [← Circle.coe_inv_eq_conj]
    norm_cast
    simp
  have hu2 : (control 2 : ℂ) * conj (control 2 : ℂ) = 1 := by
    rw [← Circle.coe_inv_eq_conj]
    norm_cast
    simp
  have hu3 : (control 3 : ℂ) * conj (control 3 : ℂ) = 1 := by
    rw [← Circle.coe_inv_eq_conj]
    norm_cast
    simp
  rw [hr0] at hu0
  rw [hr1] at hu1
  rw [hr2] at hu2
  rw [hr3] at hu3
  simp only [map_add, map_sub, map_mul, conj_I, conj_conj] at hu0 hu1 hu2 hu3
  have hsym : y 0 * conj (y 1) + y 1 * conj (y 0) = 0 := by
    linear_combination (hu0 - hu2) / 2
  have hasym : y 0 * conj (y 1) - y 1 * conj (y 0) = 0 := by
    linear_combination (-I / 2) * (hu1 - hu3) +
      (y 0 * conj (y 1) - y 1 * conj (y 0)) * hi
  have hzero : y 0 * conj (y 1) = 0 := by
    linear_combination (hsym + hasym) / 2
  rcases mul_eq_zero.mp hzero with h | h
  · exact hy0 h
  · apply hy1
    have hc := congrArg conj h
    simpa only [conj_conj, map_zero] using hc

/-- **Failure of full reachable-region universality at `N = 4`.** Both
matrices are equal-split and column-unitary, but their reachable power regions
from port zero differ. -/
theorem fourPort_reachable_regions_can_differ :
    EqualSplitNonzero walsh4 ∧ ColumnUnitary walsh4 ∧
    EqualSplitNonzero fourier4 ∧ ColumnUnitary fourier4 ∧
    ∃ p : Fin 4 → ℝ,
      p ∈ reachablePowers walsh4 (fun _ ↦ 1) 0 ∧
      p ∉ reachablePowers fourier4 (fun _ ↦ 1) 0 :=
  ⟨walsh4_equalSplitNonzero, walsh4_columnUnitary,
    fourier4_equalSplitNonzero, fourier4_columnUnitary,
    walshWitnessPower4, walshWitnessPower4_reachable,
    walshWitnessPower4_not_fourier_reachable⟩

/-- The four-port separation persists for every choice of fixed reciprocal
post-MMI phases on either device. -/
theorem fourPort_reachable_regions_can_differ_with_fixed
    (fixedWalsh fixedFourier : Fin 4 → Circle) :
    ∃ p : Fin 4 → ℝ,
      p ∈ reachablePowers walsh4 fixedWalsh 0 ∧
      p ∉ reachablePowers fourier4 fixedFourier 0 := by
  refine ⟨walshWitnessPower4, ?_, ?_⟩
  · rw [fixedPhase_reachablePowers]
    exact walshWitnessPower4_reachable
  · rw [fixedPhase_reachablePowers]
    exact walshWitnessPower4_not_fourier_reachable

end TFLNRouter
