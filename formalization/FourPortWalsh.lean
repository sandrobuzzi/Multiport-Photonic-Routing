import TFLNRouter
import Mathlib.Analysis.Fourier.FiniteAbelian.Orthogonality

/-!
# Simultaneous routing and the four-port Walsh--DFT family

This file formalizes reciprocity of the round trip, input-independence of the reachable
power set for an equal-split MMI, the binary Walsh matching construction, and
the explicit four-port equal-split family used in Supplementary Notes 2 and 7.
-/

namespace TFLNRouter

open Complex
open ComplexConjugate
open scoped BigOperators

/-! ## Reciprocity and input-independence -/

section General

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- The reciprocal round-trip matrix is symmetric for every forward matrix
and every diagonal reflector/fixed-phase setting. -/
theorem generalRoundTrip_symmetric (M : Matrix ι ι ℂ)
    (fixed control : ι → Circle) :
    (generalRoundTrip M fixed control).IsSymm := by
  apply Matrix.IsSymm.ext
  intro q n
  simp only [generalRoundTrip, generalOutput]
  apply Finset.sum_congr rfl
  intro j _
  ring

/-- The unit-modulus arm-by-arm ratio between any two columns of an
equal-split MMI. -/
noncomputable def inputColumnRatio (M : Matrix ι ι ℂ)
    (hM : EqualSplitNonzero M) (n m j : ι) : Circle :=
  ⟨M j m / M j n, mem_sphere_zero_iff_norm.2 <| by
    rw [norm_div, hM.1 j m n]
    exact div_self (norm_ne_zero_iff.mpr (hM.2 j n))⟩

/-- Translate a reflector setting used from input `m` into one producing the
same effective internal field from input `n`. -/
noncomputable def changeInputControl (M : Matrix ι ι ℂ)
    (hM : EqualSplitNonzero M) (fixed : ι → Circle)
    (n m : ι) (control : ι → Circle) : ι → Circle := fun j ↦
  (fixed j)⁻¹ * (fixed j)⁻¹ *
    (effectiveControl fixed control j * inputColumnRatio M hM n m j)

lemma changeInputControl_arm_field (M : Matrix ι ι ℂ)
    (hM : EqualSplitNonzero M) (fixed : ι → Circle)
    (n m : ι) (control : ι → Circle) (j : ι) :
    (generalArmPhase fixed (changeInputControl M hM fixed n m control) j : ℂ) * M j n =
      (generalArmPhase fixed control j : ℂ) * M j m := by
  have hphase :
      generalArmPhase fixed (changeInputControl M hM fixed n m control) j =
        effectiveControl fixed control j * inputColumnRatio M hM n m j := by
    simp only [generalArmPhase, changeInputControl]
    calc
      fixed j * ((fixed j)⁻¹ * (fixed j)⁻¹ *
          (effectiveControl fixed control j * inputColumnRatio M hM n m j)) * fixed j =
          (fixed j * (fixed j)⁻¹) * ((fixed j)⁻¹ * fixed j) *
            (effectiveControl fixed control j * inputColumnRatio M hM n m j) := by
              ac_rfl
      _ = effectiveControl fixed control j * inputColumnRatio M hM n m j := by simp
  rw [hphase]
  rw [armPhase_eq_effectiveControl]
  simp only [Circle.coe_mul, inputColumnRatio]
  change ((effectiveControl fixed control j : Circle) : ℂ) *
      (M j m / M j n) * M j n =
    ((effectiveControl fixed control j : Circle) : ℂ) * M j m
  field_simp [hM.2 j n]

lemma changeInputControl_output (M : Matrix ι ι ℂ)
    (hM : EqualSplitNonzero M) (fixed : ι → Circle)
    (n m : ι) (control : ι → Circle) :
    (fun q ↦ generalOutput M fixed
      (changeInputControl M hM fixed n m control) n q) =
    (fun q ↦ generalOutput M fixed control m q) := by
  funext q
  simp only [generalOutput]
  apply Finset.sum_congr rfl
  intro j _
  rw [mul_assoc, changeInputControl_arm_field]
  ring

/-- For every equal-split MMI, the complete reachable power set is the same
from every input port.  This is the formal version of the input-independence
used for the four-port reachable-region plots. -/
theorem reachablePowers_input_independent (M : Matrix ι ι ℂ)
    (hM : EqualSplitNonzero M) (fixed : ι → Circle) (n m : ι) :
    reachablePowers M fixed n = reachablePowers M fixed m := by
  ext p
  constructor
  · rintro ⟨z, ⟨control, rfl⟩, rfl⟩
    refine ⟨_, ⟨changeInputControl M hM fixed m n control, rfl⟩, ?_⟩
    rw [changeInputControl_output M hM fixed m n control]
  · rintro ⟨z, ⟨control, rfl⟩, rfl⟩
    refine ⟨_, ⟨changeInputControl M hM fixed n m control, rfl⟩, ?_⟩
    rw [changeInputControl_output M hM fixed n m control]

end General

/-! ## Simultaneous Fourier permutations -/

section FourierPermutation

variable {N : ℕ} [NeZero N]

/-- The simultaneous port permutation selected by the DFT ramp label `l`. -/
def dftSimultaneousRoute (l n : ZMod N) : ZMod N := l - n

/-- Every DFT routing configuration is an involution, as reciprocity requires. -/
theorem dftSimultaneousRoute_involution (l n : ZMod N) :
    dftSimultaneousRoute l (dftSimultaneousRoute l n) = n := by
  simp only [dftSimultaneousRoute]
  ring

/-- The self-reflected DFT ports are exactly the solutions of `2n = l`. -/
theorem dftSimultaneousRoute_fixed_iff (l n : ZMod N) :
    dftSimultaneousRoute l n = n ↔ n + n = l := by
  simp only [dftSimultaneousRoute]
  constructor
  · intro h
    exact (sub_eq_iff_eq_add.mp h).symm
  · intro h
    exact sub_eq_iff_eq_add.mpr h.symm

/-- A single Fourier phase ramp sends every input `n` to `l - n`; this is
the simultaneous version of the arbitrary-`N` pairwise-routing theorem. -/
theorem dft_simultaneous_routing (fixed : ZMod N → Circle)
    (global : Circle) (l n : ZMod N) :
    roundTrip fixed (controlFor fixed global l) (basis n) =
      fun q ↦ (global : ℂ) * basis (dftSimultaneousRoute l n) q := by
  simpa only [dftSimultaneousRoute, show (l - n) + n = l by ring] using
    pairwise_routing fixed global n (l - n)

/-- For the label `l = m + n`, the same Fourier setting swaps the selected
ports `n` and `m` (while its action on all remaining ports is fixed). -/
theorem dftSimultaneousRoute_contains_pair (n m : ZMod N) :
    dftSimultaneousRoute (m + n) n = m ∧
      dftSimultaneousRoute (m + n) m = n := by
  constructor <;> simp only [dftSimultaneousRoute] <;> ring

end FourierPermutation

/-! ## Binary Walsh matchings -/

section WalshMatching

variable {k : ℕ}

/-- Binary port labels for a Walsh transform of order `2^k`. -/
abbrev WalshPort (k : ℕ) := Fin k → ZMod 2

/-- The simultaneous port permutation selected by Walsh label `a`. -/
def walshRoute (a x : WalshPort k) : WalshPort k := x + a

lemma walshRoute_apply (a x : WalshPort k) (t : Fin k) :
    walshRoute a x t = x t + a t := rfl

/-- Every Walsh routing configuration is an involution. -/
theorem walshRoute_involution (a x : WalshPort k) :
    walshRoute a (walshRoute a x) = x := by
  funext t
  simp only [walshRoute_apply]
  rw [add_assoc, ZModModule.add_self, add_zero]

/-- A nonzero Walsh label has no self-reflected port. -/
theorem walshRoute_no_fixed_point (a : WalshPort k) (ha : a ≠ 0)
    (x : WalshPort k) : walshRoute a x ≠ x := by
  intro h
  apply ha
  funext t
  have ht := congrFun h t
  simp [walshRoute] at ht
  exact ht

/-- Every requested distinct pair belongs to one of the reflection-free
Walsh matchings: choose the translation label `a = m - n`. -/
theorem walshRoute_contains_pair (n m : WalshPort k) :
    walshRoute (m - n) n = m := by
  funext t
  simp [walshRoute]

theorem walshRoute_pair_label_nonzero (n m : WalshPort k) (hnm : n ≠ m) :
    m - n ≠ 0 := sub_ne_zero.mpr (Ne.symm hnm)

/-- The phase signature used by the Walsh control law. -/
noncomputable def walshControlPhase (a j : WalshPort k) : Circle :=
  ZMod.toCircle (∑ t, j t * a t)

/-- Walsh reflector coefficients square to one, hence are exactly the binary
phases `+1` or `-1` (relative phases `0` or `π`). -/
theorem walshControlPhase_sq (a j : WalshPort k) :
    walshControlPhase a j ^ 2 = 1 := by
  rw [pow_two]
  change ZMod.toCircle (∑ t, j t * a t) *
      ZMod.toCircle (∑ t, j t * a t) = 1
  rw [← ZMod.toCircle.map_add_eq_mul, ZModModule.add_self]
  exact (ZMod.toCircle (N := 2)).map_zero_eq_one

/-- The binary dot product with a fixed Walsh label, as an additive map. -/
def walshDotHom (x : WalshPort k) : WalshPort k →+ ZMod 2 where
  toFun j := ∑ t, j t * x t
  map_zero' := by simp
  map_add' u v := by
    simp only [Pi.add_apply, add_mul, Finset.sum_add_distrib]

/-- The complex Walsh character indexed by the binary vector `x`. -/
noncomputable def walshCharacter (x : WalshPort k) :
    AddChar (WalshPort k) ℂ :=
  ZMod.stdAddChar.compAddMonoidHom (walshDotHom x)

@[simp] lemma walshDotHom_single (x : WalshPort k) (t : Fin k) :
    walshDotHom x (Pi.single t 1) = x t := by
  classical
  change ∑ u, (Pi.single t (1 : ZMod 2) : WalshPort k) u * x u = x t
  rw [Finset.sum_eq_single t]
  · simp
  · intro u _ hut
    simp [Pi.single_apply, hut]
  · simp

/-- Distinct binary labels give distinct Walsh characters. -/
theorem walshCharacter_injective :
    Function.Injective (walshCharacter (k := k)) := by
  intro x y h
  funext t
  have ht := congrArg (fun χ : AddChar (WalshPort k) ℂ ↦ χ (Pi.single t 1)) h
  simp only [walshCharacter, AddChar.compAddMonoidHom_apply,
    walshDotHom_single] at ht
  exact ZMod.injective_stdAddChar ht

@[simp] lemma walshCharacter_zero :
    walshCharacter (0 : WalshPort k) = 0 := by
  apply AddChar.ext
  intro j
  simp [walshCharacter, walshDotHom]

lemma walshCharacter_add (x y : WalshPort k) :
    walshCharacter (x + y) = walshCharacter x * walshCharacter y := by
  apply AddChar.ext
  intro j
  simp only [walshCharacter, AddChar.compAddMonoidHom_apply,
    walshDotHom, Pi.add_apply, mul_add, Finset.sum_add_distrib]
  exact (ZMod.stdAddChar (N := 2)).map_add_eq_mul _ _

lemma walshCharacter_eq_zero_iff (x : WalshPort k) :
    walshCharacter x = 0 ↔ x = 0 := by
  rw [← walshCharacter_zero]
  exact walshCharacter_injective.eq_iff

/-- The normalized Walsh round trip, written directly as an expectation over
the internal binary arms.  This is exactly `Wᵀ diag(r) W`. -/
noncomputable def walshAverageRoundTrip (global : Circle)
    (a y x : WalshPort k) : ℂ :=
  𝔼 j, walshCharacter y j * (global : ℂ) *
    walshCharacter a j * walshCharacter x j

lemma walshRoute_condition (a y x : WalshPort k) :
    y + a + x = 0 ↔ y = walshRoute a x := by
  rw [add_eq_zero_iff_eq_neg, ZModModule.neg_eq_self]
  change y + a = x ↔ y = x + a
  constructor
  · intro h
    calc
      y = y + (a + a) := by rw [ZModModule.add_self, add_zero]
      _ = (y + a) + a := by abel
      _ = x + a := by rw [h]
  · intro h
    calc
      y + a = (x + a) + a := by rw [h]
      _ = x + (a + a) := by abel
      _ = x := by rw [ZModModule.add_self, add_zero]

/-- A single binary Walsh control pattern simultaneously realizes the
permutation `x ↦ x ⊕ a` for every input. -/
theorem walsh_simultaneous_routing (global : Circle)
    (a y x : WalshPort k) :
    walshAverageRoundTrip global a y x =
      (global : ℂ) * if y = walshRoute a x then 1 else 0 := by
  have hproduct : ∀ j : WalshPort k,
      walshCharacter y j * (global : ℂ) *
          walshCharacter a j * walshCharacter x j =
        (global : ℂ) * walshCharacter (y + a + x) j := by
    intro j
    rw [walshCharacter_add, walshCharacter_add]
    simp only [AddChar.mul_apply]
    ring
  simp_rw [walshAverageRoundTrip, hproduct]
  rw [← Finset.mul_expect]
  rw [AddChar.expect_eq_ite]
  simp only [walshCharacter_eq_zero_iff, walshRoute_condition]

/-- Every nonidentity Walsh configuration is therefore a complete matching:
it has no fixed points and pairs each port with its involutive partner. -/
theorem walsh_simultaneous_complete_matching (global : Circle)
    (a : WalshPort k) (ha : a ≠ 0) :
    (∀ x, walshAverageRoundTrip global a (walshRoute a x) x = global) ∧
    (∀ x, walshRoute a x ≠ x) ∧
    (∀ x, walshRoute a (walshRoute a x) = x) := by
  constructor
  · intro x
    rw [walsh_simultaneous_routing]
    simp
  exact ⟨walshRoute_no_fixed_point a ha, walshRoute_involution a⟩

end WalshMatching

/-! ## The continuous four-port equal-split family -/

/-- The normalized one-parameter order-four complex Hadamard family, with
`a = exp(i α)`. -/
noncomputable def fourPortFamily (a : Circle) : Matrix (Fin 4) (Fin 4) ℂ :=
  fun j k ↦ (2 : ℂ)⁻¹ *
    (![![1, 1, 1, 1],
       ![1, (a : ℂ), -1, -(a : ℂ)],
       ![1, -1, 1, -1],
       ![1, -(a : ℂ), -1, (a : ℂ)]] : Matrix (Fin 4) (Fin 4) ℂ) j k

lemma circle_mul_conj (a : Circle) : (a : ℂ) * conj (a : ℂ) = 1 := by
  rw [← Circle.coe_inv_eq_conj]
  norm_cast
  simp

lemma circle_mul_star (a : Circle) :
    (a : ℂ) * starRingEnd ℂ (a : ℂ) = 1 :=
  circle_mul_conj a

lemma circle_coe_inv_eq_star (a : Circle) :
    ((a⁻¹ : Circle) : ℂ) = starRingEnd ℂ (a : ℂ) :=
  Circle.coe_inv_eq_conj a

theorem fourPortFamily_columnUnitary (a : Circle) :
    ColumnUnitary (fourPortFamily a) := by
  intro q m
  fin_cases q <;> fin_cases m
  all_goals simp [fourPortFamily, Fin.sum_univ_four, map_add, map_mul,
    map_neg, Complex.conj_ofNat] <;> norm_num
  all_goals ring_nf
  all_goals rw [circle_mul_star]
  all_goals ring

theorem fourPortFamily_equalSplitNonzero (a : Circle) :
    EqualSplitNonzero (fourPortFamily a) := by
  constructor
  · intro j q m
    fin_cases j <;> fin_cases q <;> fin_cases m <;>
      simp [fourPortFamily, norm_mul]
  · intro j q
    fin_cases j <;> fin_cases q <;> simp [fourPortFamily]

theorem fourPortFamily_at_one : fourPortFamily 1 = walsh4 := by
  funext j q
  fin_cases j <;> fin_cases q <;> simp [fourPortFamily, walsh4]

theorem fourPortFamily_at_i : fourPortFamily circleI = fourier4 := by
  funext j q
  fin_cases j <;> fin_cases q <;> simp [fourPortFamily, fourier4, circleI]

/-- Reversing the family phase conjugates every matrix element. -/
theorem fourPortFamily_inv (a : Circle) (j q : Fin 4) :
    fourPortFamily a⁻¹ j q = conj (fourPortFamily a j q) := by
  fin_cases j <;> fin_cases q <;>
    simp [fourPortFamily, map_mul, map_inv₀, Complex.conj_ofNat]
  all_goals exact circle_coe_inv_eq_star a

/-- Every member of the family supports all sixteen ideal pairwise routes. -/
theorem fourPortFamily_pairwise_routing (a : Circle)
    (fixed : Fin 4 → Circle) (global : Circle) (n m : Fin 4) :
    (fun q ↦ generalOutput (fourPortFamily a) fixed
      (generalRoutingControl (fourPortFamily a)
        (fourPortFamily_equalSplitNonzero a) fixed global n m) n q) =
      fun q ↦ (global : ℂ) * if q = m then 1 else 0 :=
  general_pairwise_routing (fourPortFamily a)
    (fourPortFamily_columnUnitary a)
    (fourPortFamily_equalSplitNonzero a) fixed global n m

/-- The reachable four-port power region is independent of which physical
input is illuminated, including arbitrary fixed reciprocal arm phases. -/
theorem fourPortFamily_reachablePowers_input_independent (a : Circle)
    (fixed : Fin 4 → Circle) (n m : Fin 4) :
    reachablePowers (fourPortFamily a) fixed n =
      reachablePowers (fourPortFamily a) fixed m :=
  reachablePowers_input_independent (fourPortFamily a)
    (fourPortFamily_equalSplitNonzero a) fixed n m

/-! ## Explicit four-port Walsh simultaneous configurations -/

/-- The unit phasor `-1`. -/
noncomputable def walshCircleNegOne : Circle :=
  ⟨-1, mem_sphere_zero_iff_norm.2 (by norm_num)⟩

noncomputable def walshControl01_23 : Fin 4 → Circle :=
  ![1, 1, walshCircleNegOne, walshCircleNegOne]
noncomputable def walshControl02_13 : Fin 4 → Circle :=
  ![1, walshCircleNegOne, 1, walshCircleNegOne]
noncomputable def walshControl03_12 : Fin 4 → Circle :=
  ![1, walshCircleNegOne, walshCircleNegOne, 1]

def permutation01_23 : Matrix (Fin 4) (Fin 4) ℂ :=
  ![![0, 1, 0, 0], ![1, 0, 0, 0], ![0, 0, 0, 1], ![0, 0, 1, 0]]

def permutation02_13 : Matrix (Fin 4) (Fin 4) ℂ :=
  ![![0, 0, 1, 0], ![0, 0, 0, 1], ![1, 0, 0, 0], ![0, 1, 0, 0]]

def permutation03_12 : Matrix (Fin 4) (Fin 4) ℂ :=
  ![![0, 0, 0, 1], ![0, 0, 1, 0], ![0, 1, 0, 0], ![1, 0, 0, 0]]

/-- The Walsh setting `(0,0,π,π)` simultaneously swaps ports `0↔1` and
`2↔3`, with no self-reflections. -/
theorem walsh4_simultaneous_01_23 :
    generalRoundTrip walsh4 (fun _ ↦ 1) walshControl01_23 = permutation01_23 := by
  funext q n
  fin_cases q <;> fin_cases n <;>
    simp [generalRoundTrip, generalOutput, generalArmPhase, walsh4,
      walshControl01_23, walshCircleNegOne, permutation01_23, Fin.sum_univ_four] <;> ring

/-- The Walsh setting `(0,π,0,π)` simultaneously swaps ports `0↔2` and
`1↔3`, with no self-reflections. -/
theorem walsh4_simultaneous_02_13 :
    generalRoundTrip walsh4 (fun _ ↦ 1) walshControl02_13 = permutation02_13 := by
  funext q n
  fin_cases q <;> fin_cases n <;>
    simp [generalRoundTrip, generalOutput, generalArmPhase, walsh4,
      walshControl02_13, walshCircleNegOne, permutation02_13, Fin.sum_univ_four] <;> ring

/-- The Walsh setting `(0,π,π,0)` simultaneously swaps ports `0↔3` and
`1↔2`, with no self-reflections. -/
theorem walsh4_simultaneous_03_12 :
    generalRoundTrip walsh4 (fun _ ↦ 1) walshControl03_12 = permutation03_12 := by
  funext q n
  fin_cases q <;> fin_cases n <;>
    simp [generalRoundTrip, generalOutput, generalArmPhase, walsh4,
      walshControl03_12, walshCircleNegOne, permutation03_12, Fin.sum_univ_four] <;> ring

/-! ## Explicit planar balanced-MMI factorization -/

/-- Two disjoint symmetric 50:50 MMI couplers. -/
noncomputable def balancedMMIStage4 : Matrix (Fin 4) (Fin 4) ℂ :=
  fun j q ↦ ((Real.sqrt 2 : ℂ)⁻¹) *
    (![![1, I, 0, 0], ![I, 1, 0, 0],
       ![0, 0, 1, I], ![0, 0, I, 1]] : Matrix (Fin 4) (Fin 4) ℂ) j q

/-- The planar shuffle exchanging the two middle rails. -/
def planarShuffle4 : Matrix (Fin 4) (Fin 4) ℂ :=
  ![![1, 0, 0, 0], ![0, 0, 1, 0],
    ![0, 1, 0, 0], ![0, 0, 0, 1]]

/-- The phase-only interstage section `(1,-i,-i,-a)`. -/
noncomputable def interstagePhase4 (a : Circle) : Matrix (Fin 4) (Fin 4) ℂ :=
  ![![1, 0, 0, 0], ![0, -I, 0, 0],
    ![0, 0, -I, 0], ![0, 0, 0, -(a : ℂ)]]

/-- Fixed external phases `diag(1,i,1,i)`. -/
def externalPhase4 : Matrix (Fin 4) (Fin 4) ℂ :=
  ![![1, 0, 0, 0], ![0, I, 0, 0],
    ![0, 0, 1, 0], ![0, 0, 0, I]]

lemma balancedMMIScale_sq : ((Real.sqrt 2 : ℂ)⁻¹) ^ 2 = (2 : ℂ)⁻¹ := by
  rw [inv_pow]
  congr 1
  norm_cast
  exact Real.sq_sqrt (by norm_num)

set_option maxHeartbeats 2000000 in
/-- Exact matrix identity for the planar two-stage four-port circuit in
Supplementary Note 2.  The remaining matrices on the right are fixed port
phases and the middle-rail relabeling. -/
theorem planar_balancedMMI_factorization (a : Circle) :
    balancedMMIStage4 * interstagePhase4 a * planarShuffle4 * balancedMMIStage4 =
      externalPhase4 * planarShuffle4 * fourPortFamily a * planarShuffle4 *
        externalPhase4 := by
  funext j q
  fin_cases j <;> fin_cases q <;>
    simp [Matrix.mul_apply, balancedMMIStage4, interstagePhase4,
      planarShuffle4, externalPhase4, fourPortFamily, Fin.sum_univ_four]
  all_goals ring_nf
  all_goals rw [balancedMMIScale_sq]
  all_goals norm_num [I_mul_I]
  all_goals ring

end TFLNRouter
