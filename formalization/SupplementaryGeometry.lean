import FourPortWalsh
import Mathlib.Algebra.Module.ZMod
import Mathlib.Analysis.SpecialFunctions.Complex.Arg
import Mathlib.FieldTheory.Finiteness

/-!
# Remaining supplementary routing and reachable-region arguments

This file supplies machine-checked counterparts for the proof-like arguments
in Supplementary Notes 2 and 7 that are not stated as numbered theorems in the
manuscript.  It contains:

* the power-of-two obstruction for a regular reflection-free routing group;
* the algebra behind the exact three-port critical curve; and
* the algebraic four-port section construction.

The analytic definitions use explicit real coordinates.  This keeps the
formal statements close to the plotted power maps while avoiding any choice
of complex arguments.
-/

namespace TFLNRouter

open scoped BigOperators

/-! ## The power-of-two obstruction for reflection-free schedules

The manuscript first shows that the identity and the matching permutations
form an abelian group acting regularly on the ports.  Reciprocity makes every
element an involution.  The next theorem formalizes the final group-theoretic
step: a finite abelian group of exponent two is a vector space over
`ZMod 2`, and hence has cardinality a power of two. -/

theorem card_eq_two_pow_of_add_exponent_two
    (G : Type*) [Fintype G] [AddCommGroup G]
    (hexp : ∀ g : G, g + g = 0) :
    ∃ k : ℕ, Fintype.card G = 2 ^ k := by
  letI : Module (ZMod 2) G :=
    AddCommGroup.zmodModule (n := 2) (by
      intro g
      simpa [two_nsmul] using hexp g)
  refine ⟨Module.finrank (ZMod 2) G, ?_⟩
  simpa using (Module.card_eq_pow_finrank (K := ZMod 2) (V := G))

/-- Multiplicative form used for a group of routing permutations. -/
theorem card_eq_two_pow_of_involutive_commGroup
    (G : Type*) [Fintype G] [CommGroup G]
    (hinv : ∀ g : G, g * g = 1) :
    ∃ k : ℕ, Fintype.card G = 2 ^ k := by
  let A := Additive G
  have hA : ∀ g : A, g + g = 0 := by
    intro g
    exact hinv g
  simpa [A] using card_eq_two_pow_of_add_exponent_two A hA

/-- Evaluation at one port is a bijection for a free and transitive action.
This is the formal cardinality step behind the statement that a regular
routing group has as many elements as ports. -/
theorem card_eq_of_free_transitive_at
    (G ι : Type*) [Fintype G] [Fintype ι] [Group G] [MulAction G ι]
    (x₀ : ι)
    (hfree : ∀ g : G, g • x₀ = x₀ → g = 1)
    (htrans : ∀ y : ι, ∃ g : G, g • x₀ = y) :
    Fintype.card G = Fintype.card ι := by
  apply Fintype.card_congr
  refine Equiv.ofBijective (fun g : G ↦ g • x₀) ⟨?_, ?_⟩
  · intro g h hgh
    change g • x₀ = h • x₀ at hgh
    have hfix : (h⁻¹ * g) • x₀ = x₀ := by
      simp [mul_smul, hgh]
    have hone : h⁻¹ * g = 1 := hfree _ hfix
    exact (inv_mul_eq_one.mp hone).symm
  · intro y
    exact htrans y

/-- A regular abelian routing group made entirely of reciprocal involutions
can exist only on a power-of-two number of ports. -/
theorem port_card_eq_two_pow_of_regular_involutive_action
    (G ι : Type*) [Fintype G] [Fintype ι] [CommGroup G] [MulAction G ι]
    (x₀ : ι)
    (hfree : ∀ g : G, g • x₀ = x₀ → g = 1)
    (htrans : ∀ y : ι, ∃ g : G, g • x₀ = y)
    (hinv : ∀ g : G, g * g = 1) :
    ∃ k : ℕ, Fintype.card ι = 2 ^ k := by
  rcases card_eq_two_pow_of_involutive_commGroup G hinv with ⟨k, hk⟩
  refine ⟨k, ?_⟩
  rw [← card_eq_of_free_transitive_at G ι x₀ hfree htrans]
  exact hk

/-! ## Three-port critical-curve algebra

The following definitions are the explicit power map and the parametric
critical curve from Supplementary Note 2. -/

noncomputable def threePowerOne (α β : ℝ) : ℝ :=
  (3 + 2 * Real.cos α + 2 * Real.cos β +
    2 * Real.cos (α - β)) / 9

noncomputable def threePowerTwo (α β : ℝ) : ℝ :=
  (3 + 2 * Real.cos (α + 2 * Real.pi / 3) +
    2 * Real.cos (β + 4 * Real.pi / 3) +
    2 * Real.cos (α - β - 2 * Real.pi / 3)) / 9

noncomputable def threePowerThree (α β : ℝ) : ℝ :=
  (3 + 2 * Real.cos (α + 4 * Real.pi / 3) +
    2 * Real.cos (β + 2 * Real.pi / 3) +
    2 * Real.cos (α - β + 2 * Real.pi / 3)) / 9

/-- The trigonometric factor in the Jacobian determinant. -/
noncomputable def threeJacobianFactor (α β : ℝ) : ℝ :=
  Real.sin (α - 2 * β) + Real.sin (α + β) -
    Real.sin (2 * α - β)

/-- One coordinate of the closed critical-value curve. -/
noncomputable def threeBoundaryCoordinate (t : ℝ) : ℝ :=
  (1 + 2 * Real.cos t) ^ 2 / 9

noncomputable def threePowerVector (α β : ℝ) : Fin 3 → ℝ :=
  ![threePowerOne α β, threePowerTwo α β, threePowerThree α β]

noncomputable def threeBoundaryVector (t : ℝ) : Fin 3 → ℝ :=
  ![threeBoundaryCoordinate t,
    threeBoundaryCoordinate (t - 2 * Real.pi / 3),
    threeBoundaryCoordinate (t + 2 * Real.pi / 3)]

lemma real_cos_two_pi_div_three :
    Real.cos (2 * Real.pi / 3) = -(1 : ℝ) / 2 := by
  rw [show 2 * Real.pi / 3 = Real.pi - Real.pi / 3 by ring,
    Real.cos_pi_sub, Real.cos_pi_div_three]
  ring

lemma real_sin_two_pi_div_three :
    Real.sin (2 * Real.pi / 3) = Real.sqrt 3 / 2 := by
  rw [show 2 * Real.pi / 3 = Real.pi - Real.pi / 3 by ring,
    Real.sin_pi_sub, Real.sin_pi_div_three]

lemma real_cos_four_pi_div_three :
    Real.cos (4 * Real.pi / 3) = -(1 : ℝ) / 2 := by
  rw [show 4 * Real.pi / 3 = 2 * Real.pi - 2 * Real.pi / 3 by ring,
    Real.cos_two_pi_sub, real_cos_two_pi_div_three]

lemma real_sin_four_pi_div_three :
    Real.sin (4 * Real.pi / 3) = -(Real.sqrt 3) / 2 := by
  rw [show 4 * Real.pi / 3 = 2 * Real.pi - 2 * Real.pi / 3 by ring,
    Real.sin_two_pi_sub, real_sin_two_pi_div_three]
  ring

/-- Exact factorization of the trigonometric factor in the Jacobian. -/
theorem three_jacobian_factorization (α β : ℝ) :
    threeJacobianFactor α β =
      2 * Real.sin (α - β / 2) *
        (Real.cos (3 * β / 2) - Real.cos (α - β / 2)) := by
  rw [threeJacobianFactor]
  rw [show α - 2 * β = (α - β / 2) - 3 * β / 2 by ring,
    show α + β = (α - β / 2) + 3 * β / 2 by ring,
    show 2 * α - β = 2 * (α - β / 2) by ring,
    Real.sin_sub, Real.sin_add, Real.sin_two_mul]
  ring

/-- The four partial derivatives used in the manuscript's Jacobian. -/
noncomputable def threeP1α (α β : ℝ) : ℝ :=
  (-2 * Real.sin α - 2 * Real.sin (α - β)) / 9

noncomputable def threeP1β (α β : ℝ) : ℝ :=
  (-2 * Real.sin β + 2 * Real.sin (α - β)) / 9

noncomputable def threeP2α (α β : ℝ) : ℝ :=
  (-2 * Real.sin (α + 2 * Real.pi / 3) -
    2 * Real.sin (α - β - 2 * Real.pi / 3)) / 9

noncomputable def threeP2β (α β : ℝ) : ℝ :=
  (-2 * Real.sin (β + 4 * Real.pi / 3) +
    2 * Real.sin (α - β - 2 * Real.pi / 3)) / 9

theorem hasDerivAt_threePowerOne_α (α β : ℝ) :
    HasDerivAt (fun a ↦ threePowerOne a β) (threeP1α α β) α := by
  simp only [threePowerOne, threeP1α]
  convert ((((hasDerivAt_const α (3 : ℝ)).add
      ((Real.hasDerivAt_cos α).const_mul 2)).add
      (hasDerivAt_const α (2 * Real.cos β))).add
      (((Real.hasDerivAt_cos (α - β)).comp α
        ((hasDerivAt_id α).sub_const β)).const_mul 2) |>.div_const 9) using 1 <;>
    ring

theorem hasDerivAt_threePowerOne_β (α β : ℝ) :
    HasDerivAt (fun b ↦ threePowerOne α b) (threeP1β α β) β := by
  simp only [threePowerOne, threeP1β]
  convert ((((hasDerivAt_const β (3 + 2 * Real.cos α : ℝ)).add
      ((Real.hasDerivAt_cos β).const_mul 2)).add
      (((Real.hasDerivAt_cos (α - β)).comp β
        ((hasDerivAt_const β α).sub (hasDerivAt_id β))).const_mul 2))
      |>.div_const 9) using 1 <;> ring

theorem hasDerivAt_threePowerTwo_α (α β : ℝ) :
    HasDerivAt (fun a ↦ threePowerTwo a β) (threeP2α α β) α := by
  simp only [threePowerTwo, threeP2α]
  convert ((((hasDerivAt_const α (3 : ℝ)).add
      (((Real.hasDerivAt_cos (α + 2 * Real.pi / 3)).comp α
        ((hasDerivAt_id α).add_const (2 * Real.pi / 3))).const_mul 2)).add
      (hasDerivAt_const α (2 * Real.cos (β + 4 * Real.pi / 3)))).add
      (((Real.hasDerivAt_cos (α - β - 2 * Real.pi / 3)).comp α
        (((hasDerivAt_id α).sub_const β).sub_const
          (2 * Real.pi / 3))).const_mul 2) |>.div_const 9) using 1 <;> ring

theorem hasDerivAt_threePowerTwo_β (α β : ℝ) :
    HasDerivAt (fun b ↦ threePowerTwo α b) (threeP2β α β) β := by
  simp only [threePowerTwo, threeP2β]
  convert ((((hasDerivAt_const β
      (3 + 2 * Real.cos (α + 2 * Real.pi / 3) : ℝ)).add
      (((Real.hasDerivAt_cos (β + 4 * Real.pi / 3)).comp β
        ((hasDerivAt_id β).add_const (4 * Real.pi / 3))).const_mul 2)).add
      (((Real.hasDerivAt_cos (α - β - 2 * Real.pi / 3)).comp β
        (((hasDerivAt_const β α).sub (hasDerivAt_id β)).sub_const
          (2 * Real.pi / 3))).const_mul 2)) |>.div_const 9) using 1 <;> ring

/-- Determinant of the explicitly differentiated two-coordinate power map. -/
noncomputable def threeJacobianDet (α β : ℝ) : ℝ :=
  threeP1α α β * threeP2β α β - threeP1β α β * threeP2α α β

/-- Exact Jacobian determinant quoted in Supplementary Note 2. -/
theorem three_jacobian_determinant (α β : ℝ) :
    threeJacobianDet α β =
      -(2 * Real.sqrt 3) / 81 * threeJacobianFactor α β := by
  rw [threeJacobianDet, threeP1α, threeP1β, threeP2α, threeP2β,
    threeJacobianFactor]
  rw [show α - β - 2 * Real.pi / 3 =
        (α - β) - 2 * Real.pi / 3 by ring]
  rw [Real.sin_add α (2 * Real.pi / 3),
    Real.sin_add β (4 * Real.pi / 3),
    Real.sin_sub (α - β) (2 * Real.pi / 3),
    real_cos_two_pi_div_three, real_sin_two_pi_div_three,
    real_cos_four_pi_div_three, real_sin_four_pi_div_three]
  rw [Real.sin_sub α β, Real.sin_sub α (2 * β),
    Real.sin_add α β, Real.sin_sub (2 * α) β,
    Real.sin_two_mul α, Real.cos_two_mul α,
    Real.sin_two_mul β, Real.cos_two_mul β,
    Real.cos_sub α β]
  ring_nf
  have hα : Real.cos α ^ 2 = 1 - Real.sin α ^ 2 := by
    nlinarith [Real.sin_sq_add_cos_sq α]
  have hβ : Real.cos β ^ 2 = 1 - Real.sin β ^ 2 := by
    nlinarith [Real.sin_sq_add_cos_sq β]
  rw [hα, hβ]
  ring

/-- Complete description of the Jacobian's critical set by the three line
families printed in Supplementary Note 2. -/
theorem three_jacobian_zero_iff_critical_families (α β : ℝ) :
    threeJacobianDet α β = 0 ↔
      (∃ k : ℤ, α = β / 2 + (k : ℝ) * Real.pi) ∨
      (∃ k : ℤ, α = 2 * β + 2 * (k : ℝ) * Real.pi) ∨
      (∃ k : ℤ, α = -β + 2 * (k : ℝ) * Real.pi) := by
  rw [three_jacobian_determinant, three_jacobian_factorization]
  have hsqrt : Real.sqrt 3 ≠ 0 := ne_of_gt (Real.sqrt_pos.2 (by norm_num))
  have hcoef : -(2 * Real.sqrt 3) / 81 ≠ 0 := by
    exact div_ne_zero (neg_ne_zero.mpr (mul_ne_zero (by norm_num) hsqrt))
      (by norm_num)
  simp only [mul_eq_zero, hcoef, OfNat.ofNat_ne_zero, false_or]
  rw [Real.sin_eq_zero_iff]
  constructor
  · rintro (⟨k, hk⟩ | hcos)
    · left
      refine ⟨k, ?_⟩
      nlinarith
    · right
      have hcc : Real.cos (3 * β / 2) = Real.cos (α - β / 2) :=
        sub_eq_zero.mp hcos
      rw [Real.cos_eq_cos_iff] at hcc
      rcases hcc with ⟨k, hk | hk⟩
      · left
        refine ⟨k, ?_⟩
        nlinarith
      · right
        refine ⟨k, ?_⟩
        nlinarith
  · rintro (⟨k, hk⟩ | ⟨k, hk⟩ | ⟨k, hk⟩)
    · left
      refine ⟨k, ?_⟩
      nlinarith
    · right
      apply sub_eq_zero.mpr
      rw [Real.cos_eq_cos_iff]
      refine ⟨k, Or.inl ?_⟩
      nlinarith
    · right
      apply sub_eq_zero.mpr
      rw [Real.cos_eq_cos_iff]
      refine ⟨k, Or.inr ?_⟩
      nlinarith

/-- One critical-line family maps to the first coordinate of the exact
three-port boundary curve. -/
theorem threePowerOne_on_critical_family (t : ℝ) :
    threePowerOne (2 * t) t = threeBoundaryCoordinate t := by
  rw [threePowerOne, threeBoundaryCoordinate]
  rw [show 2 * t - t = t by ring, Real.cos_two_mul]
  nlinarith [Real.sin_sq_add_cos_sq t]

/-- The second coordinate on the same critical family is the cyclic shift
appearing in the manuscript's boundary parametrization. -/
theorem threePowerTwo_on_critical_family (t : ℝ) :
    threePowerTwo (2 * t) t =
      threeBoundaryCoordinate (t - 2 * Real.pi / 3) := by
  rw [threePowerTwo, threeBoundaryCoordinate]
  rw [show 2 * t + 2 * Real.pi / 3 =
        2 * (t - 2 * Real.pi / 3) + 2 * Real.pi by ring,
    show t + 4 * Real.pi / 3 =
        (t - 2 * Real.pi / 3) + 2 * Real.pi by ring,
    show 2 * t - t - 2 * Real.pi / 3 =
        t - 2 * Real.pi / 3 by ring,
    Real.cos_add_two_pi, Real.cos_add_two_pi, Real.cos_two_mul]
  nlinarith [Real.sin_sq_add_cos_sq (t - 2 * Real.pi / 3)]

/-- The third coordinate is the opposite cyclic shift. -/
theorem threePowerThree_on_critical_family (t : ℝ) :
    threePowerThree (2 * t) t =
      threeBoundaryCoordinate (t + 2 * Real.pi / 3) := by
  rw [threePowerThree, threeBoundaryCoordinate]
  rw [show 2 * t + 4 * Real.pi / 3 =
        2 * (t + 2 * Real.pi / 3) by ring,
    show t + 2 * Real.pi / 3 = t + 2 * Real.pi / 3 by ring,
    show 2 * t - t + 2 * Real.pi / 3 =
        t + 2 * Real.pi / 3 by ring,
    Real.cos_two_mul]
  nlinarith [Real.sin_sq_add_cos_sq (t + 2 * Real.pi / 3)]

/-- Vector form of the manuscript's exact parametrized critical-value curve. -/
theorem threePowerVector_on_critical_family (t : ℝ) :
    threePowerVector (2 * t) t = threeBoundaryVector t := by
  funext j
  fin_cases j
  · exact threePowerOne_on_critical_family t
  · exact threePowerTwo_on_critical_family t
  · exact threePowerThree_on_critical_family t

/-! ## Four-port section-coordinate algebra

These are the real coordinates used in Supplementary Note 7. -/

noncomputable def fourSectionS (φ ψ : ℝ) : ℝ :=
  (Real.cos φ ^ 2 + Real.cos ψ ^ 2) / 2

noncomputable def fourSectionX (φ ψ χ : ℝ) : ℝ :=
  Real.cos φ * Real.cos ψ * Real.cos χ

noncomputable def fourSectionY (α φ ψ χ : ℝ) : ℝ :=
  Real.sin φ * Real.sin ψ * Real.cos (χ + α)

noncomputable def fourSectionK (φ ψ : ℝ) : ℝ :=
  Real.cos φ ^ 2 * Real.cos ψ ^ 2

noncomputable def fourSectionEll (φ ψ : ℝ) : ℝ :=
  Real.sin φ ^ 2 * Real.sin ψ ^ 2

/-- The four powers are recovered uniquely from the section coordinates. -/
noncomputable def powersFromSection (s x y : ℝ) : Fin 4 → ℝ :=
  ![(s + x) / 2, (1 - s + y) / 2,
    (s - x) / 2, (1 - s - y) / 2]

theorem powersFromSection_sum (s x y : ℝ) :
    ∑ j, powersFromSection s x y j = 1 := by
  simp [powersFromSection, Fin.sum_univ_four]
  ring

/-- Positivity of all four powers is exactly the rectangular section bound. -/
theorem powersFromSection_nonnegative_iff (s x y : ℝ) :
    (∀ j, 0 ≤ powersFromSection s x y j) ↔
      |x| ≤ s ∧ |y| ≤ 1 - s := by
  constructor
  · intro h
    have h1 := h (0 : Fin 4)
    have h2 := h (1 : Fin 4)
    have h3 := h (2 : Fin 4)
    have h4 := h (3 : Fin 4)
    simp only [powersFromSection, Matrix.cons_val_zero,
      Matrix.cons_val_one, Matrix.cons_val_two, Matrix.cons_val_three] at h1 h2 h3 h4
    change 0 ≤ (s - x) / 2 at h3
    change 0 ≤ (1 - s - y) / 2 at h4
    constructor
    · rw [abs_le]
      constructor <;> linarith
    · rw [abs_le]
      constructor <;> linarith
  · rintro ⟨hx, hy⟩
    rw [abs_le] at hx hy
    intro j
    fin_cases j <;> simp [powersFromSection] <;>
      linarith [hx.1, hx.2, hy.1, hy.2]

/-- Algebraic identity `ℓ = 1 - 2s + k` used to parameterize every
constant-height section. -/
theorem fourSectionEll_eq (φ ψ : ℝ) :
    fourSectionEll φ ψ =
      1 - 2 * fourSectionS φ ψ + fourSectionK φ ψ := by
  rw [fourSectionEll, fourSectionS, fourSectionK]
  nlinarith [Real.sin_sq_add_cos_sq φ, Real.sin_sq_add_cos_sq ψ]

/-- Exact allowed interval for the auxiliary product `k`: every phase pair
obeys the two endpoint bounds quoted in Supplementary Note 7. -/
theorem fourSectionK_bounds (φ ψ : ℝ) :
    max 0 (2 * fourSectionS φ ψ - 1) ≤ fourSectionK φ ψ ∧
      fourSectionK φ ψ ≤ fourSectionS φ ψ ^ 2 := by
  have hcφ0 : 0 ≤ Real.cos φ ^ 2 := sq_nonneg _
  have hcψ0 : 0 ≤ Real.cos ψ ^ 2 := sq_nonneg _
  have hcφ1 : Real.cos φ ^ 2 ≤ 1 := by
    nlinarith [Real.sin_sq_add_cos_sq φ, sq_nonneg (Real.sin φ)]
  have hcψ1 : Real.cos ψ ^ 2 ≤ 1 := by
    nlinarith [Real.sin_sq_add_cos_sq ψ, sq_nonneg (Real.sin ψ)]
  rw [fourSectionS, fourSectionK]
  constructor
  · rw [max_le_iff]
    constructor
    · positivity
    · nlinarith [mul_nonneg (sub_nonneg.mpr hcφ1) (sub_nonneg.mpr hcψ1)]
  · nlinarith [sq_nonneg (Real.cos φ ^ 2 - Real.cos ψ ^ 2)]

/-! ### Exact curve-union description of a fixed-height section -/

/-- A point belongs to the union of the two fixed-`k` ellipse curves used in
Supplementary Note 7.  The disjunction records the independently selectable
relative sign of the two arm-pair products. -/
def FourSectionCurveUnion (α s x y : ℝ) : Prop :=
  0 ≤ s ∧ s ≤ 1 ∧
    ∃ k : ℝ,
      max 0 (2 * s - 1) ≤ k ∧ k ≤ s ^ 2 ∧
        ∃ χ : ℝ,
          x = Real.sqrt k * Real.cos χ ∧
            (y = Real.sqrt (1 - 2 * s + k) * Real.cos (χ + α) ∨
             y = -Real.sqrt (1 - 2 * s + k) * Real.cos (χ + α))

/-- The admissible interval for `k` is also sufficient: it is exactly the
range of products `a b` for two numbers `a,b ∈ [0,1]` with average `s`.
This is the reconstruction needed for the converse direction. -/
theorem section_product_roots_exist (s k : ℝ)
    (hs0 : 0 ≤ s) (hs1 : s ≤ 1)
    (hk0 : max 0 (2 * s - 1) ≤ k) (hk1 : k ≤ s ^ 2) :
    ∃ a b : ℝ,
      0 ≤ a ∧ a ≤ 1 ∧ 0 ≤ b ∧ b ≤ 1 ∧
      (a + b) / 2 = s ∧ a * b = k := by
  let d := Real.sqrt (s ^ 2 - k)
  have hrad : 0 ≤ s ^ 2 - k := sub_nonneg.mpr hk1
  have hd0 : 0 ≤ d := Real.sqrt_nonneg _
  have hd2 : d ^ 2 = s ^ 2 - k := Real.sq_sqrt hrad
  have hk_nonneg : 0 ≤ k := le_trans (le_max_left _ _) hk0
  have hds : d ≤ s := by
    nlinarith [hk_nonneg]
  have hklinear : 2 * s - 1 ≤ k := le_trans (le_max_right _ _) hk0
  have hd_one : d ≤ 1 - s := by
    have hsq : d ^ 2 ≤ (1 - s) ^ 2 := by
      rw [hd2]
      nlinarith
    nlinarith
  refine ⟨s + d, s - d, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · linarith
  · linarith
  · linarith
  · linarith
  · ring
  · rw [show (s + d) * (s - d) = s ^ 2 - d ^ 2 by ring, hd2]
    ring

/-- Every physical phase setting lies on one of the curves in the exact
fixed-height union.  Possible signs of the cosine and sine products are
absorbed into a `π` shift of `χ` and the explicit two-branch sign. -/
theorem fourSection_phases_mem_curveUnion (α φ ψ χ : ℝ) :
    FourSectionCurveUnion α (fourSectionS φ ψ)
      (fourSectionX φ ψ χ) (fourSectionY α φ ψ χ) := by
  let s := fourSectionS φ ψ
  let k := fourSectionK φ ψ
  let A := Real.cos φ * Real.cos ψ
  let B := Real.sin φ * Real.sin ψ
  have hs0 : 0 ≤ s := by
    simp only [s, fourSectionS]
    positivity
  have hs1 : s ≤ 1 := by
    simp only [s, fourSectionS]
    nlinarith [Real.sin_sq_add_cos_sq φ, Real.sin_sq_add_cos_sq ψ,
      sq_nonneg (Real.sin φ), sq_nonneg (Real.sin ψ)]
  have hkb := fourSectionK_bounds φ ψ
  have hsqrtk : Real.sqrt k = |A| := by
    rw [show k = A ^ 2 by simp only [k, A, fourSectionK]; ring,
      Real.sqrt_sq_eq_abs]
  have hradius : 1 - 2 * s + k = fourSectionEll φ ψ := by
    exact (fourSectionEll_eq φ ψ).symm
  have hsqrtℓ : Real.sqrt (1 - 2 * s + k) = |B| := by
    rw [hradius]
    rw [show fourSectionEll φ ψ = B ^ 2 by
      simp only [fourSectionEll, B]; ring, Real.sqrt_sq_eq_abs]
  rw [FourSectionCurveUnion]
  refine ⟨hs0, hs1, k, hkb.1, hkb.2, ?_⟩
  by_cases hA : 0 ≤ A
  · refine ⟨χ, ?_, ?_⟩
    · rw [hsqrtk, abs_of_nonneg hA]
      rfl
    · by_cases hB : 0 ≤ B
      · left
        rw [hsqrtℓ, abs_of_nonneg hB]
        rfl
      · right
        have hB' : B ≤ 0 := le_of_not_ge hB
        rw [hsqrtℓ, abs_of_nonpos hB']
        change B * Real.cos (χ + α) = - -B * Real.cos (χ + α)
        ring
  · have hA' : A ≤ 0 := le_of_not_ge hA
    have hshift : Real.cos ((χ + Real.pi) + α) =
        -Real.cos (χ + α) := by
      rw [show (χ + Real.pi) + α = (χ + α) + Real.pi by ring,
        Real.cos_add_pi]
    refine ⟨χ + Real.pi, ?_, ?_⟩
    · rw [hsqrtk, abs_of_nonpos hA', Real.cos_add_pi]
      change A * Real.cos χ = -A * -Real.cos χ
      ring
    · by_cases hB : 0 ≤ B
      · right
        rw [hsqrtℓ, abs_of_nonneg hB, hshift]
        change B * Real.cos (χ + α) = -B * -Real.cos (χ + α)
        ring
      · left
        have hB' : B ≤ 0 := le_of_not_ge hB
        rw [hsqrtℓ, abs_of_nonpos hB', hshift]
        change B * Real.cos (χ + α) = -B * -Real.cos (χ + α)
        ring

/-- Conversely, every point of the curve union is produced by physical
phases.  The proof explicitly reconstructs the two squared cosines and then
uses `arccos`; the negative branch is obtained by negating one phase. -/
theorem fourSection_curveUnion_exists_phases (α s x y : ℝ)
    (hcurve : FourSectionCurveUnion α s x y) :
    ∃ φ ψ χ : ℝ,
      fourSectionS φ ψ = s ∧
      fourSectionX φ ψ χ = x ∧
      fourSectionY α φ ψ χ = y := by
  rcases hcurve with ⟨hs0, hs1, k, hk0, hk1, χ, hx, hy⟩
  rcases section_product_roots_exist s k hs0 hs1 hk0 hk1 with
    ⟨a, b, ha0, ha1, hb0, hb1, habs, habk⟩
  let φ₀ := Real.arccos (Real.sqrt a)
  let ψ := Real.arccos (Real.sqrt b)
  have hsa0 : -(1 : ℝ) ≤ Real.sqrt a := by
    linarith [Real.sqrt_nonneg a]
  have hsa1 : Real.sqrt a ≤ 1 := by
    rw [Real.sqrt_le_one]
    exact ha1
  have hsb0 : -(1 : ℝ) ≤ Real.sqrt b := by
    linarith [Real.sqrt_nonneg b]
  have hsb1 : Real.sqrt b ≤ 1 := by
    rw [Real.sqrt_le_one]
    exact hb1
  have hcφ : Real.cos φ₀ = Real.sqrt a := Real.cos_arccos hsa0 hsa1
  have hcψ : Real.cos ψ = Real.sqrt b := Real.cos_arccos hsb0 hsb1
  have hsφ : Real.sin φ₀ = Real.sqrt (1 - a) := by
    dsimp [φ₀]
    rw [Real.sin_arccos, Real.sq_sqrt ha0]
  have hsψ : Real.sin ψ = Real.sqrt (1 - b) := by
    dsimp [ψ]
    rw [Real.sin_arccos, Real.sq_sqrt hb0]
  have hcosprod : Real.sqrt a * Real.sqrt b = Real.sqrt k := by
    rw [← Real.sqrt_mul ha0 b, habk]
  have honea : 0 ≤ 1 - a := sub_nonneg.mpr ha1
  have honeb : 0 ≤ 1 - b := sub_nonneg.mpr hb1
  have hell : (1 - a) * (1 - b) = 1 - 2 * s + k := by
    nlinarith
  have hsinprod : Real.sqrt (1 - a) * Real.sqrt (1 - b) =
      Real.sqrt (1 - 2 * s + k) := by
    rw [← Real.sqrt_mul honea (1 - b), hell]
  rcases hy with hy | hy
  · refine ⟨φ₀, ψ, χ, ?_, ?_, ?_⟩
    · rw [fourSectionS, hcφ, hcψ, Real.sq_sqrt ha0, Real.sq_sqrt hb0]
      exact habs
    · rw [fourSectionX, hcφ, hcψ, hcosprod, hx]
    · rw [fourSectionY, hsφ, hsψ, hsinprod, hy]
  · refine ⟨-φ₀, ψ, χ, ?_, ?_, ?_⟩
    · rw [fourSectionS, Real.cos_neg, hcφ, hcψ,
        Real.sq_sqrt ha0, Real.sq_sqrt hb0]
      exact habs
    · rw [fourSectionX, Real.cos_neg, hcφ, hcψ, hcosprod, hx]
    · rw [fourSectionY, Real.sin_neg, hsφ, hsψ]
      rw [show -Real.sqrt (1 - a) * Real.sqrt (1 - b) =
        -(Real.sqrt (1 - a) * Real.sqrt (1 - b)) by ring, hsinprod, hy]

/-- Exact equivalence between physical phase reachability and the union of
the analytic section curves.  This closes the first converse in the
four-port construction without any topological assumption. -/
theorem fourSection_reachable_iff_curveUnion (α s x y : ℝ) :
    (∃ φ ψ χ : ℝ,
      fourSectionS φ ψ = s ∧
      fourSectionX φ ψ χ = x ∧
      fourSectionY α φ ψ χ = y) ↔
      FourSectionCurveUnion α s x y := by
  constructor
  · rintro ⟨φ, ψ, χ, rfl, rfl, rfl⟩
    exact fourSection_phases_mem_curveUnion α φ ψ χ
  · exact fourSection_curveUnion_exists_phases α s x y

/-! ### Algebraic ellipse disks -/

/-- Denominator-free quadratic equation for the `+` ellipse boundary with
squared radii `k` and `ℓ`. -/
noncomputable def fourEllipseResidual
    (α k ℓ x y : ℝ) : ℝ :=
  ℓ * x ^ 2 + k * y ^ 2 -
    2 * Real.cos α * Real.sqrt k * Real.sqrt ℓ * x * y -
    k * ℓ * Real.sin α ^ 2

/-- Parametric definition of the union of the *filled* elliptical disks at
height `s`.  The common factor `ρ ∈ [0,1]` fills each boundary ellipse. -/
def FourSectionFilledEllipseUnion (α s x y : ℝ) : Prop :=
  0 ≤ s ∧ s ≤ 1 ∧
    ∃ k : ℝ,
      max 0 (2 * s - 1) ≤ k ∧ k ≤ s ^ 2 ∧
        ∃ ρ : ℝ, 0 ≤ ρ ∧ ρ ≤ 1 ∧
          ∃ χ : ℝ,
            x = ρ * Real.sqrt k * Real.cos χ ∧
              (y = ρ * Real.sqrt (1 - 2 * s + k) * Real.cos (χ + α) ∨
               y = -(ρ * Real.sqrt (1 - 2 * s + k) * Real.cos (χ + α)))

/-- Every boundary curve is contained in its corresponding filled disk. -/
theorem fourSectionCurveUnion_subset_filledEllipseUnion
    (α s x y : ℝ) (h : FourSectionCurveUnion α s x y) :
    FourSectionFilledEllipseUnion α s x y := by
  rcases h with ⟨hs0, hs1, k, hk0, hk1, χ, hx, hy⟩
  refine ⟨hs0, hs1, k, hk0, hk1, 1, by norm_num, by norm_num, χ, ?_, ?_⟩
  · simpa using hx
  · rcases hy with hy | hy
    · left
      simpa using hy
    · right
      simpa using hy

/-- The trigonometric ellipse parametrization satisfies the quadratic
boundary equation. -/
theorem fourEllipseResidual_parametric
    (α k ℓ χ : ℝ) (hk : 0 ≤ k) (hℓ : 0 ≤ ℓ) :
    fourEllipseResidual α k ℓ
      (Real.sqrt k * Real.cos χ)
      (Real.sqrt ℓ * Real.cos (χ + α)) = 0 := by
  have hcore :
      Real.cos χ ^ 2 +
          (Real.cos χ * Real.cos α - Real.sin χ * Real.sin α) ^ 2 -
        2 * Real.cos α * Real.cos χ *
          (Real.cos χ * Real.cos α - Real.sin χ * Real.sin α) -
        Real.sin α ^ 2 = 0 := by
    have hαsq : Real.cos α ^ 2 = 1 - Real.sin α ^ 2 := by
      nlinarith [Real.sin_sq_add_cos_sq α]
    have hχsq : Real.cos χ ^ 2 = 1 - Real.sin χ ^ 2 := by
      nlinarith [Real.sin_sq_add_cos_sq χ]
    ring_nf
    rw [hαsq, hχsq]
    ring
  have hscaled := congrArg (fun t : ℝ ↦ k * ℓ * t) hcore
  rw [fourEllipseResidual, Real.cos_add]
  ring_nf
  rw [Real.sq_sqrt hk, Real.sq_sqrt hℓ]
  ring_nf at hscaled
  ring_nf
  linear_combination hscaled

/-- Radially scaling a boundary point into its ellipse makes the residual
nonpositive; the deficit is exactly `(1-ρ²) k ℓ sin²α`. -/
theorem fourEllipseResidual_scaled_parametric
    (α k ℓ ρ χ : ℝ) (hk : 0 ≤ k) (hℓ : 0 ≤ ℓ) :
    fourEllipseResidual α k ℓ
      (ρ * Real.sqrt k * Real.cos χ)
      (ρ * Real.sqrt ℓ * Real.cos (χ + α)) =
      -((1 - ρ ^ 2) * k * ℓ * Real.sin α ^ 2) := by
  have hboundary := fourEllipseResidual_parametric α k ℓ χ hk hℓ
  rw [fourEllipseResidual] at hboundary ⊢
  linear_combination ρ ^ 2 * hboundary

/-- In particular, every point of a filled ellipse has nonpositive residual. -/
theorem fourEllipseResidual_scaled_nonpos
    (α k ℓ ρ χ : ℝ) (hk : 0 ≤ k) (hℓ : 0 ≤ ℓ)
    (hρ0 : 0 ≤ ρ) (hρ1 : ρ ≤ 1) :
    fourEllipseResidual α k ℓ
      (ρ * Real.sqrt k * Real.cos χ)
      (ρ * Real.sqrt ℓ * Real.cos (χ + α)) ≤ 0 := by
  rw [fourEllipseResidual_scaled_parametric α k ℓ ρ χ hk hℓ]
  have hρsq : ρ ^ 2 ≤ 1 := by nlinarith
  apply neg_nonpos.mpr
  exact mul_nonneg
    (mul_nonneg (mul_nonneg (sub_nonneg.mpr hρsq) hk) hℓ)
    (sq_nonneg (Real.sin α))

/-- At the lower endpoint of the allowed `k` interval, one ellipse radius
vanishes and the residual is nonnegative. -/
theorem fourEllipseResidual_lowerEndpoint_nonneg
    (α s x y : ℝ) (hs0 : 0 ≤ s) (hs1 : s ≤ 1) :
    0 ≤ fourEllipseResidual α (max 0 (2 * s - 1))
      (1 - 2 * s + max 0 (2 * s - 1)) x y := by
  by_cases hs : s ≤ 1 / 2
  · have hm : max 0 (2 * s - 1) = 0 := max_eq_left (by linarith)
    rw [hm, fourEllipseResidual]
    simp only [Real.sqrt_zero, mul_zero, zero_mul, zero_sub]
    have hx : 0 ≤ (1 - 2 * s) * x ^ 2 :=
      mul_nonneg (by linarith) (sq_nonneg x)
    linarith
  · have hs' : 1 / 2 ≤ s := le_of_not_ge hs
    have hm : max 0 (2 * s - 1) = 2 * s - 1 := max_eq_right (by linarith)
    rw [hm, show 1 - 2 * s + (2 * s - 1) = 0 by ring,
      fourEllipseResidual]
    simp only [Real.sqrt_zero, mul_zero, zero_mul, zero_sub]
    have hy : 0 ≤ (2 * s - 1) * y ^ 2 :=
      mul_nonneg (by linarith) (sq_nonneg y)
    linarith

/-- The residual varies continuously with `k`, including at the degenerate
ellipse endpoints because the real square root is continuous. -/
theorem continuous_fourEllipseResidual_section (α s x y : ℝ) :
    Continuous (fun k ↦
      fourEllipseResidual α k (1 - 2 * s + k) x y) := by
  unfold fourEllipseResidual
  fun_prop

/-- Intermediate-value step in the manuscript's disk-filling argument.  If
a point is inside an ellipse at an admissible `k`, some (possibly degenerate)
ellipse between the lower endpoint and that `k` has the point on its
quadratic boundary. -/
theorem fourEllipseResidual_exists_zero_between
    (α s x y k : ℝ) (hs0 : 0 ≤ s) (hs1 : s ≤ 1)
    (hk0 : max 0 (2 * s - 1) ≤ k)
    (hinside : fourEllipseResidual α k (1 - 2 * s + k) x y ≤ 0) :
    ∃ k' : ℝ,
      max 0 (2 * s - 1) ≤ k' ∧ k' ≤ k ∧
      fourEllipseResidual α k' (1 - 2 * s + k') x y = 0 := by
  let m := max 0 (2 * s - 1)
  let f := fun t : ℝ ↦ fourEllipseResidual α t (1 - 2 * s + t) x y
  have hm : 0 ≤ f m :=
    fourEllipseResidual_lowerEndpoint_nonneg α s x y hs0 hs1
  have hk : f k ≤ 0 := hinside
  have hcont : ContinuousOn f (Set.Icc m k) :=
    (continuous_fourEllipseResidual_section α s x y).continuousOn
  have hzero : (0 : ℝ) ∈ Set.Icc (f k) (f m) := ⟨hk, hm⟩
  rcases intermediate_value_Icc' hk0 hcont hzero with ⟨k', hk', hvalue⟩
  exact ⟨k', hk'.1, hk'.2, hvalue⟩

/-- For positive radii and a nondegenerate family angle, the quadratic
boundary equation is sufficient for an angular parametrization. -/
theorem fourEllipseResidual_exists_angle
    (α k ℓ x y : ℝ) (hk : 0 < k) (hℓ : 0 < ℓ)
    (hα : Real.sin α ≠ 0)
    (hres : fourEllipseResidual α k ℓ x y = 0) :
    ∃ χ : ℝ,
      x = Real.sqrt k * Real.cos χ ∧
      y = Real.sqrt ℓ * Real.cos (χ + α) := by
  let rk := Real.sqrt k
  let rℓ := Real.sqrt ℓ
  let u := x / rk
  let v := (u * Real.cos α - y / rℓ) / Real.sin α
  have hrk0 : 0 < rk := Real.sqrt_pos.2 hk
  have hrℓ0 : 0 < rℓ := Real.sqrt_pos.2 hℓ
  have hrk2 : rk ^ 2 = k := Real.sq_sqrt hk.le
  have hrℓ2 : rℓ ^ 2 = ℓ := Real.sq_sqrt hℓ.le
  have huv : u ^ 2 + v ^ 2 = 1 := by
    dsimp [u, v]
    change ℓ * x ^ 2 + k * y ^ 2 -
      2 * Real.cos α * rk * rℓ * x * y -
      k * ℓ * Real.sin α ^ 2 = 0 at hres
    rw [← hrk2, ← hrℓ2] at hres
    field_simp [ne_of_gt hrk0, ne_of_gt hrℓ0, hα]
    have hαsq : Real.cos α ^ 2 = 1 - Real.sin α ^ 2 := by
      nlinarith [Real.sin_sq_add_cos_sq α]
    ring_nf
    rw [hαsq]
    have hscaled := congrArg (fun t : ℝ ↦ rk ^ 2 * t) hres
    ring_nf at hscaled
    nlinarith
  let z : ℂ := ⟨u, v⟩
  have hzNorm : ‖z‖ = 1 := by
    rw [Complex.norm_def, Complex.normSq_apply]
    dsimp [z]
    rw [show u * u + v * v = u ^ 2 + v ^ 2 by ring]
    rw [huv, Real.sqrt_one]
  let χ := Complex.arg z
  have hcos : Real.cos χ = u := by
    have h := Complex.norm_mul_cos_arg z
    rw [hzNorm, one_mul] at h
    exact h
  have hsin : Real.sin χ = v := by
    have h := Complex.norm_mul_sin_arg z
    rw [hzNorm, one_mul] at h
    exact h
  refine ⟨χ, ?_, ?_⟩
  · rw [hcos]
    change x = rk * (x / rk)
    field_simp [ne_of_gt hrk0]
  · rw [Real.cos_add, hcos, hsin]
    dsimp [v, u]
    field_simp [ne_of_gt hrk0, ne_of_gt hrℓ0, hα]
    ring

/-- Every point of the interval with squared radius `r` has a cosine
parametrization.  This elementary lemma is used for the degenerate axis
sections of the ellipse family. -/
theorem exists_eq_sqrt_mul_cos_of_sq_le
    (r z : ℝ) (hr : 0 < r) (hz : z ^ 2 ≤ r) :
    ∃ θ : ℝ, z = Real.sqrt r * Real.cos θ := by
  let u := z / Real.sqrt r
  have hsr : 0 < Real.sqrt r := Real.sqrt_pos.2 hr
  have hu0 : -(1 : ℝ) ≤ u := by
    dsimp [u]
    rw [le_div_iff₀ hsr]
    nlinarith [Real.sq_sqrt hr.le]
  have hu1 : u ≤ 1 := by
    dsimp [u]
    rw [div_le_iff₀ hsr]
    nlinarith [Real.sq_sqrt hr.le]
  refine ⟨Real.arccos u, ?_⟩
  rw [Real.cos_arccos hu0 hu1]
  dsimp [u]
  field_simp [ne_of_gt hsr]

/-- A nondegenerate zero of the quadratic residual is a member of the exact
section-curve union. -/
theorem fourEllipseResidual_zero_mem_curveUnion_of_pos
    (α s x y k : ℝ) (hs0 : 0 ≤ s) (hs1 : s ≤ 1)
    (hk0 : max 0 (2 * s - 1) ≤ k) (hk1 : k ≤ s ^ 2)
    (hk : 0 < k) (hℓ : 0 < 1 - 2 * s + k)
    (hα : Real.sin α ≠ 0)
    (hres : fourEllipseResidual α k (1 - 2 * s + k) x y = 0) :
    FourSectionCurveUnion α s x y := by
  rcases fourEllipseResidual_exists_angle α k (1 - 2 * s + k) x y
      hk hℓ hα hres with ⟨χ, hx, hy⟩
  exact ⟨hs0, hs1, k, hk0, hk1, χ, hx, Or.inl hy⟩

/-- On the lower endpoint below the central section, the degenerate curve is
the complete vertical segment. -/
theorem fourSectionCurveUnion_vertical_endpoint
    (α s y : ℝ) (hs0 : 0 ≤ s) (hs : s < 1 / 2)
    (hy : y ^ 2 ≤ 1 - 2 * s) :
    FourSectionCurveUnion α s 0 y := by
  have hℓ : 0 < 1 - 2 * s := by linarith
  rcases exists_eq_sqrt_mul_cos_of_sq_le (1 - 2 * s) y hℓ hy with
    ⟨θ, hθ⟩
  refine ⟨hs0, by linarith, 0, ?_, by nlinarith, θ - α, ?_, ?_⟩
  · rw [max_eq_left]
    linarith
  · simp
  · left
    rw [show θ - α + α = θ by ring]
    simpa using hθ

/-- On the lower endpoint above the central section, the degenerate curve is
the complete horizontal segment. -/
theorem fourSectionCurveUnion_horizontal_endpoint
    (α s x : ℝ) (hs : 1 / 2 < s) (hs1 : s ≤ 1)
    (hx : x ^ 2 ≤ 2 * s - 1) :
    FourSectionCurveUnion α s x 0 := by
  have hk : 0 < 2 * s - 1 := by linarith
  rcases exists_eq_sqrt_mul_cos_of_sq_le (2 * s - 1) x hk hx with
    ⟨χ, hχ⟩
  have hm : max 0 (2 * s - 1) = 2 * s - 1 := max_eq_right hk.le
  refine ⟨by linarith, hs1, 2 * s - 1, hm.le, ?_, χ, hχ, ?_⟩
  · nlinarith [sq_nonneg (s - 1)]
  · left
    rw [show 1 - 2 * s + (2 * s - 1) = 0 by ring, Real.sqrt_zero,
      zero_mul]

/-- A radially scaled cosine coordinate stays within its squared radius. -/
theorem sq_le_radius_of_eq_scaled_sqrt_cos
    (r ρ θ z : ℝ) (hr : 0 ≤ r) (hρ0 : 0 ≤ ρ) (hρ1 : ρ ≤ 1)
    (hz : z = ρ * Real.sqrt r * Real.cos θ) : z ^ 2 ≤ r := by
  have hρsq : ρ ^ 2 ≤ 1 := by nlinarith
  have hcos : Real.cos θ ^ 2 ≤ 1 := by
    nlinarith [Real.sin_sq_add_cos_sq θ]
  have hprod : ρ ^ 2 * Real.cos θ ^ 2 ≤ 1 := calc
    ρ ^ 2 * Real.cos θ ^ 2 ≤ ρ ^ 2 * 1 :=
      mul_le_mul_of_nonneg_left hcos (sq_nonneg ρ)
    _ ≤ 1 := by simpa using hρsq
  rw [hz]
  have hsq : (ρ * Real.sqrt r * Real.cos θ) ^ 2 =
      r * (ρ ^ 2 * Real.cos θ ^ 2) := by
    calc
      (ρ * Real.sqrt r * Real.cos θ) ^ 2 =
          Real.sqrt r ^ 2 * (ρ ^ 2 * Real.cos θ ^ 2) := by ring
      _ = r * (ρ ^ 2 * Real.cos θ ^ 2) := by rw [Real.sq_sqrt hr]
  rw [hsq]
  nlinarith [mul_nonneg hr (sub_nonneg.mpr hprod)]

/-- Generic intermediate-value lemma with an arbitrary left endpoint. -/
theorem fourEllipseResidual_exists_zero_on_interval
    (α s x y a b : ℝ) (hab : a ≤ b)
    (ha : 0 ≤ fourEllipseResidual α a (1 - 2 * s + a) x y)
    (hb : fourEllipseResidual α b (1 - 2 * s + b) x y ≤ 0) :
    ∃ k : ℝ, a ≤ k ∧ k ≤ b ∧
      fourEllipseResidual α k (1 - 2 * s + k) x y = 0 := by
  let f := fun t : ℝ ↦ fourEllipseResidual α t (1 - 2 * s + t) x y
  have hcont : ContinuousOn f (Set.Icc a b) :=
    (continuous_fourEllipseResidual_section α s x y).continuousOn
  have hzero : (0 : ℝ) ∈ Set.Icc (f b) (f a) := ⟨hb, ha⟩
  rcases intermediate_value_Icc' hab hcont hzero with ⟨k, hk, hvalue⟩
  exact ⟨k, hk.1, hk.2, hvalue⟩

/-- Below the central section, a filled `+` ellipse point is on some exact
section curve.  The proof treats the vertical endpoint explicitly and uses a
strictly positive comparison parameter otherwise. -/
theorem fourSection_filledPlus_mem_curveUnion_below
    (α s x y k ρ χ : ℝ) (hs0 : 0 ≤ s) (hs : s < 1 / 2)
    (hk0 : max 0 (2 * s - 1) ≤ k) (hk1 : k ≤ s ^ 2)
    (hρ0 : 0 ≤ ρ) (hρ1 : ρ ≤ 1)
    (hx : x = ρ * Real.sqrt k * Real.cos χ)
    (hy : y = ρ * Real.sqrt (1 - 2 * s + k) * Real.cos (χ + α))
    (hα : Real.sin α ≠ 0) :
    FourSectionCurveUnion α s x y := by
  have hs1 : s ≤ 1 := by linarith
  have hk : 0 ≤ k := le_trans (le_max_left 0 (2 * s - 1)) hk0
  have hℓ : 0 ≤ 1 - 2 * s + k := by linarith
  have hxbound := sq_le_radius_of_eq_scaled_sqrt_cos k ρ χ x
    hk hρ0 hρ1 hx
  have hybound := sq_le_radius_of_eq_scaled_sqrt_cos
    (1 - 2 * s + k) ρ (χ + α) y hℓ hρ0 hρ1 hy
  have hinside := fourEllipseResidual_scaled_nonpos α k
    (1 - 2 * s + k) ρ χ hk hℓ hρ0 hρ1
  rw [← hx, ← hy] at hinside
  by_cases hx0 : x = 0
  · by_cases hyend : y ^ 2 ≤ 1 - 2 * s
    · rw [hx0]
      exact fourSectionCurveUnion_vertical_endpoint α s y hs0 hs hyend
    · have hylarge : 1 - 2 * s < y ^ 2 := lt_of_not_ge hyend
      let e := (y ^ 2 - (1 - 2 * s)) / 2
      have he : 0 < e := by dsimp [e]; linarith
      have hek : e ≤ k := by dsimp [e]; linarith
      have heℓ : 0 ≤ 1 - 2 * s + e := by dsimp [e]; linarith
      have hesin : Real.sin α ^ 2 ≤ 1 := by
        nlinarith [Real.sin_sq_add_cos_sq α]
      have hepos : 0 < fourEllipseResidual α e (1 - 2 * s + e) x y := by
        rw [hx0, fourEllipseResidual]
        simp only [zero_pow (by norm_num : (2 : Nat) ≠ 0), mul_zero, zero_mul]
        have hcoef : 0 < y ^ 2 -
            (1 - 2 * s + e) * Real.sin α ^ 2 := by
          have haux := mul_nonneg heℓ (sub_nonneg.mpr hesin)
          dsimp [e] at haux ⊢
          nlinarith
        nlinarith [mul_pos he hcoef]
      rcases fourEllipseResidual_exists_zero_on_interval α s x y e k hek
          hepos.le hinside with ⟨k', hk'e, hk'k, hk'res⟩
      have hk' : 0 < k' := lt_of_lt_of_le he hk'e
      have hℓ' : 0 < 1 - 2 * s + k' := by linarith
      have hk'0 : max 0 (2 * s - 1) ≤ k' := by
        rw [max_eq_left (by linarith)]
        exact hk'.le
      exact fourEllipseResidual_zero_mem_curveUnion_of_pos α s x y k'
        hs0 hs1 hk'0 (le_trans hk'k hk1) hk' hℓ' hα hk'res
  · rcases fourEllipseResidual_exists_zero_between α s x y k hs0 hs1 hk0
        hinside with ⟨k', hk'0, hk'k, hk'res⟩
    have hk' : 0 < k' := by
      rw [max_eq_left (by linarith)] at hk'0
      rcases lt_or_eq_of_le hk'0 with hk' | hk'
      · exact hk'
      · subst k'
        rw [fourEllipseResidual] at hk'res
        simp only [Real.sqrt_zero, mul_zero, zero_mul, zero_sub] at hk'res
        nlinarith [sq_pos_of_ne_zero hx0]
    have hℓ' : 0 < 1 - 2 * s + k' := by linarith
    exact fourEllipseResidual_zero_mem_curveUnion_of_pos α s x y k'
      hs0 hs1 hk'0 (le_trans hk'k hk1) hk' hℓ' hα hk'res

/-- Above the central section, the analogous endpoint is horizontal. -/
theorem fourSection_filledPlus_mem_curveUnion_above
    (α s x y k ρ χ : ℝ) (hs : 1 / 2 < s) (hs1 : s ≤ 1)
    (hk0 : max 0 (2 * s - 1) ≤ k) (hk1 : k ≤ s ^ 2)
    (hρ0 : 0 ≤ ρ) (hρ1 : ρ ≤ 1)
    (hx : x = ρ * Real.sqrt k * Real.cos χ)
    (hy : y = ρ * Real.sqrt (1 - 2 * s + k) * Real.cos (χ + α))
    (hα : Real.sin α ≠ 0) :
    FourSectionCurveUnion α s x y := by
  have hs0 : 0 ≤ s := by linarith
  have hm : max 0 (2 * s - 1) = 2 * s - 1 :=
    max_eq_right (by linarith)
  have hk : 0 ≤ k := le_trans (le_max_left 0 (2 * s - 1)) hk0
  have hℓ : 0 ≤ 1 - 2 * s + k := by rw [hm] at hk0; linarith
  have hxbound := sq_le_radius_of_eq_scaled_sqrt_cos k ρ χ x
    hk hρ0 hρ1 hx
  have hinside := fourEllipseResidual_scaled_nonpos α k
    (1 - 2 * s + k) ρ χ hk hℓ hρ0 hρ1
  rw [← hx, ← hy] at hinside
  by_cases hy0 : y = 0
  · by_cases hxend : x ^ 2 ≤ 2 * s - 1
    · rw [hy0]
      exact fourSectionCurveUnion_horizontal_endpoint α s x hs hs1 hxend
    · have hxlarge : 2 * s - 1 < x ^ 2 := lt_of_not_ge hxend
      let e := (x ^ 2 - (2 * s - 1)) / 2
      let a := 2 * s - 1 + e
      have he : 0 < e := by dsimp [e]; linarith
      have hak : a ≤ k := by dsimp [a, e]; linarith
      have haℓ : 1 - 2 * s + a = e := by dsimp [a]; ring
      have hasin : Real.sin α ^ 2 ≤ 1 := by
        nlinarith [Real.sin_sq_add_cos_sq α]
      have hapos : 0 < fourEllipseResidual α a (1 - 2 * s + a) x y := by
        rw [hy0, fourEllipseResidual, haℓ]
        simp only [zero_pow (by norm_num : (2 : Nat) ≠ 0), mul_zero]
        have ha0 : 0 ≤ a := by dsimp [a]; linarith
        have hcoef : 0 < x ^ 2 - a * Real.sin α ^ 2 := by
          have haux := mul_nonneg ha0 (sub_nonneg.mpr hasin)
          dsimp [a, e] at haux ⊢
          nlinarith
        nlinarith [mul_pos he hcoef]
      rcases fourEllipseResidual_exists_zero_on_interval α s x y a k hak
          hapos.le hinside with ⟨k', hk'a, hk'k, hk'res⟩
      have hk' : 0 < k' := by dsimp [a] at hk'a; linarith
      have hℓ' : 0 < 1 - 2 * s + k' := by
        dsimp [a] at hk'a
        linarith
      have hk'0 : max 0 (2 * s - 1) ≤ k' := by rw [hm]; linarith
      exact fourEllipseResidual_zero_mem_curveUnion_of_pos α s x y k'
        hs0 hs1 hk'0 (le_trans hk'k hk1) hk' hℓ' hα hk'res
  · rcases fourEllipseResidual_exists_zero_between α s x y k hs0 hs1 hk0
        hinside with ⟨k', hk'0, hk'k, hk'res⟩
    have hℓ' : 0 < 1 - 2 * s + k' := by
      rw [hm] at hk'0
      rcases lt_or_eq_of_le hk'0 with hk' | hk'
      · linarith
      · subst k'
        rw [fourEllipseResidual] at hk'res
        rw [show 1 - 2 * s + (2 * s - 1) = 0 by ring] at hk'res
        simp only [Real.sqrt_zero, mul_zero, zero_mul, zero_sub] at hk'res
        nlinarith [sq_pos_of_ne_zero hy0]
    have hk' : 0 < k' := by rw [hm] at hk'0; linarith
    exact fourEllipseResidual_zero_mem_curveUnion_of_pos α s x y k'
      hs0 hs1 hk'0 (le_trans hk'k hk1) hk' hℓ' hα hk'res

/-- At the central section both endpoint radii vanish.  Away from the origin,
the second boundary root is explicit. -/
theorem fourSection_filledPlus_mem_curveUnion_central
    (α x y k ρ χ : ℝ)
    (hk0 : max 0 (2 * (1 / 2 : ℝ) - 1) ≤ k)
    (hk1 : k ≤ (1 / 2 : ℝ) ^ 2)
    (hρ0 : 0 ≤ ρ) (hρ1 : ρ ≤ 1)
    (hx : x = ρ * Real.sqrt k * Real.cos χ)
    (hy : y = ρ * Real.sqrt (1 - 2 * (1 / 2 : ℝ) + k) *
      Real.cos (χ + α))
    (hα : Real.sin α ≠ 0) :
    FourSectionCurveUnion α (1 / 2) x y := by
  have hk : 0 ≤ k := by norm_num at hk0; exact hk0
  have hℓ : 0 ≤ 1 - 2 * (1 / 2 : ℝ) + k := by norm_num; exact hk
  by_cases hxy : x = 0 ∧ y = 0
  · rcases hxy with ⟨rfl, rfl⟩
    refine ⟨by norm_num, by norm_num, 0, by norm_num, by norm_num,
      0, ?_, Or.inl ?_⟩ <;> norm_num
  · have hkpos : 0 < k := by
      rcases lt_or_eq_of_le hk with hkpos | rfl
      · exact hkpos
      · norm_num at hx hy
        exact False.elim (hxy ⟨hx, hy⟩)
    let q := x ^ 2 + y ^ 2 - 2 * Real.cos α * x * y
    have hsinpos : 0 < Real.sin α ^ 2 := sq_pos_of_ne_zero hα
    have hqdecomp : q = (x - Real.cos α * y) ^ 2 +
        Real.sin α ^ 2 * y ^ 2 := by
      dsimp [q]
      nlinarith [Real.sin_sq_add_cos_sq α]
    have hq0 : 0 ≤ q := by
      rw [hqdecomp]
      positivity
    have hqpos : 0 < q := by
      rcases lt_or_eq_of_le hq0 with hqpos | hqzero
      · exact hqpos
      · have hy0 : y = 0 := by
          rw [hqdecomp] at hqzero
          have : Real.sin α ^ 2 * y ^ 2 = 0 := by
            nlinarith [sq_nonneg (x - Real.cos α * y),
              mul_nonneg hsinpos.le (sq_nonneg y)]
          exact (mul_eq_zero.mp this).resolve_left (ne_of_gt hsinpos) |>
            sq_eq_zero_iff.mp
        have hx0 : x = 0 := by
          dsimp [q] at hqzero
          rw [hy0] at hqzero
          nlinarith
        exact False.elim (hxy ⟨hx0, hy0⟩)
    have hinside := fourEllipseResidual_scaled_nonpos α k k ρ χ
      hk hk hρ0 hρ1
    have hy' : y = ρ * Real.sqrt k * Real.cos (χ + α) := by
      simpa using hy
    rw [← hx, ← hy'] at hinside
    have hqk : q ≤ k * Real.sin α ^ 2 := by
      rw [fourEllipseResidual] at hinside
      have hform : k * x ^ 2 + k * y ^ 2 -
          2 * Real.cos α * Real.sqrt k * Real.sqrt k * x * y -
          k * k * Real.sin α ^ 2 =
          k * (q - k * Real.sin α ^ 2) := by
        calc
          k * x ^ 2 + k * y ^ 2 -
              2 * Real.cos α * Real.sqrt k * Real.sqrt k * x * y -
              k * k * Real.sin α ^ 2 =
              k * x ^ 2 + k * y ^ 2 -
                2 * Real.cos α * (Real.sqrt k * Real.sqrt k) * x * y -
                k * k * Real.sin α ^ 2 := by ring
          _ = k * x ^ 2 + k * y ^ 2 -
                2 * Real.cos α * k * x * y -
                k * k * Real.sin α ^ 2 := by rw [Real.mul_self_sqrt hk]
          _ = k * (q - k * Real.sin α ^ 2) := by dsimp [q]; ring
      rw [hform] at hinside
      have hmul : k * (q - k * Real.sin α ^ 2) ≤ k * 0 := by
        simpa using hinside
      have := (mul_le_mul_left hkpos).mp hmul
      linarith
    let k' := q / Real.sin α ^ 2
    have hk' : 0 < k' := div_pos hqpos hsinpos
    have hk'k : k' ≤ k := by
      dsimp [k']
      exact (div_le_iff₀ hsinpos).2 hqk
    have hk'1 : k' ≤ (1 / 2 : ℝ) ^ 2 := le_trans hk'k hk1
    have hk'res : fourEllipseResidual α k' k' x y = 0 := by
      rw [fourEllipseResidual]
      have hk'eq : k' * Real.sin α ^ 2 = q := by
        dsimp [k']
        field_simp [ne_of_gt hsinpos]
      dsimp [q] at hk'eq
      calc
        k' * x ^ 2 + k' * y ^ 2 -
            2 * Real.cos α * Real.sqrt k' * Real.sqrt k' * x * y -
            k' * k' * Real.sin α ^ 2 =
            k' * x ^ 2 + k' * y ^ 2 -
              2 * Real.cos α * (Real.sqrt k' * Real.sqrt k') * x * y -
              k' * k' * Real.sin α ^ 2 := by ring
        _ = k' * x ^ 2 + k' * y ^ 2 -
              2 * Real.cos α * k' * x * y -
              k' * k' * Real.sin α ^ 2 := by
                rw [Real.mul_self_sqrt hk'.le]
        _ = 0 := by nlinarith
    have hk'0 : max 0 (2 * (1 / 2 : ℝ) - 1) ≤ k' := by
      norm_num
      exact hk'.le
    exact fourEllipseResidual_zero_mem_curveUnion_of_pos α (1 / 2) x y k'
      (by norm_num) (by norm_num) hk'0 hk'1 hk' (by simpa using hk') hα
      (by simpa using hk'res)

/-- The two explicit sign branches of the curve union are interchanged by
reflection across the horizontal axis. -/
theorem fourSectionCurveUnion_neg_y_iff (α s x y : ℝ) :
    FourSectionCurveUnion α s x (-y) ↔ FourSectionCurveUnion α s x y := by
  constructor
  · rintro ⟨hs0, hs1, k, hk0, hk1, χ, hx, hy⟩
    refine ⟨hs0, hs1, k, hk0, hk1, χ, hx, ?_⟩
    rcases hy with hy | hy
    · right
      linarith
    · left
      linarith
  · rintro ⟨hs0, hs1, k, hk0, hk1, χ, hx, hy⟩
    refine ⟨hs0, hs1, k, hk0, hk1, χ, hx, ?_⟩
    rcases hy with hy | hy
    · right
      linarith
    · left
      linarith

/-- A radially filled point on the positive-sign ellipse branch belongs to
the exact section-curve union.  The proof is complete across all three height
regimes, including the degenerate endpoint sections. -/
theorem fourSection_filledPlus_mem_curveUnion
    (α s x y k ρ χ : ℝ) (hs0 : 0 ≤ s) (hs1 : s ≤ 1)
    (hk0 : max 0 (2 * s - 1) ≤ k) (hk1 : k ≤ s ^ 2)
    (hρ0 : 0 ≤ ρ) (hρ1 : ρ ≤ 1)
    (hx : x = ρ * Real.sqrt k * Real.cos χ)
    (hy : y = ρ * Real.sqrt (1 - 2 * s + k) * Real.cos (χ + α))
    (hα : Real.sin α ≠ 0) :
    FourSectionCurveUnion α s x y := by
  rcases lt_trichotomy s (1 / 2) with hs | hs | hs
  · exact fourSection_filledPlus_mem_curveUnion_below α s x y k ρ χ
      hs0 hs hk0 hk1 hρ0 hρ1 hx hy hα
  · subst s
    exact fourSection_filledPlus_mem_curveUnion_central α x y k ρ χ
      hk0 hk1 hρ0 hρ1 hx hy hα
  · exact fourSection_filledPlus_mem_curveUnion_above α s x y k ρ χ
      hs hs1 hk0 hk1 hρ0 hρ1 hx hy hα

/-- For every nondegenerate member of the continuous four-port family, the
union of exact phase-section curves equals the union of their filled ellipse
disks.  This is the converse needed to turn the boundary parametrization into
the complete reachable region. -/
theorem fourSection_curveUnion_iff_filledEllipseUnion_of_sin_ne_zero
    (α s x y : ℝ) (hα : Real.sin α ≠ 0) :
    FourSectionCurveUnion α s x y ↔
      FourSectionFilledEllipseUnion α s x y := by
  constructor
  · exact fourSectionCurveUnion_subset_filledEllipseUnion α s x y
  · rintro ⟨hs0, hs1, k, hk0, hk1, ρ, hρ0, hρ1, χ, hx, hy⟩
    rcases hy with hy | hy
    · exact fourSection_filledPlus_mem_curveUnion α s x y k ρ χ
        hs0 hs1 hk0 hk1 hρ0 hρ1 hx hy hα
    · apply (fourSectionCurveUnion_neg_y_iff α s x y).mp
      apply fourSection_filledPlus_mem_curveUnion α s x (-y) k ρ χ
        hs0 hs1 hk0 hk1 hρ0 hρ1 hx
      · linarith
      · exact hα

/-- At a degenerate family angle, radial scaling is absorbed directly into
the common cosine parameter, so the filled parametrization introduces no new
points. -/
theorem fourSection_filledEllipseUnion_mem_curveUnion_of_sin_eq_zero
    (α s x y : ℝ) (hα : Real.sin α = 0)
    (h : FourSectionFilledEllipseUnion α s x y) :
    FourSectionCurveUnion α s x y := by
  rcases h with ⟨hs0, hs1, k, hk0, hk1, ρ, hρ0, hρ1, χ, hx, hy⟩
  have hρsq : ρ ^ 2 ≤ 1 := by nlinarith
  have hcos : Real.cos χ ^ 2 ≤ 1 := by
    nlinarith [Real.sin_sq_add_cos_sq χ]
  have hprod : (ρ * Real.cos χ) ^ 2 ≤ 1 := by
    have hmul : ρ ^ 2 * Real.cos χ ^ 2 ≤ 1 := calc
      ρ ^ 2 * Real.cos χ ^ 2 ≤ ρ ^ 2 * 1 :=
        mul_le_mul_of_nonneg_left hcos (sq_nonneg ρ)
      _ ≤ 1 := by simpa using hρsq
    nlinarith
  rcases exists_eq_sqrt_mul_cos_of_sq_le 1 (ρ * Real.cos χ)
      (by norm_num) (by simpa using hprod) with ⟨θ, hθ⟩
  have hθ' : ρ * Real.cos χ = Real.cos θ := by simpa using hθ
  have hχadd : Real.cos (χ + α) = Real.cos χ * Real.cos α := by
    rw [Real.cos_add, hα]
    ring
  have hθadd : Real.cos (θ + α) = Real.cos θ * Real.cos α := by
    rw [Real.cos_add, hα]
    ring
  refine ⟨hs0, hs1, k, hk0, hk1, θ, ?_, ?_⟩
  · calc
      x = ρ * Real.sqrt k * Real.cos χ := hx
      _ = Real.sqrt k * (ρ * Real.cos χ) := by ring
      _ = Real.sqrt k * Real.cos θ := by rw [hθ']
  · rcases hy with hy | hy
    · left
      calc
        y = ρ * Real.sqrt (1 - 2 * s + k) * Real.cos (χ + α) := hy
        _ = Real.sqrt (1 - 2 * s + k) *
            ((ρ * Real.cos χ) * Real.cos α) := by rw [hχadd]; ring
        _ = Real.sqrt (1 - 2 * s + k) * Real.cos (θ + α) := by
          rw [hθ', hθadd]
    · right
      calc
        y = -(ρ * Real.sqrt (1 - 2 * s + k) * Real.cos (χ + α)) := hy
        _ = -(Real.sqrt (1 - 2 * s + k) *
            ((ρ * Real.cos χ) * Real.cos α)) := by rw [hχadd]; ring
        _ = -(Real.sqrt (1 - 2 * s + k) * Real.cos (θ + α)) := by
          rw [hθ', hθadd]
        _ = -Real.sqrt (1 - 2 * s + k) * Real.cos (θ + α) := by ring

/-- The exact section curves equal the union of their radially filled disks
for every family angle.  Nondegenerate angles use the quadratic-boundary and
intermediate-value argument; degenerate angles reduce to line segments. -/
theorem fourSection_curveUnion_iff_filledEllipseUnion (α s x y : ℝ) :
    FourSectionCurveUnion α s x y ↔
      FourSectionFilledEllipseUnion α s x y := by
  by_cases hα : Real.sin α = 0
  · constructor
    · exact fourSectionCurveUnion_subset_filledEllipseUnion α s x y
    · exact fourSection_filledEllipseUnion_mem_curveUnion_of_sin_eq_zero
        α s x y hα
  · exact fourSection_curveUnion_iff_filledEllipseUnion_of_sin_ne_zero
      α s x y hα

/-- Final physical statement: at every family angle, a point at height `s`
is produced by physical phases exactly when it lies in the stated union of
filled ellipse disks (degenerating to line segments when `sin α = 0`). -/
theorem fourSection_reachable_iff_filledEllipseUnion
    (α s x y : ℝ) :
    (∃ φ ψ χ : ℝ,
      fourSectionS φ ψ = s ∧
      fourSectionX φ ψ χ = x ∧
      fourSectionY α φ ψ χ = y) ↔
      FourSectionFilledEllipseUnion α s x y := by
  rw [fourSection_reachable_iff_curveUnion]
  exact fourSection_curveUnion_iff_filledEllipseUnion α s x y

/-- Squaring the section coordinates gives the curve radii `k` and `ℓ`. -/
theorem fourSectionXY_sq (α φ ψ χ : ℝ) :
    fourSectionX φ ψ χ ^ 2 =
        fourSectionK φ ψ * Real.cos χ ^ 2 ∧
      fourSectionY α φ ψ χ ^ 2 =
        fourSectionEll φ ψ * Real.cos (χ + α) ^ 2 := by
  constructor <;>
    simp only [fourSectionX, fourSectionY, fourSectionK, fourSectionEll] <;>
    ring

/-- At the DFT endpoint `α = π/2`, every nondegenerate fixed-`k` boundary
curve obeys the stated ellipse equation. -/
theorem fourSection_dft_ellipse (φ ψ χ : ℝ)
    (hk : fourSectionK φ ψ ≠ 0)
    (hℓ : fourSectionEll φ ψ ≠ 0) :
    fourSectionX φ ψ χ ^ 2 / fourSectionK φ ψ +
      fourSectionY (Real.pi / 2) φ ψ χ ^ 2 /
        fourSectionEll φ ψ = 1 := by
  rw [(fourSectionXY_sq (Real.pi / 2) φ ψ χ).1,
    (fourSectionXY_sq (Real.pi / 2) φ ψ χ).2]
  field_simp [hk, hℓ]
  rw [show (χ * 2 + Real.pi) / 2 = χ + Real.pi / 2 by ring,
    Real.cos_add_pi_div_two]
  nlinarith [Real.sin_sq_add_cos_sq χ]

/-- Every physically reachable DFT point lies in the filled ellipse stated
in Supplementary Note 7.  The fixed-`k` curves above are nested inside the
largest curve, obtained at `k = s²`. -/
theorem fourSection_dft_full_region_bound (φ ψ χ : ℝ)
    (hs0 : 0 < fourSectionS φ ψ)
    (hs1 : fourSectionS φ ψ < 1) :
    fourSectionX φ ψ χ ^ 2 / fourSectionS φ ψ ^ 2 +
      fourSectionY (Real.pi / 2) φ ψ χ ^ 2 /
        (1 - fourSectionS φ ψ) ^ 2 ≤ 1 := by
  let s := fourSectionS φ ψ
  let k := fourSectionK φ ψ
  let ℓ := fourSectionEll φ ψ
  have hs_sq : 0 < s ^ 2 := sq_pos_of_pos hs0
  have hcs_sq : 0 < (1 - s) ^ 2 := sq_pos_of_pos (sub_pos.mpr hs1)
  have hk : k ≤ s ^ 2 := (fourSectionK_bounds φ ψ).2
  have hℓeq : ℓ = 1 - 2 * s + k := fourSectionEll_eq φ ψ
  have hℓ : ℓ ≤ (1 - s) ^ 2 := by
    rw [hℓeq]
    nlinarith
  rw [(fourSectionXY_sq (Real.pi / 2) φ ψ χ).1,
    (fourSectionXY_sq (Real.pi / 2) φ ψ χ).2]
  rw [Real.cos_add_pi_div_two]
  have hx : k * Real.cos χ ^ 2 / s ^ 2 ≤ Real.cos χ ^ 2 := by
    apply (div_le_iff₀ hs_sq).2
    nlinarith [sq_nonneg (Real.cos χ)]
  have hy : ℓ * (-Real.sin χ) ^ 2 / (1 - s) ^ 2 ≤
      Real.sin χ ^ 2 := by
    apply (div_le_iff₀ hcs_sq).2
    nlinarith [sq_nonneg (Real.sin χ)]
  change k * Real.cos χ ^ 2 / s ^ 2 +
      ℓ * (-Real.sin χ) ^ 2 / (1 - s) ^ 2 ≤ 1
  nlinarith [Real.sin_sq_add_cos_sq χ]

/-- At the Walsh endpoint, eliminating the common `cos χ` parameter gives
the pair of straight section curves. -/
theorem fourSection_walsh_lines (φ ψ χ : ℝ) :
    fourSectionEll φ ψ * fourSectionX φ ψ χ ^ 2 =
      fourSectionK φ ψ * fourSectionY 0 φ ψ χ ^ 2 := by
  rw [(fourSectionXY_sq 0 φ ψ χ).1,
    (fourSectionXY_sq 0 φ ψ χ).2]
  simp only [add_zero]
  ring

/-- Every reachable Walsh point below the central section obeys the two
polynomial inequalities equivalent to the manuscript's exact lower-section
description.  This direction is the nontrivial necessity calculation. -/
theorem fourSection_walsh_lower_bounds (φ ψ χ : ℝ)
    (hs : fourSectionS φ ψ < 1 / 2) :
    (1 - fourSectionS φ ψ) ^ 2 * fourSectionX φ ψ χ ^ 2 ≤
        fourSectionS φ ψ ^ 2 * fourSectionY 0 φ ψ χ ^ 2 ∧
      fourSectionY 0 φ ψ χ ^ 2 - fourSectionX φ ψ χ ^ 2 ≤
        1 - 2 * fourSectionS φ ψ := by
  let s := fourSectionS φ ψ
  let k := fourSectionK φ ψ
  let ℓ := fourSectionEll φ ψ
  have hk : k ≤ s ^ 2 := (fourSectionK_bounds φ ψ).2
  have hℓ : ℓ = 1 - 2 * s + k := fourSectionEll_eq φ ψ
  have hc0 : 0 ≤ Real.cos χ ^ 2 := sq_nonneg _
  have hc1 : Real.cos χ ^ 2 ≤ 1 := by
    nlinarith [Real.sin_sq_add_cos_sq χ, sq_nonneg (Real.sin χ)]
  rw [(fourSectionXY_sq 0 φ ψ χ).1,
    (fourSectionXY_sq 0 φ ψ χ).2]
  simp only [add_zero]
  change (1 - s) ^ 2 * (k * Real.cos χ ^ 2) ≤
      s ^ 2 * (ℓ * Real.cos χ ^ 2) ∧
    ℓ * Real.cos χ ^ 2 - k * Real.cos χ ^ 2 ≤ 1 - 2 * s
  rw [hℓ]
  constructor
  · have hfac : 0 ≤ (1 - 2 * s) * (s ^ 2 - k) :=
      mul_nonneg (by linarith) (sub_nonneg.mpr hk)
    have hprod : 0 ≤ Real.cos χ ^ 2 * ((1 - 2 * s) * (s ^ 2 - k)) :=
      mul_nonneg hc0 hfac
    nlinarith
  · nlinarith

/-- Quotient form of the first Walsh lower-section inequality, matching the
displayed equation in Supplementary Note 7. -/
theorem fourSection_walsh_lower_slope (φ ψ χ : ℝ)
    (hs0 : 0 < fourSectionS φ ψ)
    (hs : fourSectionS φ ψ < 1 / 2) :
    (1 - fourSectionS φ ψ) ^ 2 /
          fourSectionS φ ψ ^ 2 * fourSectionX φ ψ χ ^ 2 ≤
      fourSectionY 0 φ ψ χ ^ 2 := by
  have h := (fourSection_walsh_lower_bounds φ ψ χ hs).1
  have hs2 : 0 < fourSectionS φ ψ ^ 2 := sq_pos_of_pos hs0
  rw [div_mul_eq_mul_div]
  apply (div_le_iff₀ hs2).2
  nlinarith

/-- Algebraic equation of the two central-section ellipse boundaries.  The
two signs correspond to the two crossed disks in the manuscript. -/
theorem central_section_ellipse_identity
    (k α χ : ℝ) (hk : 0 ≤ k) :
    let x := Real.sqrt k * Real.cos χ
    let y := Real.sqrt k * Real.cos (χ + α)
    x ^ 2 + y ^ 2 - 2 * Real.cos α * x * y =
      k * Real.sin α ^ 2 := by
  dsimp
  rw [Real.cos_add]
  have hsqrt : Real.sqrt k ^ 2 = k := Real.sq_sqrt hk
  have hα : 1 - Real.cos α ^ 2 = Real.sin α ^ 2 := by
    nlinarith [Real.sin_sq_add_cos_sq α]
  have hχ : Real.cos χ ^ 2 + Real.sin χ ^ 2 = 1 := by
    nlinarith [Real.sin_sq_add_cos_sq χ]
  calc
    (Real.sqrt k * Real.cos χ) ^ 2 +
          (Real.sqrt k *
            (Real.cos χ * Real.cos α - Real.sin χ * Real.sin α)) ^ 2 -
        2 * Real.cos α * (Real.sqrt k * Real.cos χ) *
          (Real.sqrt k *
            (Real.cos χ * Real.cos α - Real.sin χ * Real.sin α)) =
        Real.sqrt k ^ 2 *
          (Real.cos χ ^ 2 * (1 - Real.cos α ^ 2) +
            Real.sin χ ^ 2 * Real.sin α ^ 2) := by ring
    _ = k * (Real.cos χ ^ 2 * (1 - Real.cos α ^ 2) +
          Real.sin χ ^ 2 * Real.sin α ^ 2) := by rw [hsqrt]
    _ = k * (Real.cos χ ^ 2 * Real.sin α ^ 2 +
          Real.sin χ ^ 2 * Real.sin α ^ 2) := by rw [hα]
    _ = k * ((Real.cos χ ^ 2 + Real.sin χ ^ 2) *
          Real.sin α ^ 2) := by ring
    _ = k * Real.sin α ^ 2 := by rw [hχ, one_mul]

/-- The largest central-section ellipse has `k = 1/4`; all smaller
fixed-`k` curves lie inside the absolute-value inequality used for the union
of the two crossed disks. -/
theorem central_section_family_bound
    (k α χ : ℝ) (hk0 : 0 ≤ k) (hk1 : k ≤ 1 / 4)
    (hcos : 0 ≤ Real.cos α) :
    let x := Real.sqrt k * Real.cos χ
    let y := Real.sqrt k * Real.cos (χ + α)
    x ^ 2 + y ^ 2 - 2 * Real.cos α * |x * y| ≤
      Real.sin α ^ 2 / 4 := by
  dsimp
  have hid := central_section_ellipse_identity k α χ hk0
  dsimp at hid
  let x := Real.sqrt k * Real.cos χ
  let y := Real.sqrt k * Real.cos (χ + α)
  have habs : x * y ≤ |x * y| := le_abs_self _
  have hcross : -2 * Real.cos α * |x * y| ≤
      -2 * Real.cos α * (x * y) := by
    have hmul : 2 * Real.cos α * (x * y) ≤
        2 * Real.cos α * |x * y| :=
      mul_le_mul_of_nonneg_left habs (mul_nonneg (by norm_num) hcos)
    linarith
  have hscale : k * Real.sin α ^ 2 ≤ Real.sin α ^ 2 / 4 := by
    have := mul_le_mul_of_nonneg_right hk1 (sq_nonneg (Real.sin α))
    nlinarith
  change x ^ 2 + y ^ 2 - 2 * Real.cos α * |x * y| ≤
      Real.sin α ^ 2 / 4
  change x ^ 2 + y ^ 2 - 2 * Real.cos α * x * y =
      k * Real.sin α ^ 2 at hid
  linarith

/-- Negating the second ellipse coordinate gives the other crossed disk and
obeys the same central-section bound. -/
theorem central_section_family_bound_neg
    (k α χ : ℝ) (hk0 : 0 ≤ k) (hk1 : k ≤ 1 / 4)
    (hcos : 0 ≤ Real.cos α) :
    let x := Real.sqrt k * Real.cos χ
    let y := -Real.sqrt k * Real.cos (χ + α)
    x ^ 2 + y ^ 2 - 2 * Real.cos α * |x * y| ≤
      Real.sin α ^ 2 / 4 := by
  simpa [abs_neg] using central_section_family_bound k α χ hk0 hk1 hcos

/-- Every point of the opposite `(1,3)` simplex edge occurs in the explicit
phase map, independently of the family parameter. -/
theorem opposite_edge_thirteen_reachable (x : ℝ)
    (hx0 : -1 ≤ x) (hx1 : x ≤ 1) :
    ∃ φ ψ χ : ℝ,
      fourSectionS φ ψ = 1 ∧
      fourSectionX φ ψ χ = x ∧
      ∀ α : ℝ, fourSectionY α φ ψ χ = 0 := by
  refine ⟨0, 0, Real.arccos x, ?_⟩
  rw [fourSectionS, fourSectionX]
  simp [Real.cos_arccos hx0 hx1]
  intro α
  simp [fourSectionY]

/-- Every point of the opposite `(2,4)` simplex edge likewise occurs for
every member of the continuous four-port family. -/
theorem opposite_edge_twofour_reachable (α y : ℝ)
    (hy0 : -1 ≤ y) (hy1 : y ≤ 1) :
    ∃ φ ψ χ : ℝ,
      fourSectionS φ ψ = 0 ∧
      fourSectionX φ ψ χ = 0 ∧
      fourSectionY α φ ψ χ = y := by
  refine ⟨Real.pi / 2, Real.pi / 2, Real.arccos y - α, ?_⟩
  rw [fourSectionS, fourSectionX, fourSectionY]
  simp [Real.cos_arccos hy0 hy1]

/-- At the Walsh endpoint, every split `p : 1-p` on a representative
adjacent edge is reached explicitly.  The other adjacent edges follow by the
row sign/permutation symmetries of the Walsh matrix. -/
theorem walsh_adjacent_edge_reachable (p : ℝ) (hp0 : 0 ≤ p) (hp1 : p ≤ 1) :
    ∃ θ : ℝ,
      fourSectionS θ θ = p ∧
      fourSectionX θ θ 0 = p ∧
      fourSectionY 0 θ θ 0 = 1 - p := by
  let θ := Real.arccos (Real.sqrt p)
  have hsqrt0 : 0 ≤ Real.sqrt p := Real.sqrt_nonneg p
  have hs0 : -(1 : ℝ) ≤ Real.sqrt p := by linarith
  have hs1 : Real.sqrt p ≤ 1 := by
    rw [Real.sqrt_le_one]
    exact hp1
  have hcos : Real.cos θ = Real.sqrt p := Real.cos_arccos hs0 hs1
  have hsqrt : Real.sqrt p ^ 2 = p := Real.sq_sqrt hp0
  have hcos_sq : Real.cos θ ^ 2 = p := by rw [hcos, hsqrt]
  refine ⟨θ, ?_, ?_, ?_⟩
  · rw [fourSectionS, hcos, hsqrt]
    ring
  · rw [fourSectionX, hcos]
    norm_num
    nlinarith
  · rw [fourSectionY]
    norm_num
    nlinarith [Real.sin_sq_add_cos_sq θ]

/-! ### Connection to the physical four-port transfer matrix -/

noncomputable def realPhase (t : ℝ) : Circle := Circle.exp t

@[simp] theorem realPhase_coe (t : ℝ) :
    (realPhase t : ℂ) =
      (Real.cos t : ℂ) + (Real.sin t : ℂ) * Complex.I := by
  rw [realPhase, Circle.coe_exp, Complex.exp_mul_I,
    ← Complex.ofReal_cos, ← Complex.ofReal_sin]

noncomputable def fourPhaseControl (φ ψ χ : ℝ) : Fin 4 → Circle :=
  ![realPhase φ, realPhase (χ + ψ), realPhase (-φ),
    realPhase (χ - ψ)]

noncomputable def fourPhaseField (α φ ψ χ : ℝ) : Fin 4 → ℂ := fun q ↦
  generalOutput (fourPortFamily (realPhase α)) (fun _ ↦ 1)
    (fourPhaseControl φ ψ χ) 0 q

noncomputable def fourPhasePower (α φ ψ χ : ℝ) : Fin 4 → ℝ :=
  fieldPower (fourPhaseField α φ ψ χ)

/-- Explicit field at output zero. -/
theorem fourPhaseField_zero (α φ ψ χ : ℝ) :
    fourPhaseField α φ ψ χ 0 =
      (2 : ℂ)⁻¹ * ((Real.cos φ : ℂ) +
        ((Real.cos χ : ℂ) + (Real.sin χ : ℂ) * Complex.I) * Real.cos ψ) := by
  simp [fourPhaseField, generalOutput, generalArmPhase, fourPortFamily,
    fourPhaseControl, Fin.sum_univ_four, Real.cos_add, Real.sin_add,
    Real.cos_sub, Real.sin_sub]
  simp only [← Complex.ofReal_cos, ← Complex.ofReal_sin]
  norm_num [Complex.I_mul_I]
  ring

/-- Explicit field at output two. -/
theorem fourPhaseField_two (α φ ψ χ : ℝ) :
    fourPhaseField α φ ψ χ 2 =
      (2 : ℂ)⁻¹ * ((Real.cos φ : ℂ) -
        ((Real.cos χ : ℂ) + (Real.sin χ : ℂ) * Complex.I) * Real.cos ψ) := by
  simp [fourPhaseField, generalOutput, generalArmPhase, fourPortFamily,
    fourPhaseControl, Fin.sum_univ_four, Real.cos_add, Real.sin_add,
    Real.cos_sub, Real.sin_sub]
  simp only [← Complex.ofReal_cos, ← Complex.ofReal_sin]
  norm_num [Complex.I_mul_I]
  ring

/-- Explicit field at output one. -/
theorem fourPhaseField_one (α φ ψ χ : ℝ) :
    fourPhaseField α φ ψ χ 1 =
      Complex.I * (2 : ℂ)⁻¹ * ((Real.sin φ : ℂ) +
        ((Real.cos (χ + α) : ℂ) +
          (Real.sin (χ + α) : ℂ) * Complex.I) * Real.sin ψ) := by
  simp [fourPhaseField, generalOutput, generalArmPhase, fourPortFamily,
    fourPhaseControl, Fin.sum_univ_four, Real.cos_add, Real.sin_add,
    Real.cos_sub, Real.sin_sub]
  simp only [← Complex.ofReal_cos, ← Complex.ofReal_sin]
  norm_num [Complex.I_mul_I]
  ring_nf
  norm_num [pow_two, Complex.I_mul_I]
  ring

/-- Explicit field at output three. -/
theorem fourPhaseField_three (α φ ψ χ : ℝ) :
    fourPhaseField α φ ψ χ 3 =
      Complex.I * (2 : ℂ)⁻¹ * ((Real.sin φ : ℂ) -
        ((Real.cos (χ + α) : ℂ) +
          (Real.sin (χ + α) : ℂ) * Complex.I) * Real.sin ψ) := by
  simp [fourPhaseField, generalOutput, generalArmPhase, fourPortFamily,
    fourPhaseControl, Fin.sum_univ_four, Real.cos_add, Real.sin_add,
    Real.cos_sub, Real.sin_sub]
  simp only [← Complex.ofReal_cos, ← Complex.ofReal_sin]
  norm_num [Complex.I_mul_I]
  ring_nf
  norm_num [pow_two, Complex.I_mul_I]

/-- The powers produced by the matrix family obey the exact section-height
formula used in Supplementary Note 7. -/
theorem fourPhasePower_sectionS (α φ ψ χ : ℝ) :
    fourPhasePower α φ ψ χ 0 + fourPhasePower α φ ψ χ 2 =
      fourSectionS φ ψ := by
  change ‖fourPhaseField α φ ψ χ 0‖ ^ 2 +
    ‖fourPhaseField α φ ψ χ 2‖ ^ 2 = fourSectionS φ ψ
  rw [fourPhaseField_zero, fourPhaseField_two,
    Complex.sq_norm, Complex.sq_norm, Complex.normSq_apply,
    Complex.normSq_apply]
  simp only [Complex.add_re, Complex.add_im, Complex.sub_re, Complex.sub_im,
    Complex.mul_re, Complex.mul_im, Complex.ofReal_re, Complex.ofReal_im,
    Complex.I_re, Complex.I_im, Complex.inv_re, Complex.inv_im,
    fourSectionS]
  norm_num
  ring_nf
  nlinarith [Real.sin_sq_add_cos_sq χ]

/-- The matrix-derived power difference `P₁-P₃` is exactly the section
coordinate `x`. -/
theorem fourPhasePower_sectionX (α φ ψ χ : ℝ) :
    fourPhasePower α φ ψ χ 0 - fourPhasePower α φ ψ χ 2 =
      fourSectionX φ ψ χ := by
  change ‖fourPhaseField α φ ψ χ 0‖ ^ 2 -
    ‖fourPhaseField α φ ψ χ 2‖ ^ 2 = fourSectionX φ ψ χ
  rw [fourPhaseField_zero, fourPhaseField_two,
    Complex.sq_norm, Complex.sq_norm, Complex.normSq_apply,
    Complex.normSq_apply]
  simp only [Complex.add_re, Complex.add_im, Complex.sub_re, Complex.sub_im,
    Complex.mul_re, Complex.mul_im, Complex.ofReal_re, Complex.ofReal_im,
    Complex.I_re, Complex.I_im, Complex.inv_re, Complex.inv_im,
    fourSectionX]
  norm_num
  ring

/-- The second matrix-derived power difference `P₂-P₄` is exactly `y`. -/
theorem fourPhasePower_sectionY (α φ ψ χ : ℝ) :
    fourPhasePower α φ ψ χ 1 - fourPhasePower α φ ψ χ 3 =
      fourSectionY α φ ψ χ := by
  change ‖fourPhaseField α φ ψ χ 1‖ ^ 2 -
    ‖fourPhaseField α φ ψ χ 3‖ ^ 2 = fourSectionY α φ ψ χ
  rw [fourPhaseField_one, fourPhaseField_three,
    Complex.sq_norm, Complex.sq_norm, Complex.normSq_apply,
    Complex.normSq_apply]
  simp only [Complex.add_re, Complex.add_im, Complex.sub_re, Complex.sub_im,
    Complex.mul_re, Complex.mul_im, Complex.ofReal_re, Complex.ofReal_im,
    Complex.I_re, Complex.I_im, Complex.inv_re, Complex.inv_im,
    fourSectionY]
  norm_num
  ring

/-- The three section coordinates calculated directly from the physical
transfer matrix are exactly the explicit trigonometric map used throughout
Supplementary Note 7. -/
theorem fourPhasePower_section_coordinates (α φ ψ χ : ℝ) :
    fourPhasePower α φ ψ χ 0 + fourPhasePower α φ ψ χ 2 =
        fourSectionS φ ψ ∧
      fourPhasePower α φ ψ χ 0 - fourPhasePower α φ ψ χ 2 =
        fourSectionX φ ψ χ ∧
      fourPhasePower α φ ψ χ 1 - fourPhasePower α φ ψ χ 3 =
        fourSectionY α φ ψ χ :=
  ⟨fourPhasePower_sectionS α φ ψ χ,
    fourPhasePower_sectionX α φ ψ χ,
    fourPhasePower_sectionY α φ ψ χ⟩

/-! ### Two-output edge phase matching -/

/-- The first Walsh-type inverse-control pair has equal moduli when the two
nonzero output amplitudes are in quadrature. -/
theorem quadrature_pair_normSq_eq (a b : ℝ) :
    Complex.normSq ((a : ℂ) + (b : ℂ) * Complex.I) =
      Complex.normSq ((a : ℂ) - (b : ℂ) * Complex.I) := by
  simp only [Complex.normSq_apply, Complex.add_re, Complex.add_im,
    Complex.sub_re, Complex.sub_im, Complex.mul_re, Complex.mul_im,
    Complex.ofReal_re, Complex.ofReal_im, Complex.I_re, Complex.I_im]
  ring

/-- Exact mismatch of the second inverse-control pair after imposing that
quadrature.  It is proportional to `sin α`; this is the phase-matching
obstruction for the four non-opposite simplex edges. -/
theorem rotated_quadrature_pair_normSq_difference (a b α : ℝ) :
    Complex.normSq
        ((a : ℂ) + (realPhase (-α) : ℂ) * (b : ℂ) * Complex.I) -
      Complex.normSq
        ((a : ℂ) - (realPhase (-α) : ℂ) * (b : ℂ) * Complex.I) =
      4 * a * b * Real.sin α := by
  rw [realPhase_coe]
  simp only [Real.cos_neg, Real.sin_neg, Complex.normSq_apply,
    Complex.add_re, Complex.add_im, Complex.sub_re, Complex.sub_im,
    Complex.mul_re, Complex.mul_im, Complex.ofReal_re, Complex.ofReal_im,
    Complex.I_re, Complex.I_im]
  ring

/-- For two genuinely nonzero output amplitudes, equal-modulus controls on
both arm pairs force the Walsh endpoint condition `sin α = 0`. -/
theorem adjacent_edge_phase_matching_forces_sin_zero
    (a b α : ℝ) (ha : a ≠ 0) (hb : b ≠ 0)
    (hmatch :
      Complex.normSq
          ((a : ℂ) + (realPhase (-α) : ℂ) * (b : ℂ) * Complex.I) =
        Complex.normSq
          ((a : ℂ) - (realPhase (-α) : ℂ) * (b : ℂ) * Complex.I)) :
    Real.sin α = 0 := by
  have hdiff := rotated_quadrature_pair_normSq_difference a b α
  rw [hmatch, sub_self] at hdiff
  have hab : 4 * a * b ≠ 0 := mul_ne_zero (mul_ne_zero (by norm_num) ha) hb
  exact (mul_eq_zero.mp hdiff.symm).resolve_left hab

/-- Hence every strict non-Walsh member `0 < α ≤ π/2` forbids a
nontrivial adjacent-edge split. -/
theorem adjacent_edge_no_phase_match
    (a b α : ℝ) (ha : a ≠ 0) (hb : b ≠ 0)
    (hα0 : 0 < α) (hα1 : α ≤ Real.pi / 2) :
    Complex.normSq
        ((a : ℂ) + (realPhase (-α) : ℂ) * (b : ℂ) * Complex.I) ≠
      Complex.normSq
        ((a : ℂ) - (realPhase (-α) : ℂ) * (b : ℂ) * Complex.I) := by
  intro hmatch
  have hzero := adjacent_edge_phase_matching_forces_sin_zero
    a b α ha hb hmatch
  have hαpi : α < Real.pi := by
    nlinarith [Real.pi_pos]
  have hpos : 0 < Real.sin α := Real.sin_pos_of_pos_of_lt_pi hα0 hαpi
  linarith

end TFLNRouter
