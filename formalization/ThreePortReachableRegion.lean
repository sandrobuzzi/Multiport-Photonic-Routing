import ThreePortBalanced
import SupplementaryGeometry
import Mathlib.FieldTheory.IsAlgClosed.Basic

/-!
# Exact three-port reachable region

This module proves the global converse behind the analytic three-port
reachable-region boundary.  Rather than invoking a Jordan-curve argument, it
characterizes the complete region by the triangle inequalities forced by the
cyclic field-correlation identity.
-/

namespace TFLNRouter

open Complex
open ComplexConjugate
open scoped BigOperators

/-- The three nonnegative field magnitudes associated with a power vector. -/
noncomputable def threePowerRadius (p : Fin 3 → ℝ) (j : Fin 3) : ℝ :=
  Real.sqrt (p j)

/-- Algebraic candidate for the exact three-port reachable power region.
The three pairwise products of field magnitudes must be side lengths of a
possibly degenerate triangle. -/
def ThreePowerTriangleRegion (p : Fin 3 → ℝ) : Prop :=
  (∀ j, 0 ≤ p j) ∧
  (∑ j, p j) = 1 ∧
  threePowerRadius p 0 * threePowerRadius p 1 ≤
      threePowerRadius p 1 * threePowerRadius p 2 +
        threePowerRadius p 2 * threePowerRadius p 0 ∧
  threePowerRadius p 1 * threePowerRadius p 2 ≤
      threePowerRadius p 2 * threePowerRadius p 0 +
        threePowerRadius p 0 * threePowerRadius p 1 ∧
  threePowerRadius p 2 * threePowerRadius p 0 ≤
      threePowerRadius p 0 * threePowerRadius p 1 +
        threePowerRadius p 1 * threePowerRadius p 2

/-- The cyclic zero-correlation identity forces the three magnitude products
to obey all three triangle inequalities. -/
theorem three_field_triangle_inequalities
    (a b c : ℂ)
    (hcorr : a * conj b + b * conj c + c * conj a = 0) :
    ‖a‖ * ‖b‖ ≤ ‖b‖ * ‖c‖ + ‖c‖ * ‖a‖ ∧
    ‖b‖ * ‖c‖ ≤ ‖c‖ * ‖a‖ + ‖a‖ * ‖b‖ ∧
    ‖c‖ * ‖a‖ ≤ ‖a‖ * ‖b‖ + ‖b‖ * ‖c‖ := by
  have hab : a * conj b = -(b * conj c + c * conj a) := by
    linear_combination hcorr
  have hbc : b * conj c = -(c * conj a + a * conj b) := by
    linear_combination hcorr
  have hca : c * conj a = -(a * conj b + b * conj c) := by
    linear_combination hcorr
  constructor
  · calc
      ‖a‖ * ‖b‖ = ‖a * conj b‖ := by simp [norm_mul]
      _ = ‖b * conj c + c * conj a‖ := by rw [hab, norm_neg]
      _ ≤ ‖b * conj c‖ + ‖c * conj a‖ := norm_add_le _ _
      _ = ‖b‖ * ‖c‖ + ‖c‖ * ‖a‖ := by simp [norm_mul]
  · constructor
    · calc
        ‖b‖ * ‖c‖ = ‖b * conj c‖ := by simp [norm_mul]
        _ = ‖c * conj a + a * conj b‖ := by rw [hbc, norm_neg]
        _ ≤ ‖c * conj a‖ + ‖a * conj b‖ := norm_add_le _ _
        _ = ‖c‖ * ‖a‖ + ‖a‖ * ‖b‖ := by simp [norm_mul]
    · calc
        ‖c‖ * ‖a‖ = ‖c * conj a‖ := by simp [norm_mul]
        _ = ‖a * conj b + b * conj c‖ := by rw [hca, norm_neg]
        _ ≤ ‖a * conj b‖ + ‖b * conj c‖ := norm_add_le _ _
        _ = ‖a‖ * ‖b‖ + ‖b‖ * ‖c‖ := by simp [norm_mul]

lemma threePowerRadius_fieldPower (z : Fin 3 → ℂ) (j : Fin 3) :
    threePowerRadius (fieldPower z) j = ‖z j‖ := by
  rw [threePowerRadius, fieldPower, Real.sqrt_sq_eq_abs,
    abs_of_nonneg (norm_nonneg _)]

/-- Every physical three-DFT phase setting lies in the triangle region. -/
theorem threePhaseOutput_mem_triangleRegion
    (w : Circle) (hw : (w : ℂ)^2 + (w : ℂ) + 1 = 0)
    (phase : Fin 3 → Circle) :
    ThreePowerTriangleRegion (fieldPower (threePhaseOutput w phase)) := by
  let z := threePhaseOutput w phase
  have hsum := threePhaseOutput_power_sum w hw phase
  have hcorr := threePhaseOutput_correlation w hw phase
  have htri := three_field_triangle_inequalities (z 0) (z 1) (z 2) hcorr
  refine ⟨fun j ↦ sq_nonneg ‖z j‖, ?_, ?_, ?_, ?_⟩
  · simpa [z, fieldPower, Fin.sum_univ_three] using hsum
  · simpa only [threePowerRadius_fieldPower] using htri.1
  · simpa only [threePowerRadius_fieldPower] using htri.2.1
  · simpa only [threePowerRadius_fieldPower] using htri.2.2

/-! ## Constructing phases from three triangle side lengths -/

/-- Three positive lengths satisfying the triangle inequalities admit unit
complex phases whose weighted sum vanishes and whose phase product is one.
The product normalization is the compatibility condition needed to recover
three underlying field phases from the three cyclic correlation terms. -/
theorem triangle_unit_phases_exist
    (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
    (hAB : A ≤ B + C) (hBC : B ≤ C + A) (hCA : C ≤ A + B) :
    ∃ ζ₀ ζ₁ ζ₂ : ℂ,
      ‖ζ₀‖ = 1 ∧ ‖ζ₁‖ = 1 ∧ ‖ζ₂‖ = 1 ∧
      ζ₀ * ζ₁ * ζ₂ = 1 ∧
      (A : ℂ) * ζ₀ + (B : ℂ) * ζ₁ + (C : ℂ) * ζ₂ = 0 := by
  let q := (C ^ 2 - A ^ 2 - B ^ 2) / (2 * A * B)
  have hden : 0 < 2 * A * B := mul_pos (mul_pos (by norm_num) hA) hB
  have hCsq : C ^ 2 ≤ (A + B) ^ 2 := by nlinarith
  have hdiffsq : (A - B) ^ 2 ≤ C ^ 2 := by
    have h₁ : A - B ≤ C := by linarith
    have h₂ : B - A ≤ C := by linarith
    nlinarith
  have hq0 : -(1 : ℝ) ≤ q := by
    dsimp [q]
    apply (le_div_iff₀ hden).2
    nlinarith
  have hq1 : q ≤ 1 := by
    dsimp [q]
    apply (div_le_iff₀ hden).2
    nlinarith
  have hqSq : q ^ 2 ≤ 1 := by nlinarith
  let t := Real.sqrt (1 - q ^ 2)
  have ht0 : 0 ≤ t := Real.sqrt_nonneg _
  have htSq : t ^ 2 = 1 - q ^ 2 := by
    dsimp [t]
    rw [Real.sq_sqrt]
    linarith
  let η₀ : ℂ := ⟨1, 0⟩
  let η₁ : ℂ := ⟨q, t⟩
  let z₂ : ℂ := ⟨-(A + B * q), -(B * t)⟩
  let η₂ : ℂ := (C : ℂ)⁻¹ * z₂
  have hη₀ : ‖η₀‖ = 1 := by
    rw [Complex.norm_def, Complex.normSq_apply]
    norm_num [η₀]
  have hη₁ : ‖η₁‖ = 1 := by
    rw [Complex.norm_def, Complex.normSq_apply]
    dsimp [η₁]
    rw [show q * q + t * t = 1 by nlinarith, Real.sqrt_one]
  have hz₂Sq : Complex.normSq z₂ = C ^ 2 := by
    rw [Complex.normSq_apply]
    dsimp [z₂]
    have hqeq : 2 * A * B * q = C ^ 2 - A ^ 2 - B ^ 2 := by
      dsimp [q]
      field_simp [ne_of_gt hA, ne_of_gt hB]
    nlinarith
  have hz₂ : ‖z₂‖ = C := by
    have hsquare : ‖z₂‖ ^ 2 = C ^ 2 := by
      rw [Complex.sq_norm, hz₂Sq]
    nlinarith [norm_nonneg z₂]
  have hη₂ : ‖η₂‖ = 1 := by
    dsimp [η₂]
    rw [norm_mul, norm_inv, Complex.norm_real, Real.norm_eq_abs,
      abs_of_pos hC, hz₂]
    field_simp [ne_of_gt hC]
  have hweighted :
      (A : ℂ) * η₀ + (B : ℂ) * η₁ + (C : ℂ) * η₂ = 0 := by
    apply Complex.ext
    · dsimp [η₀, η₁, η₂, z₂]
      simp [ne_of_gt hC]
      ring
    · dsimp [η₀, η₁, η₂, z₂]
      simp [ne_of_gt hC]
  let v := η₀ * η₁ * η₂
  have hvNorm : ‖v‖ = 1 := by
    dsimp [v]
    rw [norm_mul, norm_mul, hη₀, hη₁, hη₂]
    norm_num
  have hv : v ≠ 0 := by
    intro hv0
    rw [hv0, norm_zero] at hvNorm
    norm_num at hvNorm
  obtain ⟨u : ℂ, hu⟩ := IsAlgClosed.exists_pow_nat_eq (v⁻¹) (by norm_num : 0 < 3)
  have huNormCube : ‖u‖ ^ 3 = 1 := by
    calc
      ‖u‖ ^ 3 = ‖u ^ 3‖ := by rw [norm_pow]
      _ = ‖v⁻¹‖ := by rw [hu]
      _ = 1 := by rw [norm_inv, hvNorm, inv_one]
  have huNorm : ‖u‖ = 1 := by
    apply (show Odd 3 by decide).pow_injective
    simpa using huNormCube
  refine ⟨u * η₀, u * η₁, u * η₂, ?_, ?_, ?_, ?_, ?_⟩
  · rw [norm_mul, huNorm, hη₀, one_mul]
  · rw [norm_mul, huNorm, hη₁, one_mul]
  · rw [norm_mul, huNorm, hη₂, one_mul]
  · calc
      (u * η₀) * (u * η₁) * (u * η₂) = u ^ 3 * v := by
        dsimp [v]
        ring
      _ = v⁻¹ * v := by rw [hu]
      _ = 1 := inv_mul_cancel₀ hv
  · calc
      (A : ℂ) * (u * η₀) + (B : ℂ) * (u * η₁) +
          (C : ℂ) * (u * η₂) =
          u * ((A : ℂ) * η₀ + (B : ℂ) * η₁ + (C : ℂ) * η₂) := by ring
      _ = 0 := by rw [hweighted, mul_zero]

lemma conj_eq_inv_of_norm_eq_one (z : ℂ) (hz : ‖z‖ = 1) :
    conj z = z⁻¹ := by
  rw [Complex.inv_def]
  have hsq : Complex.normSq z = 1 := by
    rw [Complex.normSq_eq_norm_sq, hz]
    norm_num
  rw [hsq]
  simp

/-- For strictly positive powers, the triangle inequalities construct output
fields with the prescribed powers and zero cyclic correlation. -/
theorem triangleRegion_positive_fields_exist
    (p : Fin 3 → ℝ) (hreg : ThreePowerTriangleRegion p)
    (hp0 : 0 < p 0) (hp1 : 0 < p 1) (hp2 : 0 < p 2) :
    ∃ z : Fin 3 → ℂ,
      fieldPower z = p ∧
      z 0 * conj (z 1) + z 1 * conj (z 2) + z 2 * conj (z 0) = 0 := by
  let r₀ := threePowerRadius p 0
  let r₁ := threePowerRadius p 1
  let r₂ := threePowerRadius p 2
  have hr₀ : 0 < r₀ := Real.sqrt_pos.2 hp0
  have hr₁ : 0 < r₁ := Real.sqrt_pos.2 hp1
  have hr₂ : 0 < r₂ := Real.sqrt_pos.2 hp2
  have htri := hreg.2.2
  rcases triangle_unit_phases_exist
      (r₀ * r₁) (r₁ * r₂) (r₂ * r₀)
      (mul_pos hr₀ hr₁) (mul_pos hr₁ hr₂) (mul_pos hr₂ hr₀)
      htri.1 htri.2.1 htri.2.2 with
    ⟨ζ₀, ζ₁, ζ₂, hζ₀, hζ₁, hζ₂, hprod, hsum⟩
  have hζ₀ne : ζ₀ ≠ 0 := by
    intro h
    rw [h, norm_zero] at hζ₀
    norm_num at hζ₀
  have hζ₂ne : ζ₂ ≠ 0 := by
    intro h
    rw [h, norm_zero] at hζ₂
    norm_num at hζ₂
  have hphase : conj ζ₀ * conj ζ₂ = ζ₁ := by
    rw [conj_eq_inv_of_norm_eq_one ζ₀ hζ₀,
      conj_eq_inv_of_norm_eq_one ζ₂ hζ₂]
    calc
      ζ₀⁻¹ * ζ₂⁻¹ = (ζ₀ * ζ₂)⁻¹ := by rw [mul_inv_rev]; ring
      _ = ζ₁ := by
        apply inv_eq_of_mul_eq_one_right
        calc
          (ζ₀ * ζ₂) * ζ₁ = ζ₀ * ζ₁ * ζ₂ := by ring
          _ = 1 := hprod
  let z : Fin 3 → ℂ :=
    ![(r₀ : ℂ), (r₁ : ℂ) * conj ζ₀, (r₂ : ℂ) * ζ₂]
  have hz0 : ‖z 0‖ = r₀ := by simp [z, abs_of_pos hr₀]
  have hz1 : ‖z 1‖ = r₁ := by
    simp [z, norm_mul, hζ₀, abs_of_pos hr₁]
  have hz2 : ‖z 2‖ = r₂ := by
    simp [z, norm_mul, hζ₂, abs_of_pos hr₂]
  have hr₀sq : r₀ ^ 2 = p 0 := Real.sq_sqrt hp0.le
  have hr₁sq : r₁ ^ 2 = p 1 := Real.sq_sqrt hp1.le
  have hr₂sq : r₂ ^ 2 = p 2 := Real.sq_sqrt hp2.le
  refine ⟨z, ?_, ?_⟩
  · funext j
    fin_cases j
    · change ‖z 0‖ ^ 2 = p 0
      rw [hz0, hr₀sq]
    · change ‖z 1‖ ^ 2 = p 1
      rw [hz1, hr₁sq]
    · change ‖z 2‖ ^ 2 = p 2
      rw [hz2, hr₂sq]
  · change
      (r₀ : ℂ) * conj ((r₁ : ℂ) * conj ζ₀) +
        ((r₁ : ℂ) * conj ζ₀) * conj ((r₂ : ℂ) * ζ₂) +
        ((r₂ : ℂ) * ζ₂) * conj (r₀ : ℂ) = 0
    simp only [map_mul, conj_conj, conj_ofReal]
    calc
      (r₀ : ℂ) * ((r₁ : ℂ) * ζ₀) +
          ((r₁ : ℂ) * conj ζ₀) * ((r₂ : ℂ) * conj ζ₂) +
          ((r₂ : ℂ) * ζ₂) * (r₀ : ℂ) =
          ((r₀ * r₁ : ℝ) : ℂ) * ζ₀ +
            ((r₁ * r₂ : ℝ) : ℂ) * (conj ζ₀ * conj ζ₂) +
            ((r₂ * r₀ : ℝ) : ℂ) * ζ₂ := by norm_num; ring
      _ =
          ((r₀ * r₁ : ℝ) : ℂ) * ζ₀ +
            ((r₁ * r₂ : ℝ) : ℂ) * ζ₁ +
            ((r₂ * r₀ : ℝ) : ℂ) * ζ₂ := by rw [hphase]
      _ = 0 := hsum

/-! ## Inverting the three-point transform -/

/-- The unnormalized inverse three-point Fourier transform. -/
noncomputable def threeInverseOutput (w : Circle)
    (z : Fin 3 → ℂ) : Fin 3 → ℂ :=
  ![z 0 + z 1 + z 2,
    z 0 + (w : ℂ)^2 * z 1 + (w : ℂ) * z 2,
    z 0 + (w : ℂ) * z 1 + (w : ℂ)^2 * z 2]

/-- Power conservation and zero cyclic correlation make every component of
the inverse transform a unit phasor. -/
theorem threeInverseOutput_norm_one
    (w : Circle) (hw : (w : ℂ)^2 + (w : ℂ) + 1 = 0)
    (z : Fin 3 → ℂ)
    (hsum : ‖z 0‖ ^ 2 + ‖z 1‖ ^ 2 + ‖z 2‖ ^ 2 = 1)
    (hcorr : z 0 * conj (z 1) + z 1 * conj (z 2) +
      z 2 * conj (z 0) = 0) :
    ∀ j, ‖threeInverseOutput w z j‖ = 1 := by
  let a := z 0
  let b := z 1
  let c := z 2
  let W : ℂ := w
  have hwc : conj W = W ^ 2 := three_root_conj w hw
  have hw3 : W ^ 3 = 1 := three_root_cube w hw
  have hw4 : W ^ 4 = W := by
    calc W ^ 4 = W ^ 3 * W := by ring
         _ = W := by rw [hw3, one_mul]
  have hw22 : (W ^ 2) ^ 2 = W := by
    calc
      (W ^ 2) ^ 2 = W ^ 4 := by ring
      _ = W := hw4
  have hsum' := hsum
  rw [Complex.sq_norm, Complex.sq_norm, Complex.sq_norm] at hsum'
  have hsumC := congrArg (fun r : ℝ ↦ (r : ℂ)) hsum'
  simp only [ofReal_add, ofReal_one] at hsumC
  rw [Complex.normSq_eq_conj_mul_self,
    Complex.normSq_eq_conj_mul_self,
    Complex.normSq_eq_conj_mul_self] at hsumC
  have hcorr' : a * conj b + b * conj c + c * conj a = 0 := hcorr
  have hcorrConj : conj a * b + conj b * c + conj c * a = 0 := by
    have h := congrArg conj hcorr'
    simp only [map_add, map_mul, conj_conj, map_zero] at h
    linear_combination h
  have hone_of_conj_mul (u : ℂ) (hu : conj u * u = 1) : ‖u‖ = 1 := by
    have hsq : ‖u‖ ^ 2 = 1 := by
      rw [Complex.sq_norm, ← Complex.ofReal_inj]
      rw [Complex.normSq_eq_conj_mul_self]
      exact hu
    nlinarith [norm_nonneg u]
  intro j
  fin_cases j
  · apply hone_of_conj_mul
    change conj (a + b + c) * (a + b + c) = 1
    simp only [map_add]
    calc
      (conj a + conj b + conj c) * (a + b + c) =
          (conj a * a + conj b * b + conj c * c) +
          (a * conj b + b * conj c + c * conj a) +
          (conj a * b + conj b * c + conj c * a) := by ring
      _ = 1 := by rw [hcorr', hcorrConj]; simpa [a, b, c] using hsumC
  · apply hone_of_conj_mul
    change conj (a + W ^ 2 * b + W * c) *
      (a + W ^ 2 * b + W * c) = 1
    simp only [map_add, map_mul, map_pow, hwc]
    calc
      (conj a + (W ^ 2) ^ 2 * conj b + W ^ 2 * conj c) *
          (a + W ^ 2 * b + W * c) =
          (conj a * a + conj b * b + conj c * c) +
          W * (a * conj b + b * conj c + c * conj a) +
          W ^ 2 * (conj a * b + conj b * c + conj c * a) := by
            rw [hw22]
            ring_nf
            rw [hw3, hw4]
            ring
      _ = 1 := by rw [hcorr', hcorrConj]; simpa [a, b, c] using hsumC
  · apply hone_of_conj_mul
    change conj (a + W * b + W ^ 2 * c) *
      (a + W * b + W ^ 2 * c) = 1
    simp only [map_add, map_mul, map_pow, hwc]
    calc
      (conj a + W ^ 2 * conj b + (W ^ 2) ^ 2 * conj c) *
          (a + W * b + W ^ 2 * c) =
          (conj a * a + conj b * b + conj c * c) +
          W ^ 2 * (a * conj b + b * conj c + c * conj a) +
          W * (conj a * b + conj b * c + conj c * a) := by
            rw [hw22]
            ring_nf
            rw [hw3, hw4]
            ring
      _ = 1 := by rw [hcorr', hcorrConj]; simpa [a, b, c] using hsumC

/-- The three-point transform of its unnormalized inverse is the original
field vector. -/
theorem threePhaseOutput_of_coe_eq_inverse
    (w : Circle) (hw : (w : ℂ)^2 + (w : ℂ) + 1 = 0)
    (z : Fin 3 → ℂ) (phase : Fin 3 → Circle)
    (hphase : ∀ j, (phase j : ℂ) = threeInverseOutput w z j) :
    threePhaseOutput w phase = z := by
  let W : ℂ := w
  have hw2 : W ^ 2 = -W - 1 := by linear_combination hw
  have hw3 : W ^ 3 = 1 := three_root_cube w hw
  have hw4 : W ^ 4 = W := by
    calc
      W ^ 4 = W ^ 3 * W := by ring
      _ = W := by rw [hw3]; ring
  funext j
  fin_cases j
  · change threePhaseOutput w phase 0 = z 0
    rw [threePhaseOutput_zero, hphase 0, hphase 1, hphase 2]
    dsimp [threeInverseOutput, W]
    field_simp
    rw [hw2]
    ring
  · change threePhaseOutput w phase 1 = z 1
    rw [threePhaseOutput_one, hphase 0, hphase 1, hphase 2]
    dsimp [threeInverseOutput, W]
    field_simp
    change z 0 + z 1 + z 2 +
      W * (z 0 + W ^ 2 * z 1 + W * z 2) +
      W ^ 2 * (z 0 + W * z 1 + W ^ 2 * z 2) = z 1 * 3
    ring_nf
    rw [hw4, hw3, hw2]
    ring
  · change threePhaseOutput w phase 2 = z 2
    rw [threePhaseOutput_two w hw, hphase 0, hphase 1, hphase 2]
    dsimp [threeInverseOutput, W]
    field_simp
    change z 0 + z 1 + z 2 +
      W ^ 2 * (z 0 + W ^ 2 * z 1 + W * z 2) +
      W * (z 0 + W * z 1 + W ^ 2 * z 2) = z 2 * 3
    ring_nf
    rw [hw4, hw3, hw2]
    ring

/-- Every unit-power, zero-correlation output field is the three-point
Fourier transform of physical unit-modulus arm phases. -/
theorem zeroCorrelation_fields_exist_phase
    (w : Circle) (hw : (w : ℂ)^2 + (w : ℂ) + 1 = 0)
    (z : Fin 3 → ℂ)
    (hsum : ‖z 0‖ ^ 2 + ‖z 1‖ ^ 2 + ‖z 2‖ ^ 2 = 1)
    (hcorr : z 0 * conj (z 1) + z 1 * conj (z 2) +
      z 2 * conj (z 0) = 0) :
    ∃ phase : Fin 3 → Circle, threePhaseOutput w phase = z := by
  have hnorm := threeInverseOutput_norm_one w hw z hsum hcorr
  let phase : Fin 3 → Circle := fun j ↦
    ⟨threeInverseOutput w z j, mem_sphere_zero_iff_norm.2 (hnorm j)⟩
  refine ⟨phase, threePhaseOutput_of_coe_eq_inverse w hw z phase ?_⟩
  intro j
  rfl

/-- Strictly positive points of the triangle region are physically produced
by three reflector phases. -/
theorem triangleRegion_positive_exists_phase
    (w : Circle) (hw : (w : ℂ)^2 + (w : ℂ) + 1 = 0)
    (p : Fin 3 → ℝ) (hreg : ThreePowerTriangleRegion p)
    (hp0 : 0 < p 0) (hp1 : 0 < p 1) (hp2 : 0 < p 2) :
    ∃ phase : Fin 3 → Circle,
      fieldPower (threePhaseOutput w phase) = p := by
  rcases triangleRegion_positive_fields_exist p hreg hp0 hp1 hp2 with
    ⟨z, hp, hcorr⟩
  have hsum : ‖z 0‖ ^ 2 + ‖z 1‖ ^ 2 + ‖z 2‖ ^ 2 = 1 := by
    have h := hreg.2.1
    rw [← hp] at h
    simpa [fieldPower, Fin.sum_univ_three] using h
  rcases zeroCorrelation_fields_exist_phase w hw z hsum hcorr with
    ⟨phase, hphase⟩
  refine ⟨phase, ?_⟩
  rw [hphase, hp]

/-! ## Degenerate points and the exact phase-level region -/

/-- Equal arm phases route all power to output zero. -/
noncomputable def threeVertexPhaseZero : Fin 3 → Circle := ![1, 1, 1]

/-- The conjugate Fourier character routes all power to output one. -/
noncomputable def threeVertexPhaseOne (w : Circle) : Fin 3 → Circle :=
  ![1, w ^ 2, w]

/-- The Fourier character routes all power to output two. -/
noncomputable def threeVertexPhaseTwo (w : Circle) : Fin 3 → Circle :=
  ![1, w, w ^ 2]

theorem threeVertexPhaseZero_powers
    (w : Circle) (hw : (w : ℂ)^2 + (w : ℂ) + 1 = 0) :
    fieldPower (threePhaseOutput w threeVertexPhaseZero) =
      ![(1 : ℝ), 0, 0] := by
  have h0 : threePhaseOutput w threeVertexPhaseZero 0 = 1 := by
    rw [threePhaseOutput_zero]
    change ((1 : ℂ) + 1 + 1) / 3 = 1
    norm_num
  have h1 : threePhaseOutput w threeVertexPhaseZero 1 = 0 := by
    rw [threePhaseOutput_one]
    rw [show threeVertexPhaseZero 0 = 1 by rfl,
      show threeVertexPhaseZero 1 = 1 by rfl,
      show threeVertexPhaseZero 2 = 1 by rfl]
    simp only [Circle.coe_one, Circle.coe_mul, mul_one]
    linear_combination (1 / 3 : ℂ) * hw
  have h2 : threePhaseOutput w threeVertexPhaseZero 2 = 0 := by
    rw [threePhaseOutput_two w hw]
    rw [show threeVertexPhaseZero 0 = 1 by rfl,
      show threeVertexPhaseZero 1 = 1 by rfl,
      show threeVertexPhaseZero 2 = 1 by rfl]
    simp only [Circle.coe_one, Circle.coe_mul, mul_one]
    linear_combination (1 / 3 : ℂ) * hw
  funext j
  fin_cases j <;> simp [fieldPower, h0, h1, h2]

theorem threeVertexPhaseOne_powers
    (w : Circle) (hw : (w : ℂ)^2 + (w : ℂ) + 1 = 0) :
    fieldPower (threePhaseOutput w (threeVertexPhaseOne w)) =
      ![(0 : ℝ), 1, 0] := by
  have hw3 : (w : ℂ) ^ 3 = 1 := three_root_cube w hw
  have hw4 : (w : ℂ) ^ 4 = (w : ℂ) := by
    calc
      (w : ℂ) ^ 4 = (w : ℂ) ^ 3 * (w : ℂ) := by ring
      _ = (w : ℂ) := by rw [hw3]; ring
  have h0 : threePhaseOutput w (threeVertexPhaseOne w) 0 = 0 := by
    rw [threePhaseOutput_zero]
    change ((1 : ℂ) + (w : ℂ)^2 + (w : ℂ)) / 3 = 0
    linear_combination (1 / 3 : ℂ) * hw
  have h1 : threePhaseOutput w (threeVertexPhaseOne w) 1 = 1 := by
    rw [threePhaseOutput_one]
    change ((1 : ℂ) + (w : ℂ) * (w : ℂ)^2 +
      (w : ℂ)^2 * (w : ℂ)) / 3 = 1
    rw [show (w : ℂ) * (w : ℂ)^2 = (w : ℂ)^3 by ring,
      show (w : ℂ)^2 * (w : ℂ) = (w : ℂ)^3 by ring, hw3]
    norm_num
  have h2 : threePhaseOutput w (threeVertexPhaseOne w) 2 = 0 := by
    rw [threePhaseOutput_two w hw]
    change ((1 : ℂ) + (w : ℂ)^2 * (w : ℂ)^2 +
      (w : ℂ) * (w : ℂ)) / 3 = 0
    rw [show (w : ℂ)^2 * (w : ℂ)^2 = (w : ℂ)^4 by ring, hw4]
    linear_combination (1 / 3 : ℂ) * hw
  funext j
  fin_cases j <;> simp [fieldPower, h0, h1, h2]

theorem threeVertexPhaseTwo_powers
    (w : Circle) (hw : (w : ℂ)^2 + (w : ℂ) + 1 = 0) :
    fieldPower (threePhaseOutput w (threeVertexPhaseTwo w)) =
      ![(0 : ℝ), 0, 1] := by
  have hw3 : (w : ℂ) ^ 3 = 1 := three_root_cube w hw
  have hw4 : (w : ℂ) ^ 4 = (w : ℂ) := by
    calc
      (w : ℂ) ^ 4 = (w : ℂ) ^ 3 * (w : ℂ) := by ring
      _ = (w : ℂ) := by rw [hw3]; ring
  have h0 : threePhaseOutput w (threeVertexPhaseTwo w) 0 = 0 := by
    rw [threePhaseOutput_zero]
    change ((1 : ℂ) + (w : ℂ) + (w : ℂ)^2) / 3 = 0
    linear_combination (1 / 3 : ℂ) * hw
  have h1 : threePhaseOutput w (threeVertexPhaseTwo w) 1 = 0 := by
    rw [threePhaseOutput_one]
    change ((1 : ℂ) + (w : ℂ) * (w : ℂ) +
      (w : ℂ)^2 * (w : ℂ)^2) / 3 = 0
    rw [show (w : ℂ)^2 * (w : ℂ)^2 = (w : ℂ)^4 by ring, hw4]
    linear_combination (1 / 3 : ℂ) * hw
  have h2 : threePhaseOutput w (threeVertexPhaseTwo w) 2 = 1 := by
    rw [threePhaseOutput_two w hw]
    change ((1 : ℂ) + (w : ℂ)^2 * (w : ℂ) +
      (w : ℂ) * (w : ℂ)^2) / 3 = 1
    rw [show (w : ℂ)^2 * (w : ℂ) = (w : ℂ)^3 by ring,
      show (w : ℂ) * (w : ℂ)^2 = (w : ℂ)^3 by ring, hw3]
    norm_num
  funext j
  fin_cases j <;> simp [fieldPower, h0, h1, h2]

/-- The triangle inequalities are not merely necessary: every point they
describe is produced by physical unit-modulus arm phases. -/
theorem triangleRegion_exists_phase
    (w : Circle) (hw : (w : ℂ)^2 + (w : ℂ) + 1 = 0)
    (p : Fin 3 → ℝ) (hreg : ThreePowerTriangleRegion p) :
    ∃ phase : Fin 3 → Circle,
      fieldPower (threePhaseOutput w phase) = p := by
  have hp0nonneg : 0 ≤ p 0 := hreg.1 0
  have hp1nonneg : 0 ≤ p 1 := hreg.1 1
  have hp2nonneg : 0 ≤ p 2 := hreg.1 2
  have hsum : p 0 + p 1 + p 2 = 1 := by
    simpa [Fin.sum_univ_three] using hreg.2.1
  by_cases hp0z : p 0 = 0
  · have hprodle : threePowerRadius p 1 * threePowerRadius p 2 ≤ 0 := by
      have h := hreg.2.2.2.1
      simpa [threePowerRadius, hp0z] using h
    have hprod : threePowerRadius p 1 * threePowerRadius p 2 = 0 :=
      le_antisymm hprodle
        (mul_nonneg (Real.sqrt_nonneg _) (Real.sqrt_nonneg _))
    rcases mul_eq_zero.mp hprod with hs1 | hs2
    · have hp1z : p 1 = 0 :=
        (Real.sqrt_eq_zero hp1nonneg).mp hs1
      have hp2one : p 2 = 1 := by linarith
      have hp : p = ![(0 : ℝ), 0, 1] := by
        funext j
        fin_cases j <;> simp [hp0z, hp1z, hp2one]
      exact ⟨threeVertexPhaseTwo w,
        (threeVertexPhaseTwo_powers w hw).trans hp.symm⟩
    · have hp2z : p 2 = 0 :=
        (Real.sqrt_eq_zero hp2nonneg).mp hs2
      have hp1one : p 1 = 1 := by linarith
      have hp : p = ![(0 : ℝ), 1, 0] := by
        funext j
        fin_cases j <;> simp [hp0z, hp1one, hp2z]
      exact ⟨threeVertexPhaseOne w,
        (threeVertexPhaseOne_powers w hw).trans hp.symm⟩
  · have hp0pos : 0 < p 0 := lt_of_le_of_ne hp0nonneg (Ne.symm hp0z)
    by_cases hp1z : p 1 = 0
    · have hprodle :
          threePowerRadius p 2 * threePowerRadius p 0 ≤ 0 := by
        have h := hreg.2.2.2.2
        simpa [threePowerRadius, hp1z] using h
      have hprod : threePowerRadius p 2 * threePowerRadius p 0 = 0 :=
        le_antisymm hprodle
          (mul_nonneg (Real.sqrt_nonneg _) (Real.sqrt_nonneg _))
      have hs0 : threePowerRadius p 0 ≠ 0 := by
        exact ne_of_gt (Real.sqrt_pos.2 hp0pos)
      have hs2 : threePowerRadius p 2 = 0 :=
        (mul_eq_zero.mp hprod).resolve_right hs0
      have hp2z : p 2 = 0 :=
        (Real.sqrt_eq_zero hp2nonneg).mp hs2
      have hp0one : p 0 = 1 := by linarith
      have hp : p = ![(1 : ℝ), 0, 0] := by
        funext j
        fin_cases j <;> simp [hp0one, hp1z, hp2z]
      exact ⟨threeVertexPhaseZero,
        (threeVertexPhaseZero_powers w hw).trans hp.symm⟩
    · have hp1pos : 0 < p 1 := lt_of_le_of_ne hp1nonneg (Ne.symm hp1z)
      have hp2pos : 0 < p 2 := by
        by_contra hn
        have hp2z : p 2 = 0 := le_antisymm (not_lt.mp hn) hp2nonneg
        have h := hreg.2.2.1
        have hpos : 0 < threePowerRadius p 0 * threePowerRadius p 1 :=
          mul_pos (Real.sqrt_pos.2 hp0pos) (Real.sqrt_pos.2 hp1pos)
        simp [threePowerRadius, hp2z] at h
        have hpos' : 0 < Real.sqrt (p 0) * Real.sqrt (p 1) := by
          simpa [threePowerRadius] using hpos
        exact (not_lt_of_ge h hpos')
      exact triangleRegion_positive_exists_phase w hw p hreg hp0pos hp1pos hp2pos

/-- Exact global characterization of the three-port phase map. -/
theorem threePhaseOutput_reachable_iff_triangleRegion
    (w : Circle) (hw : (w : ℂ)^2 + (w : ℂ) + 1 = 0)
    (p : Fin 3 → ℝ) :
    (∃ phase : Fin 3 → Circle,
      fieldPower (threePhaseOutput w phase) = p) ↔
      ThreePowerTriangleRegion p := by
  constructor
  · rintro ⟨phase, rfl⟩
    exact threePhaseOutput_mem_triangleRegion w hw phase
  · exact triangleRegion_exists_phase w hw p

/-- Device-level form of the exact theorem.  It applies for every fixed
fabrication phase and every selected input port because the programmable
reflector phases absorb both effects. -/
theorem threeDFT_reachablePowers_iff_triangleRegion
    (w : Circle) (hw : (w : ℂ)^2 + (w : ℂ) + 1 = 0)
    (fixed : Fin 3 → Circle) (n : Fin 3) (p : Fin 3 → ℝ) :
    p ∈ reachablePowers (threeDFT w) fixed n ↔
      ThreePowerTriangleRegion p := by
  constructor
  · rintro ⟨z, ⟨control, rfl⟩, rfl⟩
    have hout :
        (fun q ↦ generalOutput (threeDFT w) fixed control n q) =
          threePhaseOutput w (threeEffectivePhase w fixed control n) := by
      funext q
      exact threeDFT_generalOutput_eq_threePhaseOutput w fixed control n q
    rw [hout]
    exact threePhaseOutput_mem_triangleRegion w hw _
  · intro hreg
    rcases triangleRegion_exists_phase w hw p hreg with ⟨phase, hphase⟩
    let control := controlForThreePhase w fixed phase n
    refine ⟨_, ⟨control, rfl⟩, ?_⟩
    rw [show (fun q ↦ generalOutput (threeDFT w) fixed control n q) =
        threePhaseOutput w phase by
      funext q
      rw [threeDFT_generalOutput_eq_threePhaseOutput]
      congr 1
      funext j
      exact threeEffectivePhase_controlFor w fixed phase n j]
    exact hphase

/-! ## Completeness of the analytic boundary curve -/

/-- Every real point on the intersection of the unit sphere with the plane
`x₀+x₁+x₂=1` has the cosine parametrization used by the manuscript. -/
theorem three_plane_sphere_parametrization
    (x₀ x₁ x₂ : ℝ) (hsum : x₀ + x₁ + x₂ = 1)
    (hsq : x₀ ^ 2 + x₁ ^ 2 + x₂ ^ 2 = 1) :
    ∃ t : ℝ, threeBoundaryVector t = ![x₀ ^ 2, x₁ ^ 2, x₂ ^ 2] := by
  let c := (3 * x₀ - 1) / 2
  let s := Real.sqrt 3 * (x₁ - x₂) / 2
  have hsqrt : (Real.sqrt 3) ^ 2 = 3 := Real.sq_sqrt (by norm_num)
  have hcs : c ^ 2 + s ^ 2 = 1 := by
    dsimp [c, s]
    rw [show (Real.sqrt 3 * (x₁ - x₂) / 2) ^ 2 =
        3 * (x₁ - x₂) ^ 2 / 4 by
      calc
        (Real.sqrt 3 * (x₁ - x₂) / 2) ^ 2 =
            (Real.sqrt 3) ^ 2 * (x₁ - x₂) ^ 2 / 4 := by ring
        _ = 3 * (x₁ - x₂) ^ 2 / 4 := by rw [hsqrt]]
    nlinarith
  let z : ℂ := ⟨c, s⟩
  have hnorm : ‖z‖ = 1 := by
    rw [Complex.norm_def, Complex.normSq_apply]
    change Real.sqrt (c * c + s * s) = 1
    rw [show c * c + s * s = 1 by nlinarith, Real.sqrt_one]
  have hz : z ≠ 0 := by
    intro hz0
    rw [hz0, norm_zero] at hnorm
    norm_num at hnorm
  let t := Complex.arg z
  have hcos : Real.cos t = c := by
    dsimp [t]
    rw [Complex.cos_arg hz, hnorm]
    simp [z]
  have hsin : Real.sin t = s := by
    dsimp [t]
    rw [Complex.sin_arg, hnorm]
    simp [z]
  refine ⟨t, ?_⟩
  funext j
  fin_cases j
  · change threeBoundaryCoordinate t = x₀ ^ 2
    rw [threeBoundaryCoordinate, hcos]
    dsimp [c]
    ring
  · change threeBoundaryCoordinate (t - 2 * Real.pi / 3) = x₁ ^ 2
    rw [threeBoundaryCoordinate, Real.cos_sub,
      real_cos_two_pi_div_three, real_sin_two_pi_div_three, hcos, hsin]
    dsimp [c, s]
    have hprod : Real.sqrt 3 * (x₁ - x₂) / 2 *
        (Real.sqrt 3 / 2) = 3 * (x₁ - x₂) / 4 := by
      calc
        Real.sqrt 3 * (x₁ - x₂) / 2 * (Real.sqrt 3 / 2) =
            (Real.sqrt 3) ^ 2 * (x₁ - x₂) / 4 := by ring
        _ = 3 * (x₁ - x₂) / 4 := by rw [hsqrt]
    rw [hprod]
    rw [show 1 + 2 * ((3 * x₀ - 1) / 2 * (-(1 : ℝ) / 2) +
        3 * (x₁ - x₂) / 4) = 3 * x₁ by linarith]
    ring
  · change threeBoundaryCoordinate (t + 2 * Real.pi / 3) = x₂ ^ 2
    rw [threeBoundaryCoordinate, Real.cos_add,
      real_cos_two_pi_div_three, real_sin_two_pi_div_three, hcos, hsin]
    dsimp [c, s]
    have hprod : Real.sqrt 3 * (x₁ - x₂) / 2 *
        (Real.sqrt 3 / 2) = 3 * (x₁ - x₂) / 4 := by
      calc
        Real.sqrt 3 * (x₁ - x₂) / 2 * (Real.sqrt 3 / 2) =
            (Real.sqrt 3) ^ 2 * (x₁ - x₂) / 4 := by ring
        _ = 3 * (x₁ - x₂) / 4 := by rw [hsqrt]
    rw [hprod]
    rw [show 1 + 2 * ((3 * x₀ - 1) / 2 * (-(1 : ℝ) / 2) -
        3 * (x₁ - x₂) / 4) = 3 * x₂ by linarith]
    ring

/-- A signed square-root vector with zero pair correlation lies on the
manuscript's closed boundary curve. -/
theorem threeBoundaryVector_exists_of_signed_roots
    (p : Fin 3 → ℝ) (x₀ x₁ x₂ : ℝ)
    (hsum : p 0 + p 1 + p 2 = 1)
    (hx₀ : x₀ ^ 2 = p 0) (hx₁ : x₁ ^ 2 = p 1) (hx₂ : x₂ ^ 2 = p 2)
    (hcorr : x₀ * x₁ + x₁ * x₂ + x₂ * x₀ = 0) :
    ∃ t : ℝ, threeBoundaryVector t = p := by
  have htotalSq : (x₀ + x₁ + x₂) ^ 2 = 1 := by
    calc
      (x₀ + x₁ + x₂) ^ 2 =
          x₀ ^ 2 + x₁ ^ 2 + x₂ ^ 2 +
            2 * (x₀ * x₁ + x₁ * x₂ + x₂ * x₀) := by ring
      _ = 1 := by rw [hx₀, hx₁, hx₂, hcorr]; linarith
  rcases sq_eq_one_iff.mp htotalSq with hplus | hminus
  · rcases three_plane_sphere_parametrization x₀ x₁ x₂ hplus (by
        rw [hx₀, hx₁, hx₂]
        exact hsum) with ⟨t, ht⟩
    refine ⟨t, ht.trans ?_⟩
    funext j
    fin_cases j <;> simp [hx₀, hx₁, hx₂]
  · have hplus : -x₀ + -x₁ + -x₂ = 1 := by linarith
    rcases three_plane_sphere_parametrization (-x₀) (-x₁) (-x₂) hplus (by
        rw [neg_sq, neg_sq, neg_sq, hx₀, hx₁, hx₂]
        exact hsum) with ⟨t, ht⟩
    refine ⟨t, ht.trans ?_⟩
    funext j
    fin_cases j <;> simp [hx₀, hx₁, hx₂]

/-- Boundary points of the triangle region are precisely those at which at
least one of its three defining triangle inequalities is saturated. -/
def ThreePowerTriangleBoundary (p : Fin 3 → ℝ) : Prop :=
  ThreePowerTriangleRegion p ∧
    (threePowerRadius p 0 * threePowerRadius p 1 =
        threePowerRadius p 1 * threePowerRadius p 2 +
          threePowerRadius p 2 * threePowerRadius p 0 ∨
     threePowerRadius p 1 * threePowerRadius p 2 =
        threePowerRadius p 2 * threePowerRadius p 0 +
          threePowerRadius p 0 * threePowerRadius p 1 ∨
     threePowerRadius p 2 * threePowerRadius p 0 =
        threePowerRadius p 0 * threePowerRadius p 1 +
          threePowerRadius p 1 * threePowerRadius p 2)

/-- Three real numbers summing to zero have absolute values forming a
degenerate triangle. -/
theorem abs_triangle_saturation (a b c : ℝ) (h : a + b + c = 0) :
    |a| = |b| + |c| ∨ |b| = |c| + |a| ∨ |c| = |a| + |b| := by
  by_cases ha : 0 ≤ a
  · by_cases hb : 0 ≤ b
    · by_cases hc : 0 ≤ c
      · have : a = 0 ∧ b = 0 ∧ c = 0 := by
          constructor
          · linarith
          · constructor <;> linarith
        rcases this with ⟨rfl, rfl, rfl⟩
        simp
      · right; right
        rw [abs_of_nonneg ha, abs_of_nonneg hb,
          abs_of_nonpos (le_of_not_ge hc)]
        linarith
    · by_cases hc : 0 ≤ c
      · right; left
        rw [abs_of_nonneg ha, abs_of_nonpos (le_of_not_ge hb),
          abs_of_nonneg hc]
        linarith
      · left
        rw [abs_of_nonneg ha, abs_of_nonpos (le_of_not_ge hb),
          abs_of_nonpos (le_of_not_ge hc)]
        linarith
  · by_cases hb : 0 ≤ b
    · by_cases hc : 0 ≤ c
      · left
        rw [abs_of_nonpos (le_of_not_ge ha), abs_of_nonneg hb,
          abs_of_nonneg hc]
        linarith
      · right; left
        rw [abs_of_nonpos (le_of_not_ge ha), abs_of_nonneg hb,
          abs_of_nonpos (le_of_not_ge hc)]
        linarith
    · by_cases hc : 0 ≤ c
      · right; right
        rw [abs_of_nonpos (le_of_not_ge ha),
          abs_of_nonpos (le_of_not_ge hb), abs_of_nonneg hc]
        linarith
      · exfalso
        linarith

/-- Signed square roots of the three coordinates on the analytic curve. -/
noncomputable def threeBoundarySigned (t : ℝ) : Fin 3 → ℝ :=
  ![(1 + 2 * Real.cos t) / 3,
    (1 + 2 * Real.cos (t - 2 * Real.pi / 3)) / 3,
    (1 + 2 * Real.cos (t + 2 * Real.pi / 3)) / 3]

lemma threeBoundarySigned_sq (t : ℝ) (j : Fin 3) :
    (threeBoundarySigned t j) ^ 2 = threeBoundaryVector t j := by
  fin_cases j <;> simp [threeBoundarySigned, threeBoundaryVector,
    threeBoundaryCoordinate] <;> ring

lemma threeBoundarySigned_explicit_one (t : ℝ) :
    threeBoundarySigned t 1 =
      (1 - Real.cos t + Real.sqrt 3 * Real.sin t) / 3 := by
  change (1 + 2 * Real.cos (t - 2 * Real.pi / 3)) / 3 = _
  rw [Real.cos_sub, real_cos_two_pi_div_three,
    real_sin_two_pi_div_three]
  ring

lemma threeBoundarySigned_explicit_two (t : ℝ) :
    threeBoundarySigned t 2 =
      (1 - Real.cos t - Real.sqrt 3 * Real.sin t) / 3 := by
  change (1 + 2 * Real.cos (t + 2 * Real.pi / 3)) / 3 = _
  rw [Real.cos_add, real_cos_two_pi_div_three,
    real_sin_two_pi_div_three]
  ring

lemma threeBoundarySigned_square_sum (t : ℝ) :
    (threeBoundarySigned t 0) ^ 2 +
      (threeBoundarySigned t 1) ^ 2 +
      (threeBoundarySigned t 2) ^ 2 = 1 := by
  rw [threeBoundarySigned_explicit_one, threeBoundarySigned_explicit_two]
  change ((1 + 2 * Real.cos t) / 3) ^ 2 +
    ((1 - Real.cos t + Real.sqrt 3 * Real.sin t) / 3) ^ 2 +
    ((1 - Real.cos t - Real.sqrt 3 * Real.sin t) / 3) ^ 2 = 1
  have hsqrt : (Real.sqrt 3) ^ 2 = 3 := Real.sq_sqrt (by norm_num)
  nlinarith [Real.sin_sq_add_cos_sq t]

lemma threeBoundarySigned_pair_sum (t : ℝ) :
    threeBoundarySigned t 0 * threeBoundarySigned t 1 +
      threeBoundarySigned t 1 * threeBoundarySigned t 2 +
      threeBoundarySigned t 2 * threeBoundarySigned t 0 = 0 := by
  rw [threeBoundarySigned_explicit_one, threeBoundarySigned_explicit_two]
  change ((1 + 2 * Real.cos t) / 3) *
      ((1 - Real.cos t + Real.sqrt 3 * Real.sin t) / 3) +
    ((1 - Real.cos t + Real.sqrt 3 * Real.sin t) / 3) *
      ((1 - Real.cos t - Real.sqrt 3 * Real.sin t) / 3) +
    ((1 - Real.cos t - Real.sqrt 3 * Real.sin t) / 3) *
      ((1 + 2 * Real.cos t) / 3) = 0
  have hsqrt : (Real.sqrt 3) ^ 2 = 3 := Real.sq_sqrt (by norm_num)
  nlinarith [Real.sin_sq_add_cos_sq t]

/-- Every point on the analytic curve saturates one triangle inequality and
is therefore on the algebraic boundary of the exact reachable region. -/
theorem threeBoundaryVector_mem_triangleBoundary (t : ℝ) :
    ThreePowerTriangleBoundary (threeBoundaryVector t) := by
  let x₀ := threeBoundarySigned t 0
  let x₁ := threeBoundarySigned t 1
  let x₂ := threeBoundarySigned t 2
  have hs0 : x₀ ^ 2 = threeBoundaryVector t 0 :=
    threeBoundarySigned_sq t 0
  have hs1 : x₁ ^ 2 = threeBoundaryVector t 1 :=
    threeBoundarySigned_sq t 1
  have hs2 : x₂ ^ 2 = threeBoundaryVector t 2 :=
    threeBoundarySigned_sq t 2
  have hsum : ∑ j, threeBoundaryVector t j = 1 := by
    simp only [Fin.sum_univ_three]
    rw [← hs0, ← hs1, ← hs2]
    exact threeBoundarySigned_square_sum t
  have hr0 : threePowerRadius (threeBoundaryVector t) 0 = |x₀| := by
    rw [threePowerRadius, ← hs0, Real.sqrt_sq_eq_abs]
  have hr1 : threePowerRadius (threeBoundaryVector t) 1 = |x₁| := by
    rw [threePowerRadius, ← hs1, Real.sqrt_sq_eq_abs]
  have hr2 : threePowerRadius (threeBoundaryVector t) 2 = |x₂| := by
    rw [threePowerRadius, ← hs2, Real.sqrt_sq_eq_abs]
  have hsat := abs_triangle_saturation (x₀ * x₁) (x₁ * x₂) (x₂ * x₀)
    (threeBoundarySigned_pair_sum t)
  simp only [abs_mul] at hsat
  have htri :
      |x₀| * |x₁| ≤ |x₁| * |x₂| + |x₂| * |x₀| ∧
      |x₁| * |x₂| ≤ |x₂| * |x₀| + |x₀| * |x₁| ∧
      |x₂| * |x₀| ≤ |x₀| * |x₁| + |x₁| * |x₂| := by
    rcases hsat with h | h | h
    · constructor
      · exact h.le
      · constructor <;> nlinarith [abs_nonneg x₀, abs_nonneg x₁, abs_nonneg x₂]
    · constructor
      · nlinarith [abs_nonneg x₀, abs_nonneg x₁, abs_nonneg x₂]
      · constructor
        · exact h.le
        · nlinarith [abs_nonneg x₀, abs_nonneg x₁, abs_nonneg x₂]
    · constructor
      · nlinarith [abs_nonneg x₀, abs_nonneg x₁, abs_nonneg x₂]
      · constructor
        · nlinarith [abs_nonneg x₀, abs_nonneg x₁, abs_nonneg x₂]
        · exact h.le
  constructor
  · refine ⟨?_, hsum, ?_, ?_, ?_⟩
    · intro j
      fin_cases j
      · change 0 ≤ threeBoundaryVector t 0
        rw [← hs0]; positivity
      · change 0 ≤ threeBoundaryVector t 1
        rw [← hs1]; positivity
      · change 0 ≤ threeBoundaryVector t 2
        rw [← hs2]; positivity
    · simpa [hr0, hr1, hr2] using htri.1
    · simpa [hr0, hr1, hr2] using htri.2.1
    · simpa [hr0, hr1, hr2] using htri.2.2
  · simpa [hr0, hr1, hr2] using hsat

/-- Conversely, every saturation point of the exact reachable-region
inequalities lies on the manuscript's analytic curve. -/
theorem triangleBoundary_exists_threeBoundaryVector
    (p : Fin 3 → ℝ) (hbd : ThreePowerTriangleBoundary p) :
    ∃ t : ℝ, threeBoundaryVector t = p := by
  let r₀ := threePowerRadius p 0
  let r₁ := threePowerRadius p 1
  let r₂ := threePowerRadius p 2
  have hp0 : 0 ≤ p 0 := hbd.1.1 0
  have hp1 : 0 ≤ p 1 := hbd.1.1 1
  have hp2 : 0 ≤ p 2 := hbd.1.1 2
  have hrsq0 : r₀ ^ 2 = p 0 := by
    dsimp [r₀, threePowerRadius]
    exact Real.sq_sqrt hp0
  have hrsq1 : r₁ ^ 2 = p 1 := by
    dsimp [r₁, threePowerRadius]
    exact Real.sq_sqrt hp1
  have hrsq2 : r₂ ^ 2 = p 2 := by
    dsimp [r₂, threePowerRadius]
    exact Real.sq_sqrt hp2
  have hsum : p 0 + p 1 + p 2 = 1 := by
    simpa [Fin.sum_univ_three] using hbd.1.2.1
  rcases hbd.2 with h | h | h
  · apply threeBoundaryVector_exists_of_signed_roots p r₀ r₁ (-r₂)
      hsum hrsq0 hrsq1 (by simpa using hrsq2)
    dsimp [r₀, r₁, r₂] at h ⊢
    nlinarith
  · apply threeBoundaryVector_exists_of_signed_roots p (-r₀) r₁ r₂
      hsum (by simpa using hrsq0) hrsq1 hrsq2
    dsimp [r₀, r₁, r₂] at h ⊢
    nlinarith
  · apply threeBoundaryVector_exists_of_signed_roots p r₀ (-r₁) r₂
      hsum hrsq0 (by simpa using hrsq1) hrsq2
    dsimp [r₀, r₁, r₂] at h ⊢
    nlinarith

/-- Global completeness of the analytic curve: it is exactly the full
saturation boundary of the exact three-port reachable region. -/
theorem triangleBoundary_iff_threeBoundaryVector (p : Fin 3 → ℝ) :
    ThreePowerTriangleBoundary p ↔
      ∃ t : ℝ, threeBoundaryVector t = p := by
  constructor
  · exact triangleBoundary_exists_threeBoundaryVector p
  · rintro ⟨t, rfl⟩
    exact threeBoundaryVector_mem_triangleBoundary t

end TFLNRouter
