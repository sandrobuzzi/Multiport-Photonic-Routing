import TFLNRouter

/-! # Sharp balanced two-output bound for ideal three-port routers -/

namespace TFLNRouter

open Complex
open ComplexConjugate

theorem balanced_three_field_bound
    (a b c : ℂ)
    (hbal : ‖a‖ ^ 2 = ‖b‖ ^ 2)
    (hsum : ‖a‖ ^ 2 + ‖b‖ ^ 2 + ‖c‖ ^ 2 = 1)
    (hcorr : a * conj b + b * conj c + c * conj a = 0) :
    (1 : ℝ) / 9 ≤ ‖c‖ ^ 2 ∧ ‖a‖ ^ 2 ≤ (4 : ℝ) / 9 := by
  have habnorm : ‖a‖ = ‖b‖ := by
    nlinarith [norm_nonneg a, norm_nonneg b]
  have habeq : a * conj b = -(b * conj c + c * conj a) := by
    linear_combination hcorr
  have htriangle : ‖a‖ * ‖b‖ ≤ ‖b‖ * ‖c‖ + ‖c‖ * ‖a‖ := by
    calc
      ‖a‖ * ‖b‖ = ‖a * conj b‖ := by simp [norm_mul]
      _ = ‖b * conj c + c * conj a‖ := by rw [habeq, norm_neg]
      _ ≤ ‖b * conj c‖ + ‖c * conj a‖ := norm_add_le _ _
      _ = ‖b‖ * ‖c‖ + ‖c‖ * ‖a‖ := by simp [norm_mul]
  rw [← habnorm] at htriangle
  have hac : ‖a‖ ^ 2 ≤ 4 * ‖c‖ ^ 2 := by
    by_cases ha : ‖a‖ = 0
    · simp [ha]
    · have hapos : 0 < ‖a‖ := lt_of_le_of_ne (norm_nonneg a) (Ne.symm ha)
      have hlin : ‖a‖ ≤ 2 * ‖c‖ := by nlinarith
      nlinarith [sq_nonneg (‖a‖ - 2 * ‖c‖)]
  constructor
  · nlinarith
  · nlinarith

noncomputable def threePhaseOutput (w : Circle)
    (phase : Fin 3 → Circle) : Fin 3 → ℂ := fun q ↦
  (3 : ℂ)⁻¹ * ∑ j, (phase j : ℂ) * (FourierPhase3 w j q : ℂ)

lemma three_root_cube (w : Circle)
    (hw : (w : ℂ)^2 + (w : ℂ) + 1 = 0) :
    (w : ℂ)^3 = 1 := by
  linear_combination (w : ℂ) * hw - hw

lemma three_root_conj (w : Circle)
    (hw : (w : ℂ)^2 + (w : ℂ) + 1 = 0) :
    conj (w : ℂ) = (w : ℂ)^2 := by
  rw [← Circle.coe_inv_eq_conj]
  change (w : ℂ)⁻¹ = (w : ℂ)^2
  calc
    (w : ℂ)⁻¹ = (w : ℂ)⁻¹ * (w : ℂ)^3 := by
      rw [three_root_cube w hw, mul_one]
    _ = (w : ℂ)^2 := by field_simp [Circle.coe_ne_zero w] <;> ring

lemma threePhaseOutput_zero (w : Circle) (phase : Fin 3 → Circle) :
    threePhaseOutput w phase 0 =
      ((phase 0 : ℂ) + phase 1 + phase 2) / 3 := by
  simp [threePhaseOutput, FourierPhase3, Fin.sum_univ_three]
  norm_cast
  ring

lemma threePhaseOutput_one (w : Circle) (phase : Fin 3 → Circle) :
    threePhaseOutput w phase 1 =
      ((phase 0 : ℂ) + (w : ℂ) * phase 1 + (w : ℂ)^2 * phase 2) / 3 := by
  simp [threePhaseOutput, FourierPhase3, Fin.sum_univ_three]
  rw [show ((w ^ 2 : Circle) : ℂ) = (w : ℂ)^2 by norm_cast]
  ring

lemma threePhaseOutput_two (w : Circle)
    (hw : (w : ℂ)^2 + (w : ℂ) + 1 = 0) (phase : Fin 3 → Circle) :
    threePhaseOutput w phase 2 =
      ((phase 0 : ℂ) + (w : ℂ)^2 * phase 1 + (w : ℂ) * phase 2) / 3 := by
  simp [threePhaseOutput, FourierPhase3, Fin.sum_univ_three]
  rw [show ((w ^ 2 : Circle) : ℂ) = (w : ℂ)^2 by norm_cast,
    show ((w ^ 4 : Circle) : ℂ) = (w : ℂ)^4 by norm_cast]
  rw [show (w : ℂ)^4 = w by
    calc
      (w : ℂ)^4 = (w : ℂ)^3 * (w : ℂ) := by ring
      _ = (w : ℂ) := by rw [three_root_cube w hw, one_mul]]
  ring

lemma threePhaseOutput_correlation (w : Circle)
    (hw : (w : ℂ)^2 + (w : ℂ) + 1 = 0) (phase : Fin 3 → Circle) :
    threePhaseOutput w phase 0 * conj (threePhaseOutput w phase 1) +
      threePhaseOutput w phase 1 * conj (threePhaseOutput w phase 2) +
      threePhaseOutput w phase 2 * conj (threePhaseOutput w phase 0) = 0 := by
  rw [threePhaseOutput_zero, threePhaseOutput_one,
    threePhaseOutput_two w hw]
  have hwc := three_root_conj w hw
  have hw3 := three_root_cube w hw
  have hw2 : (w : ℂ)^2 = -w - 1 := by linear_combination hw
  have hx : (phase 0 : ℂ) * conj (phase 0 : ℂ) = 1 := by
    rw [← Circle.coe_inv_eq_conj]
    norm_cast
    simp
  have hy : (phase 1 : ℂ) * conj (phase 1 : ℂ) = 1 := by
    rw [← Circle.coe_inv_eq_conj]
    norm_cast
    simp
  have hz : (phase 2 : ℂ) * conj (phase 2 : ℂ) = 1 := by
    rw [← Circle.coe_inv_eq_conj]
    norm_cast
    simp
  have hx' : conj (phase 0 : ℂ) * (phase 0 : ℂ) = 1 := by rw [mul_comm, hx]
  have hy' : conj (phase 1 : ℂ) * (phase 1 : ℂ) = 1 := by rw [mul_comm, hy]
  have hz' : conj (phase 2 : ℂ) * (phase 2 : ℂ) = 1 := by rw [mul_comm, hz]
  simp only [map_div₀, map_add, map_mul, map_ofNat, hwc, map_pow,
    conj_conj]
  field_simp
  rw [hw2]
  ring_nf
  rw [hw3, hw2]
  ring_nf
  simp only [hx, hy, hz, hx', hy', hz']
  linear_combination 3 * (w : ℂ) * (hz' - hy')

lemma threePhaseOutput_power_sum (w : Circle)
    (hw : (w : ℂ)^2 + (w : ℂ) + 1 = 0) (phase : Fin 3 → Circle) :
    ‖threePhaseOutput w phase 0‖ ^ 2 +
      ‖threePhaseOutput w phase 1‖ ^ 2 +
      ‖threePhaseOutput w phase 2‖ ^ 2 = 1 := by
  rw [Complex.sq_norm, Complex.sq_norm, Complex.sq_norm]
  rw [← Complex.ofReal_inj]
  simp only [ofReal_add, Complex.normSq_eq_conj_mul_self]
  rw [threePhaseOutput_zero, threePhaseOutput_one,
    threePhaseOutput_two w hw]
  have hwc := three_root_conj w hw
  have hw3 := three_root_cube w hw
  have hw2 : (w : ℂ)^2 = -w - 1 := by linear_combination hw
  have hx : (phase 0 : ℂ) * conj (phase 0 : ℂ) = 1 := by
    rw [← Circle.coe_inv_eq_conj]
    norm_cast
    simp
  have hy : (phase 1 : ℂ) * conj (phase 1 : ℂ) = 1 := by
    rw [← Circle.coe_inv_eq_conj]
    norm_cast
    simp
  have hz : (phase 2 : ℂ) * conj (phase 2 : ℂ) = 1 := by
    rw [← Circle.coe_inv_eq_conj]
    norm_cast
    simp
  have hx' : conj (phase 0 : ℂ) * (phase 0 : ℂ) = 1 := by rw [mul_comm, hx]
  have hy' : conj (phase 1 : ℂ) * (phase 1 : ℂ) = 1 := by rw [mul_comm, hy]
  have hz' : conj (phase 2 : ℂ) * (phase 2 : ℂ) = 1 := by rw [mul_comm, hz]
  simp only [map_div₀, map_add, map_mul, map_ofNat, hwc, map_pow,
    conj_conj]
  field_simp
  rw [hw2]
  ring_nf
  rw [hw3, hw2]
  ring_nf
  simp only [hx, hy, hz, hx', hy', hz']
  ring

theorem threePhaseOutput_balanced_leakage (w : Circle)
    (hw : (w : ℂ)^2 + (w : ℂ) + 1 = 0) (phase : Fin 3 → Circle)
    (hbalanced : fieldPower (threePhaseOutput w phase) 0 =
      fieldPower (threePhaseOutput w phase) 1) :
    (1 : ℝ) / 9 ≤ fieldPower (threePhaseOutput w phase) 2 ∧
      fieldPower (threePhaseOutput w phase) 0 ≤ (4 : ℝ) / 9 := by
  apply balanced_three_field_bound
  · exact hbalanced
  · exact threePhaseOutput_power_sum w hw phase
  · exact threePhaseOutput_correlation w hw phase

noncomputable def threeEffectivePhase (w : Circle)
    (fixed control : Fin 3 → Circle) (n : Fin 3) : Fin 3 → Circle := fun j ↦
  generalArmPhase fixed control j * FourierPhase3 w j n

lemma threeDFT_generalOutput_eq_threePhaseOutput (w : Circle)
    (fixed control : Fin 3 → Circle) (n q : Fin 3) :
    generalOutput (threeDFT w) fixed control n q =
      threePhaseOutput w (threeEffectivePhase w fixed control n) q := by
  simp only [generalOutput, threeDFT, threePhaseOutput,
    threeEffectivePhase, Circle.coe_mul]
  have hs : (Real.sqrt 3 : ℂ) ≠ 0 := by
    exact_mod_cast ne_of_gt (Real.sqrt_pos.2 (by norm_num))
  have hcoef : (Real.sqrt 3 : ℂ)⁻¹ * (Real.sqrt 3 : ℂ)⁻¹ = (3 : ℂ)⁻¹ := by
    field_simp [hs]
    exact_mod_cast (Real.mul_self_sqrt (by norm_num : (0 : ℝ) ≤ 3)).symm
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro j _
  calc
    (Real.sqrt 3 : ℂ)⁻¹ * (FourierPhase3 w j q : ℂ) *
        (generalArmPhase fixed control j : ℂ) *
        ((Real.sqrt 3 : ℂ)⁻¹ * (FourierPhase3 w j n : ℂ)) =
      ((Real.sqrt 3 : ℂ)⁻¹ * (Real.sqrt 3 : ℂ)⁻¹) *
        ((generalArmPhase fixed control j : ℂ) *
          (FourierPhase3 w j n : ℂ) * (FourierPhase3 w j q : ℂ)) := by ring
    _ = (3 : ℂ)⁻¹ * ((generalArmPhase fixed control j : ℂ) *
          (FourierPhase3 w j n : ℂ) * (FourierPhase3 w j q : ℂ)) := by rw [hcoef]

theorem threeDFT_balanced_leakage (w : Circle)
    (hw : (w : ℂ)^2 + (w : ℂ) + 1 = 0)
    (fixed control : Fin 3 → Circle) (n : Fin 3)
    (hbalanced : fieldPower
        (fun q ↦ generalOutput (threeDFT w) fixed control n q) 0 =
      fieldPower
        (fun q ↦ generalOutput (threeDFT w) fixed control n q) 1) :
    (1 : ℝ) / 9 ≤ fieldPower
        (fun q ↦ generalOutput (threeDFT w) fixed control n q) 2 ∧
      fieldPower
        (fun q ↦ generalOutput (threeDFT w) fixed control n q) 0 ≤
          (4 : ℝ) / 9 := by
  simp_rw [threeDFT_generalOutput_eq_threePhaseOutput] at hbalanced ⊢
  exact threePhaseOutput_balanced_leakage w hw _ hbalanced

/-- A phase setting that attains the balanced two-output optimum. -/
noncomputable def circleNegOne : Circle :=
  ⟨-1, mem_sphere_zero_iff_norm.2 (by norm_num)⟩

noncomputable def balancedThreePhase (w : Circle) : Fin 3 → Circle :=
  ![1, w, circleNegOne * w^2]

lemma balancedThreePhase_output_zero (w : Circle)
    (hw : (w : ℂ)^2 + (w : ℂ) + 1 = 0) :
    threePhaseOutput w (balancedThreePhase w) 0 =
      -(2 : ℂ) / 3 * (w : ℂ)^2 := by
  rw [threePhaseOutput_zero]
  simp [balancedThreePhase, circleNegOne]
  rw [show ((w ^ 2 : Circle) : ℂ) = (w : ℂ)^2 by norm_cast]
  linear_combination (1 / 3 : ℂ) * hw

lemma balancedThreePhase_output_one (w : Circle)
    (hw : (w : ℂ)^2 + (w : ℂ) + 1 = 0) :
    threePhaseOutput w (balancedThreePhase w) 1 =
      -(2 : ℂ) / 3 * (w : ℂ) := by
  rw [threePhaseOutput_one]
  simp [balancedThreePhase, circleNegOne]
  rw [show ((w ^ 2 : Circle) : ℂ) = (w : ℂ)^2 by norm_cast]
  have hw4 : (w : ℂ)^4 = w := by
    calc
      (w : ℂ)^4 = (w : ℂ)^3 * (w : ℂ) := by ring
      _ = (w : ℂ) := by rw [three_root_cube w hw, one_mul]
  ring_nf
  rw [hw4]
  linear_combination (1 / 3 : ℂ) * hw

lemma balancedThreePhase_output_two (w : Circle)
    (hw : (w : ℂ)^2 + (w : ℂ) + 1 = 0) :
    threePhaseOutput w (balancedThreePhase w) 2 = 1 / 3 := by
  rw [threePhaseOutput_two w hw]
  simp [balancedThreePhase, circleNegOne]
  rw [show ((w ^ 2 : Circle) : ℂ) = (w : ℂ)^2 by norm_cast]
  ring

theorem balancedThreePhase_powers (w : Circle)
    (hw : (w : ℂ)^2 + (w : ℂ) + 1 = 0) :
    fieldPower (threePhaseOutput w (balancedThreePhase w)) =
      ![(4 : ℝ) / 9, (4 : ℝ) / 9, (1 : ℝ) / 9] := by
  funext q
  fin_cases q
  · change fieldPower (threePhaseOutput w (balancedThreePhase w)) 0 =
      (4 : ℝ) / 9
    rw [fieldPower, balancedThreePhase_output_zero w hw, norm_mul,
      norm_pow, Circle.norm_coe]
    norm_num
  · change fieldPower (threePhaseOutput w (balancedThreePhase w)) 1 =
      (4 : ℝ) / 9
    rw [fieldPower, balancedThreePhase_output_one w hw, norm_mul,
      Circle.norm_coe]
    norm_num
  · change fieldPower (threePhaseOutput w (balancedThreePhase w)) 2 =
      (1 : ℝ) / 9
    rw [fieldPower, balancedThreePhase_output_two w hw]
    norm_num

/-- Reflector phases that synthesize an arbitrary target phase vector after
the two fixed traversals and the input-column phase. -/
noncomputable def controlForThreePhase (w : Circle)
    (fixed target : Fin 3 → Circle) (n : Fin 3) : Fin 3 → Circle := fun j ↦
  (fixed j)⁻¹ * (fixed j)⁻¹ * target j * (FourierPhase3 w j n)⁻¹

lemma threeEffectivePhase_controlFor (w : Circle)
    (fixed target : Fin 3 → Circle) (n j : Fin 3) :
    threeEffectivePhase w fixed
      (controlForThreePhase w fixed target n) n j = target j := by
  simp only [threeEffectivePhase, controlForThreePhase,
    generalArmPhase]
  calc
    fixed j * ((fixed j)⁻¹ * (fixed j)⁻¹ * target j *
          (FourierPhase3 w j n)⁻¹) * fixed j * FourierPhase3 w j n =
        ((fixed j)⁻¹ * fixed j) * ((fixed j)⁻¹ * fixed j) *
          ((FourierPhase3 w j n)⁻¹ * FourierPhase3 w j n) * target j := by
      ac_rfl
    _ = target j := by simp

theorem threeDFT_balanced_witness_reachable (w : Circle)
    (hw : (w : ℂ)^2 + (w : ℂ) + 1 = 0)
    (fixed : Fin 3 → Circle) (n : Fin 3) :
    ![(4 : ℝ) / 9, (4 : ℝ) / 9, (1 : ℝ) / 9] ∈
      reachablePowers (threeDFT w) fixed n := by
  let control := controlForThreePhase w fixed (balancedThreePhase w) n
  refine ⟨_, ⟨control, rfl⟩, ?_⟩
  rw [show (fun q ↦ generalOutput (threeDFT w) fixed control n q) =
      threePhaseOutput w (balancedThreePhase w) by
    funext q
    rw [threeDFT_generalOutput_eq_threePhaseOutput]
    congr 1
    funext j
    exact threeEffectivePhase_controlFor w fixed _ n j]
  exact balancedThreePhase_powers w hw

theorem threeDFT_reachable_balanced_leakage (w : Circle)
    (hw : (w : ℂ)^2 + (w : ℂ) + 1 = 0)
    (fixed : Fin 3 → Circle) (n : Fin 3) (p : Fin 3 → ℝ)
    (hp : p ∈ reachablePowers (threeDFT w) fixed n)
    (hbalanced : p 0 = p 1) :
    (1 : ℝ) / 9 ≤ p 2 ∧ p 0 ≤ (4 : ℝ) / 9 := by
  rcases hp with ⟨z, ⟨control, rfl⟩, rfl⟩
  exact threeDFT_balanced_leakage w hw fixed control n hbalanced

theorem threePort_balanced_leakage
    (M : Matrix (Fin 3) (Fin 3) ℂ)
    (hM : EqualSplit3 M) (hU : RowUnitary3 M)
    (fixed : Fin 3 → Circle) (n : Fin 3) (p : Fin 3 → ℝ)
    (hp : p ∈ reachablePowers M fixed n) (hbalanced : p 0 = p 1) :
    (1 : ℝ) / 9 ≤ p 2 ∧ p 0 ≤ (4 : ℝ) / 9 := by
  rcases threePort_reachable_region M hM hU fixed n with ⟨F, hF, hregion⟩
  rw [hregion] at hp
  rcases hF with ⟨w, hw, rfl⟩
  exact threeDFT_reachable_balanced_leakage w hw (fun _ ↦ 1) n p hp hbalanced

theorem threePort_balanced_witness_reachable
    (M : Matrix (Fin 3) (Fin 3) ℂ)
    (hM : EqualSplit3 M) (hU : RowUnitary3 M)
    (fixed : Fin 3 → Circle) (n : Fin 3) :
    ![(4 : ℝ) / 9, (4 : ℝ) / 9, (1 : ℝ) / 9] ∈
      reachablePowers M fixed n := by
  rcases threePort_reachable_region M hM hU fixed n with ⟨F, hF, hregion⟩
  rw [hregion]
  rcases hF with ⟨w, hw, rfl⟩
  exact threeDFT_balanced_witness_reachable w hw (fun _ ↦ 1) n

end TFLNRouter
