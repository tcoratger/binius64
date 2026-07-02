// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use binius_field::{Field, PackedField};
use binius_ip::sumcheck::RoundCoeffs;
use binius_math::AsSlicesMut;

use super::{
	batch_quadratic_mle::{BatchQuadraticMleCheckProver, QuadraticComposition},
	common::{MleCheckProver, SumcheckProver},
	error::Error,
};

/// MLE-check prover for a quadratic composition of `N` multilinear polynomials.
///
/// It proves a claim about the multilinear extension of the composite `C(M_1, ..., M_N)`.
/// The claim is reduced to a claim on each multilinear's evaluation at the challenge point.
/// Round polynomials are degree 2, interpolated with the Karatsuba optimization.
///
/// This is the single-claim specialization of [`BatchQuadraticMleCheckProver`].
/// It runs one claim through the same chunked, wide-accumulated round evaluation.
pub struct QuadraticMleCheckProver<P: PackedField, Composition, InfinityComposition, const N: usize>(
	BatchQuadraticMleCheckProver<
		P,
		SingleComposition<Composition>,
		SingleComposition<InfinityComposition>,
		N,
		1,
	>,
);

impl<F, P, Composition, InfinityComposition, const N: usize>
	QuadraticMleCheckProver<P, Composition, InfinityComposition, N>
where
	F: Field,
	P: PackedField<Scalar = F>,
	Composition: Fn([P; N]) -> P + Sync,
	InfinityComposition: Fn([P; N]) -> P + Sync,
{
	/// Creates a new prover for verifying a quadratic composite polynomial evaluation.
	///
	/// # Arguments
	///
	/// * `multilinears` - the `N` input multilinears, all with the same number of variables.
	/// * `composition` - evaluates `C` on one packed row of the inputs, e.g. `|[a, b]| a * b`.
	/// * `infinity_composition` - the highest-degree part of `C`, e.g. `a * b` for `a * b - c`.
	/// * `eval_point` - the point at which the composite's multilinear extension is claimed.
	/// * `eval_claim` - the claimed value of that extension at `eval_point`.
	///
	/// # Errors
	///
	/// Returns [`Error::MultilinearSizeMismatch`] if a multilinear's variable count differs from
	/// the length of `eval_point`.
	pub fn new(
		multilinears: impl AsSlicesMut<P, N> + Send + 'static,
		composition: Composition,
		infinity_composition: InfinityComposition,
		eval_point: Vec<F>,
		eval_claim: F,
	) -> Result<Self, Error> {
		// Reuse the batch prover with a single claim.
		let inner = BatchQuadraticMleCheckProver::new(
			multilinears,
			SingleComposition(composition),
			SingleComposition(infinity_composition),
			eval_point,
			[eval_claim],
		)?;
		Ok(Self(inner))
	}
}

impl<F, P, Composition, InfinityComposition, const N: usize> SumcheckProver<F>
	for QuadraticMleCheckProver<P, Composition, InfinityComposition, N>
where
	F: Field,
	P: PackedField<Scalar = F>,
	Composition: Fn([P; N]) -> P + Sync,
	InfinityComposition: Fn([P; N]) -> P + Sync,
{
	fn n_vars(&self) -> usize {
		// Variables left to bind, tracked by the underlying batch prover.
		self.0.n_vars()
	}

	fn n_claims(&self) -> usize {
		// A single claim.
		self.0.n_claims()
	}

	fn round_claim(&self) -> Vec<F> {
		// The one claim carried into this round.
		self.0.round_claim()
	}

	fn execute(&mut self) -> Result<Vec<RoundCoeffs<F>>, Error> {
		// The single round polynomial for the variable being bound.
		self.0.execute()
	}

	fn fold(&mut self, challenge: F) -> Result<(), Error> {
		// Bind the current variable to the verifier challenge.
		self.0.fold(challenge)
	}

	fn finish(self) -> Result<Vec<F>, Error> {
		// The N multilinear evaluations at the challenge point.
		self.0.finish()
	}
}

impl<F, P, Composition, InfinityComposition, const N: usize> MleCheckProver<F>
	for QuadraticMleCheckProver<P, Composition, InfinityComposition, N>
where
	F: Field,
	P: PackedField<Scalar = F>,
	Composition: Fn([P; N]) -> P + Sync,
	InfinityComposition: Fn([P; N]) -> P + Sync,
{
	fn eval_point(&self) -> &[F] {
		// The remaining coordinates of the evaluation point.
		self.0.eval_point()
	}
}

/// Adapts a single-output composition into the [`QuadraticComposition`] shape with `M = 1`.
struct SingleComposition<C>(C);

impl<P, C, const N: usize> QuadraticComposition<P, N, 1> for SingleComposition<C>
where
	P: PackedField,
	C: Fn([P; N]) -> P,
{
	fn evaluate(&self, inputs: [P; N], outputs: &mut [P; 1]) {
		// The single composite value is the only output.
		outputs[0] = (self.0)(inputs);
	}
}

#[cfg(test)]
mod tests {
	use std::{array, iter};

	use binius_field::{arch::OptimalPackedB128, field::FieldOps};
	use binius_ip::mlecheck;
	use binius_math::{
		FieldBuffer,
		multilinear::evaluate::evaluate,
		test_utils::{random_field_buffer, random_scalars},
	};
	use binius_transcript::{ProverTranscript, fiat_shamir::HasherChallenger};

	type StdChallenger = HasherChallenger<sha2::Sha256>;
	use itertools::{self, Itertools};
	use rand::prelude::*;

	use super::*;
	use crate::sumcheck::prove_single_mlecheck;

	fn test_mlecheck_prove_verify<F, P, Composition, InfinityComposition, const N: usize>(
		prover: QuadraticMleCheckProver<P, Composition, InfinityComposition, N>,
		composition: Composition,
		eval_claim: F,
		eval_point: &[F],
		multilinears: Vec<FieldBuffer<P>>,
	) where
		F: Field,
		P: PackedField<Scalar = F>,
		Composition: Fn([P; N]) -> P + Sync,
		InfinityComposition: Fn([P; N]) -> P + Sync,
	{
		// Run the proving protocol
		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let output = prove_single_mlecheck(prover, &mut prover_transcript).unwrap();

		// Write the multilinear evaluations to the transcript
		prover_transcript
			.message()
			.write_slice(&output.multilinear_evals);

		// Convert to verifier transcript and run verification
		let mut verifier_transcript = prover_transcript.into_verifier();
		let sumcheck_output = mlecheck::verify(
			eval_point,
			2, // degree 2 for composite polynomials
			eval_claim,
			&mut verifier_transcript,
		)
		.unwrap();

		let mut reduced_eval_point = sumcheck_output.challenges.clone();
		reduced_eval_point.reverse();

		// Read the multilinear evaluations from the transcript
		let multilinear_evals: Vec<F> = verifier_transcript.message().read_vec(N).unwrap();

		// Check that the composition of the evaluations equals the reduced evaluation
		let evals_packed: [P; N] = array::from_fn(|i| P::broadcast(multilinear_evals[i]));
		let composition_result = composition(evals_packed);
		assert_eq!(
			composition_result.iter().next().unwrap(),
			sumcheck_output.eval,
			"Composition of multilinear evaluations should equal the reduced evaluation"
		);

		// Check that the original multilinears evaluate to the claimed values at the challenge
		// point
		for (multilinear, claimed_eval) in iter::zip(&multilinears, multilinear_evals) {
			let actual_eval = evaluate(multilinear, &reduced_eval_point);
			assert_eq!(actual_eval, claimed_eval);
		}

		// Also verify the challenges match what the prover saw
		assert_eq!(
			output.challenges, sumcheck_output.challenges,
			"Prover and verifier challenges should match"
		);
	}

	fn test_quadratic_mlecheck_prove_verify<F, P, const N: usize>(
		composition: impl Fn([P; N]) -> P + Clone + Sync,
		infinity_composition: impl Fn([P; N]) -> P + Clone + Sync,
	) where
		F: Field,
		P: PackedField<Scalar = F>,
	{
		let n_vars = 8;
		let mut rng = StdRng::seed_from_u64(0);

		// Generate random multilinear polynomials
		let multilinears: [_; N] = array::from_fn(|_| random_field_buffer::<P>(&mut rng, n_vars));

		// Compute product multilinear
		let composite_vals = (0..1 << n_vars.saturating_sub(P::LOG_WIDTH))
			.map(|i| {
				let vals = array::from_fn(|j| multilinears[j].as_ref()[i]);
				composition(vals)
			})
			.collect_vec();
		let composite_vals = FieldBuffer::new(n_vars, composite_vals);

		let eval_point = random_scalars::<F>(&mut rng, n_vars);
		let eval_claim = evaluate(&composite_vals, &eval_point);

		// Create the prover
		let mlecheck_prover = QuadraticMleCheckProver::new(
			multilinears.clone(),
			composition.clone(),
			infinity_composition,
			eval_point.clone(),
			eval_claim,
		)
		.unwrap();

		test_mlecheck_prove_verify(
			mlecheck_prover,
			composition,
			eval_claim,
			&eval_point,
			multilinears.to_vec(),
		);
	}

	// Test that quadratic MLE-check handles multilinears. It's not the most efficient strategy
	// for a multilinear MLE-check, but it's a good edge case.
	#[test]
	fn test_linear_mlecheck() {
		test_quadratic_mlecheck_prove_verify::<_, OptimalPackedB128, 2>(
			|[a, b]| a + b,
			|[_a, _b]| OptimalPackedB128::zero(), // coefficient on the quadratic term is 0
		);
	}

	#[test]
	fn test_bivariate_product_mlecheck() {
		test_quadratic_mlecheck_prove_verify::<_, OptimalPackedB128, 2>(
			|[a, b]| a * b,
			|[a, b]| a * b,
		);
	}

	#[test]
	fn test_mul_gate_mlecheck() {
		test_quadratic_mlecheck_prove_verify::<_, OptimalPackedB128, 3>(
			|[a, b, c]| a * b - c,
			|[a, b, _c]| a * b,
		);
	}

	#[test]
	fn test_4_variate_composition_mlecheck() {
		test_quadratic_mlecheck_prove_verify::<_, OptimalPackedB128, 4>(
			|[a, b, c, d]| (a + b) * (c + d),
			|[a, b, c, d]| (a + b) * (c + d),
		);
	}

	// `round_claim` must return the same value before and after `execute()`: the MLE-check claim
	// recovered from the round coefficients via `lerp_over_endpoints` must equal the stored claim.
	#[test]
	fn test_round_claim_lerp_recovery() {
		use binius_field::{Random, arch::OptimalB128};
		type P = OptimalPackedB128;
		type F = OptimalB128;

		let n_vars = 8;
		let mut rng = StdRng::seed_from_u64(0);

		let multilinears: [_; 2] = array::from_fn(|_| random_field_buffer::<P>(&mut rng, n_vars));
		let composition = |[a, b]: [P; 2]| a * b;
		let composite_vals = (0..1 << n_vars.saturating_sub(P::LOG_WIDTH))
			.map(|i| composition(array::from_fn(|j| multilinears[j].as_ref()[i])))
			.collect_vec();
		let composite_vals = FieldBuffer::new(n_vars, composite_vals);
		let eval_point = random_scalars::<F>(&mut rng, n_vars);
		let eval_claim = evaluate(&composite_vals, &eval_point);

		let mut prover = QuadraticMleCheckProver::new(
			multilinears,
			composition,
			composition,
			eval_point,
			eval_claim,
		)
		.unwrap();

		let mut expected = vec![eval_claim];
		for _ in 0..n_vars {
			assert_eq!(prover.round_claim(), expected, "claim before execute");
			let round = prover.execute().unwrap();
			assert_eq!(prover.round_claim(), expected, "claim recovered from coeffs");
			let challenge = F::random(&mut rng);
			expected = round.iter().map(|r| r.evaluate(challenge)).collect();
			prover.fold(challenge).unwrap();
		}
	}
}
