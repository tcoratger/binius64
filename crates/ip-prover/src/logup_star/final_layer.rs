// Copyright 2026 The Binius Developers

//! The batched final layer of the table side.
//!
//! It fuses two reductions that both end in an evaluation of the pushforward `Y`:
//!
//! - the last fractional-addition GKR layer of the table circuit,
//! - the product sumcheck `<T, Y> = e`.
//!
//! Both split their leaf multilinears on the highest variable.
//! So both can share one `(m-1)`-variable sumcheck and one line-fold over that variable.
//! That collapses the two `Y` evaluations into a single evaluation point.
//!
//! This is the prover mirror of the verifier's final layer in [`binius_ip::logup_star`].

use binius_field::{Field, PackedField};
use binius_ip::{MultilinearEvalClaim, sumcheck::RoundCoeffs};
use binius_math::{
	FieldBuffer, line::extrapolate_line, multilinear::fold::fold_highest_var_inplace,
};
use either::Either;
use itertools::izip;

use super::error::Error;
use crate::{
	channel::IPProverChannel,
	sumcheck::{
		self, MleToSumCheckDecorator, batch::batch_prove, common::SumcheckProver, frac_add_mle,
	},
};

/// The state of a round-by-round sumcheck prover between its two phases.
///
/// The prover alternates: it emits the round polynomial, then folds it on a challenge.
/// So between calls it holds either the running sum claim or the last round's coefficients.
#[derive(Debug, Clone)]
enum RoundCoeffsOrSum<F: Field> {
	/// The coefficients of the round polynomial just emitted, awaiting a fold.
	Coeffs(RoundCoeffs<F>),
	/// The running sum claim before a round is executed.
	Sum(F),
}

/// The evaluation claims that the batched final layer reduces to.
pub struct FinalLayerOutput<F> {
	/// The `m`-coordinate point shared by the table and pushforward evaluation claims.
	pub table_eval_point: Vec<F>,
	/// The claimed evaluation of the table multilinear `T` at the point.
	pub table_eval_claim: F,
	/// The claimed evaluation of the pushforward multilinear `Y` at the point.
	pub pushforward_eval_claim: F,
}

/// A regular sumcheck prover for the leaf product `sum_{x'} (Y_0 * T_0 + Y_1 * T_1)(x')`.
///
/// This is the product claim `<T, Y> = e`, rewritten over the low `m-1` variables.
/// It is obtained by splitting `T` and `Y` on the highest variable.
///
/// The composite has degree 2 and carries no equality factor.
/// That is why it cannot ride along inside the eq-weighted fractional-addition prover.
///
/// The four multilinears are held as `[Y_0, Y_1, T_0, T_1]`.
/// So the final evaluations are exactly the leaf halves the verifier reads, in that order.
///
/// The table is small in the logUp* regime, so the round polynomial is summed at the scalar level.
/// Variables are bound from the highest index to the lowest, matching the other final-layer prover.
struct LeafProductProver<P: PackedField> {
	/// The leaf halves `[Y_0, Y_1, T_0, T_1]`, folded in place one variable per round.
	multilinears: [FieldBuffer<P>; 4],
	/// The sum claim before a round, or the round polynomial awaiting a fold.
	last_coeffs_or_sum: RoundCoeffsOrSum<P::Scalar>,
}

/// Prove the batched final layer of the table side.
///
/// Three sum claims are batched over the `m-1`-variable cube, then the highest variable is folded:
///
/// ```text
///     S_1 = sum_{x'} eq(x'; Z) * (Y_0 * D_1 + Y_1 * D_0)(x') = num_1(Z)
///     S_2 = sum_{x'} eq(x'; Z) * (D_0 * D_1)(x')             = den_1(Z)
///     S_3 = sum_{x'} (T_0 * Y_0 + T_1 * Y_1)(x')             = e
/// ```
///
/// The claims play two different roles:
///
/// - `S_1` and `S_2` are the layer-1 numerator and denominator of the fractional-addition circuit.
/// - They carry the equality factor `eq(x'; Z)`, so they run as a decorated MLE-check prover.
/// - `S_3` is the product claim `<T, Y> = e`, split on the highest variable, with no `eq` factor.
///
/// One batching coefficient combines the three round polynomials.
/// The verifier reconstructs the same batch and recomputes the public denominator halves itself.
/// So this routine writes only the leaf-half evaluations `[Y_0, Y_1, T_0, T_1]`.
///
/// # Arguments
///
/// * `eval_claim` - The product claim `e = <T, Y>`.
/// * `layer1` - The layer-1 numerator and denominator claims, sharing the point `Z`.
/// * `pushforward` - The pushforward `Y` over the `m`-variable cube.
/// * `table_denominator` - The table denominator `D = c - J` over the `m`-variable cube.
/// * `table` - The table `T` over the `m`-variable cube.
/// * `channel` - The prover channel.
pub fn prove_final_layer<F, P>(
	eval_claim: F,
	layer1: (MultilinearEvalClaim<F>, MultilinearEvalClaim<F>),
	pushforward: &FieldBuffer<P>,
	table_denominator: &FieldBuffer<P>,
	table: &FieldBuffer<P>,
	channel: &mut impl IPProverChannel<F>,
) -> Result<FinalLayerOutput<F>, Error>
where
	F: Field,
	P: PackedField<Scalar = F>,
{
	// Both layer-1 claims share the point Z; the numerator claim carries it.
	let (num_claim, den_claim) = layer1;
	debug_assert_eq!(num_claim.point, den_claim.point, "layer-1 claims must share the point");

	// Split each leaf multilinear on the highest variable into two halves over the low m-1 vars.
	//
	//     half 0: highest variable fixed to 0
	//     half 1: highest variable fixed to 1
	let [y_0, y_1] = split_halves(pushforward);
	let [d_0, d_1] = split_halves(table_denominator);
	let [t_0, t_1] = split_halves(table);

	// S_1, S_2: the fractional-addition numerator/denominator, weighted by eq(.; Z).
	//
	// The eq factor is kept implicit by the MLE-check prover.
	// The decorator factors it back in per round as a (X - alpha) multiplier.
	// That turns the MLE-check into a regular sumcheck the verifier's batch reconstructs.
	let frac_prover = frac_add_mle::new(
		[y_0.clone(), y_1.clone(), d_0, d_1],
		num_claim.point,
		[num_claim.eval, den_claim.eval],
	)?;
	let frac_prover = MleToSumCheckDecorator::new(frac_prover);

	// S_3: the product leaf check, split on the highest variable, carrying no eq factor.
	let product_prover = LeafProductProver::new([y_0, y_1, t_0, t_1], eval_claim)?;

	// Batch all three claims into one sumcheck.
	//
	// The flattened round-polynomial order is [num_1, den_1, e].
	// That matches the verifier's batched order [layer1_num, layer1_den, eval_claim].
	let output =
		batch_prove(vec![Either::Left(frac_prover), Either::Right(product_prover)], channel)?;

	// The product prover holds [Y_0, Y_1, T_0, T_1], so its evaluations are the four leaf halves.
	// The fractional prover's Y evaluations agree at the same point, so they need not be sent.
	let [y_0_eval, y_1_eval, t_0_eval, t_1_eval] = output.multilinear_evals[1]
		.as_slice()
		.try_into()
		.expect("the product prover evaluates four multilinears");
	channel.send_many(&[y_0_eval, y_1_eval, t_0_eval, t_1_eval]);

	// Fold the highest variable once to collapse each pair of halves into one evaluation.
	let r = channel.sample();
	let pushforward_eval_claim = extrapolate_line(y_0_eval, y_1_eval, r);
	let table_eval_claim = extrapolate_line(t_0_eval, t_1_eval, r);

	// batch_prove returns challenges low-to-high; the folded variable is the highest coordinate.
	let mut table_eval_point = output.challenges;
	table_eval_point.push(r);

	Ok(FinalLayerOutput {
		table_eval_point,
		table_eval_claim,
		pushforward_eval_claim,
	})
}

/// Split a multilinear on its highest variable into owned low and high halves.
///
/// The low half fixes the highest variable to 0, the high half to 1.
fn split_halves<P: PackedField>(buffer: &FieldBuffer<P>) -> [FieldBuffer<P>; 2] {
	// split_half_ref borrows the two halves; copy each into an owned buffer for the sub-provers.
	let (low, high) = buffer.split_half_ref();
	[
		FieldBuffer::new(low.log_len(), low.as_ref().into()),
		FieldBuffer::new(high.log_len(), high.as_ref().into()),
	]
}

impl<F: Field, P: PackedField<Scalar = F>> LeafProductProver<P> {
	fn new(multilinears: [FieldBuffer<P>; 4], sum: F) -> Result<Self, sumcheck::Error> {
		// All four halves live over the same number of variables, so they fold in lockstep.
		let n_vars = multilinears[0].log_len();
		if multilinears.iter().any(|m| m.log_len() != n_vars) {
			return Err(sumcheck::Error::MultilinearSizeMismatch);
		}
		Ok(Self {
			multilinears,
			last_coeffs_or_sum: RoundCoeffsOrSum::Sum(sum),
		})
	}
}

impl<F: Field, P: PackedField<Scalar = F>> SumcheckProver<F> for LeafProductProver<P> {
	fn n_vars(&self) -> usize {
		self.multilinears[0].log_len()
	}

	fn n_claims(&self) -> usize {
		1
	}

	fn round_claim(&self) -> Vec<F> {
		// Before a round the claim is the stored sum; after it, recover it as R(0) + R(1).
		let claim = match &self.last_coeffs_or_sum {
			RoundCoeffsOrSum::Sum(sum) => *sum,
			RoundCoeffsOrSum::Coeffs(coeffs) => coeffs.sum_over_endpoints(),
		};
		vec![claim]
	}

	fn execute(&mut self) -> Result<Vec<RoundCoeffs<F>>, sumcheck::Error> {
		// Execute consumes the running sum and produces this round's coefficients.
		let RoundCoeffsOrSum::Sum(last_sum) = &self.last_coeffs_or_sum else {
			return Err(sumcheck::Error::ExpectedFold);
		};
		let last_sum = *last_sum;

		// At least one variable must remain to sum over in this round.
		assert!(self.n_vars() > 0);

		// Split each multilinear on the highest variable.
		//
		//     lo = value at highest variable 0
		//     hi = value at highest variable 1
		let (y_0_lo, y_0_hi) = self.multilinears[0].split_half_ref();
		let (y_1_lo, y_1_hi) = self.multilinears[1].split_half_ref();
		let (t_0_lo, t_0_hi) = self.multilinears[2].split_half_ref();
		let (t_1_lo, t_1_hi) = self.multilinears[3].split_half_ref();

		// The round polynomial is R(X) = sum_{x'} (Y_0 T_0 + Y_1 T_1)(x', X), degree 2 in X.
		// Two evaluations pin it down together with the sum identity below.
		//
		//     R(1)   : the high halves, where every line takes its X=1 value
		//     R(inf) : the leading coefficient, where each line contributes M(0) + M(1)
		let (mut y_1, mut y_inf) = (F::ZERO, F::ZERO);
		for (y_0_0, y_0_1, y_1_0, y_1_1, t_0_0, t_0_1, t_1_0, t_1_1) in izip!(
			y_0_lo.iter_scalars(),
			y_0_hi.iter_scalars(),
			y_1_lo.iter_scalars(),
			y_1_hi.iter_scalars(),
			t_0_lo.iter_scalars(),
			t_0_hi.iter_scalars(),
			t_1_lo.iter_scalars(),
			t_1_hi.iter_scalars(),
		) {
			y_1 += y_0_1 * t_0_1 + y_1_1 * t_1_1;
			y_inf += (y_0_0 + y_0_1) * (t_0_0 + t_0_1) + (y_1_0 + y_1_1) * (t_1_0 + t_1_1);
		}

		// Interpolate c_2 X^2 + c_1 X + c_0 from the three evaluations.
		//
		//     R(0) = c_0 = sum - R(1)   (regular sumcheck identity sum = R(0) + R(1))
		//     R(inf) = c_2              (leading coefficient)
		//     R(1) = c_2 + c_1 + c_0
		let c_0 = last_sum - y_1;
		let c_2 = y_inf;
		let c_1 = y_1 - c_0 - c_2;
		let round_coeffs = RoundCoeffs(vec![c_0, c_1, c_2]);

		// Hand the coefficients to the upcoming fold.
		self.last_coeffs_or_sum = RoundCoeffsOrSum::Coeffs(round_coeffs.clone());
		Ok(vec![round_coeffs])
	}

	fn fold(&mut self, challenge: F) -> Result<(), sumcheck::Error> {
		// Fold consumes the round coefficients produced by execute.
		let RoundCoeffsOrSum::Coeffs(last_coeffs) = self.last_coeffs_or_sum.clone() else {
			return Err(sumcheck::Error::ExpectedExecute);
		};

		// Bind the highest variable of every multilinear to the verifier's challenge.
		for multilinear in &mut self.multilinears {
			fold_highest_var_inplace(multilinear, challenge);
		}

		// The round polynomial at the challenge is the next round's sum claim.
		let round_sum = last_coeffs.evaluate(challenge);
		self.last_coeffs_or_sum = RoundCoeffsOrSum::Sum(round_sum);
		Ok(())
	}

	fn finish(self) -> Result<Vec<F>, sumcheck::Error> {
		// Finishing is only valid once every variable has been folded away.
		if self.n_vars() > 0 {
			let error = match self.last_coeffs_or_sum {
				RoundCoeffsOrSum::Coeffs(_) => sumcheck::Error::ExpectedFold,
				RoundCoeffsOrSum::Sum(_) => sumcheck::Error::ExpectedExecute,
			};
			return Err(error);
		}

		// With no variables left, each multilinear is a single scalar: its evaluation at the point.
		Ok(self
			.multilinears
			.into_iter()
			.map(|multilinear| multilinear.get(0))
			.collect())
	}
}
