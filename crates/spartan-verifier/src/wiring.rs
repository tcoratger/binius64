// Copyright 2025 Irreducible Inc.

use std::{iter, rc::Rc};

use binius_field::{Field, field::FieldOps, util::FieldFn};
use binius_math::{multilinear::eq::eq_ind_partial_eval_scalars, univariate::evaluate_univariate};
use binius_spartan_frontend::constraint_system::{MulConstraint, WitnessIndex, WitnessSegment};

use crate::constraint_system::ConstraintSystemPadded;

/// Returns a closure that evaluates the wiring transparent polynomial for a specific segment.
///
/// The returned closure computes the expected evaluation of the wiring MLE for the given
/// segment, batched with lambda, given a challenge point from the BaseFold opening.
/// `r_x_tensor` is the eq-indicator partial evaluation at r_x; it is taken as a shared `Rc` so the
/// returned closure owns it (the opening is deferred to the channel's `finish()`, so the closure
/// must outlive the local `r_x_tensor`). The closure still borrows `constraint_system`, which is
/// long-lived.
pub fn eval_transparent<'a, G: Field, F: FieldOps + 'a>(
	constraint_system: &'a ConstraintSystemPadded<G>,
	segment: WitnessSegment,
	r_x_tensor: Rc<[F]>,
	lambda: F,
) -> binius_iop::channel::TransparentEvalFn<'a, F> {
	Box::new(move |r_y: &[F]| {
		evaluate_segment_wiring_mle(
			constraint_system.mul_constraints(),
			segment,
			lambda.clone(),
			&r_x_tensor,
			r_y,
		)
	})
}

/// Evaluates the wiring MLE for a specific segment at a point (r_x, r_y).
///
/// `r_x_tensor` is the eq-indicator partial evaluation at r_x, i.e.
/// `eq_ind_partial_eval_scalars(r_x)`. Accepting it as a parameter avoids redundant
/// computation when evaluating multiple segments with the same r_x.
pub fn evaluate_segment_wiring_mle<F: FieldOps>(
	mul_constraints: &[MulConstraint<WitnessIndex>],
	segment: WitnessSegment,
	lambda: F,
	r_x_tensor: &[F],
	r_y: &[F],
) -> F {
	let mut acc = [F::zero(), F::zero(), F::zero()];

	let r_y_tensor = eq_ind_partial_eval_scalars(r_y);
	for (r_x_tensor_i, MulConstraint { a, b, c }) in iter::zip(r_x_tensor, mul_constraints) {
		for (dst, operand) in iter::zip(&mut acc, [a, b, c]) {
			let r_y_tensor_sum = operand
				.wires()
				.iter()
				.flat_map(|index| {
					if index.segment == segment {
						Some(r_y_tensor[index.index as usize].clone())
					} else {
						None
					}
				})
				.sum::<F>();
			*dst += r_x_tensor_i.clone() * r_y_tensor_sum;
		}
	}

	evaluate_univariate(&acc, lambda)
}

/// Evaluates the public segment's contribution to the wiring MLE.
///
/// `r_x_tensor` is the eq-indicator partial evaluation at r_x.
pub fn evaluate_wiring_mle_public<F: FieldOps>(
	mul_constraints: &[MulConstraint<WitnessIndex>],
	public: &[F],
	lambda: F,
	r_x_tensor: &[F],
) -> F {
	let mut acc = [F::zero(), F::zero(), F::zero()];
	for (r_x_tensor_i, MulConstraint { a, b, c }) in iter::zip(r_x_tensor, mul_constraints) {
		for (dst, operand) in iter::zip(&mut acc, [a, b, c]) {
			let public_sum = operand
				.wires()
				.iter()
				.flat_map(|index| {
					if let WitnessSegment::Public = index.segment {
						Some(public[index.index as usize].clone())
					} else {
						None
					}
				})
				.sum::<F>();
			*dst += r_x_tensor_i.clone() * public_sum;
		}
	}

	evaluate_univariate(&acc, lambda)
}

/// The public-segment contribution to the operand evaluations, as a [`FieldFn`].
///
/// Input layout: `[ public scalars | lambda | r_x ]`.
/// The caller passes `r_x` directly; its eq-indicator tensor is expanded inside.
pub struct PublicWiringEvalFn<'a> {
	/// The MUL constraints whose public-segment wiring MLE is evaluated.
	mul_constraints: &'a [MulConstraint<WitnessIndex>],
	/// The number of leading input entries that are public scalars.
	public_len: usize,
}

impl<'a> PublicWiringEvalFn<'a> {
	/// Builds the function over the given MUL constraints and public-scalar count.
	pub fn new(mul_constraints: &'a [MulConstraint<WitnessIndex>], public_len: usize) -> Self {
		Self {
			mul_constraints,
			public_len,
		}
	}
}

impl<G: Field> FieldFn<G> for PublicWiringEvalFn<'_> {
	fn n_outputs(&self) -> usize {
		// The function produces the single public-segment evaluation.
		1
	}

	fn call<F: FieldOps<Scalar = G> + From<G>>(&self, inputs: &[F]) -> Vec<F> {
		// Split the flat input into its three logical parts.
		let public = &inputs[..self.public_len];
		let lambda = inputs[self.public_len].clone();
		let r_x = &inputs[self.public_len + 1..];

		// Expand r_x's eq-indicator tensor here.
		// Callers pass r_x itself, not its 2^|r_x| expansion.
		let r_x_tensor = eq_ind_partial_eval_scalars(r_x);

		vec![evaluate_wiring_mle_public(
			self.mul_constraints,
			public,
			lambda,
			&r_x_tensor,
		)]
	}
}
