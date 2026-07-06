// Copyright 2025 Irreducible Inc.

//! Verifier glue for the M4 batched shift reduction (BitAnd only).
//!
//! The batched BitAnd reduction leaves a claim over the row index `row = kappa * n_and + x`.
//! The row splits into the local constraint index and the instance index:
//!
//! ```text
//!   eval_point = [ x .......... | kappa ...... ]
//!                  low c_and       high k coords
//! ```
//!
//! This splits the point, verifies the local reduction, and returns `r_kappa`.

use binius_core::constraint_system::ConstraintSystem;
use binius_field::AESTowerField8b as B8;
use binius_ip::channel::IPVerifierChannel;
use binius_math::BinarySubspace;
use binius_verifier::{
	Error,
	config::{B128, LOG_WORD_SIZE_BITS},
	protocols::{
		bitand::AndCheckOutput,
		shift::{OperatorData, verify_batch},
	},
};

/// Output of the M4 shift reduction verification.
///
/// The committed batch witness is evaluated at `(r_j, r_y, r_kappa)`.
/// The challenge point `[r_j, r_y]` comes from the shift reduction; `r_kappa` from the split.
#[derive(Debug)]
pub struct ShiftReductionOutput {
	/// The shift reduction challenge point `[r_j, r_y]` (bit index, then word index).
	pub challenges: Vec<B128>,
	/// The instance challenge, the high coordinates of the committed-witness evaluation point.
	pub r_kappa: Vec<B128>,
	/// The claimed evaluation of the committed batch witness at `(r_j, r_y, r_kappa)`.
	pub witness_eval: B128,
}

impl ShiftReductionOutput {
	/// Verifies the M4 batched shift reduction for the BitAnd operands.
	///
	/// # Arguments
	///
	/// - `constraint_system`: the per-instance constraint system, shared by every instance.
	/// - `log_instances`: base-2 logarithm of the instance count `K`.
	/// - `bitand`: the batched BitAnd reduction output, its point carrying the instance index high.
	/// - `channel`: the verifier channel.
	///
	/// # Errors
	///
	/// Returns an error if the shift reduction sumchecks or the monster identity fail.
	pub fn verify<C>(
		constraint_system: &ConstraintSystem,
		log_instances: usize,
		bitand: AndCheckOutput<B128>,
		channel: &mut C,
	) -> Result<Self, Error>
	where
		C: IPVerifierChannel<B128, Elem = B128>,
	{
		let AndCheckOutput {
			a_eval,
			b_eval,
			c_eval,
			z_challenge,
			eval_point,
		} = bitand;

		// Split the row-index point: low = local constraint index, high k coords = instance index.
		let c_and = eval_point.len() - log_instances;
		let (r_x, r_kappa) = eval_point.split_at(c_and);

		let bitand_data = OperatorData::new(r_x.to_vec(), [a_eval, b_eval, c_eval]);

		// The Lagrange basis subspace over the 64 word bits, matching the prover's shift kernels.
		let subspace = BinarySubspace::<B8>::with_dim(LOG_WORD_SIZE_BITS).isomorphic();

		let out = verify_batch(constraint_system, &bitand_data, &subspace, z_challenge, channel)?;

		Ok(Self {
			challenges: [out.r_j, out.r_y].concat(),
			r_kappa: r_kappa.to_vec(),
			witness_eval: out.witness_eval,
		})
	}
}
