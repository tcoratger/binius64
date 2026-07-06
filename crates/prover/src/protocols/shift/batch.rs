// Copyright 2025 Irreducible Inc.

//! Batched BitAnd shift reduction prover for the M4 data-parallel proof system.
//!
//! M4 commits `K = 2^k` instances of one circuit, stacked instance-major.
//! The batched BitAnd reduction runs over `K * n_and` rows.
//! Its operand claim splits into a local constraint part `r_x` and an instance part `r_kappa`.
//!
//! The instance index enters the claim only through the committed witness bit.
//! The shift structure is shared by every instance, so the instance factor folds onto the witness:
//!
//! ```text
//!   W_tilde(j, y) = sum_kappa eq(r_kappa, kappa) * bit_j(w_kappa[y]) = W_hat(j, y, r_kappa)
//! ```
//!
//! Each 0/1 bit becomes one field element, folded across the batch rows.
//! The single-instance two-phase reduction then runs unchanged on this virtual instance.
//! Its claim is `W_hat(r_j, r_y, r_kappa)`, with `r_kappa` on the high (instance) coordinates.
//!
//! This handles BitAnd only; IntMul is out of scope for the initial M4 batch.

use binius_core::word::Word;
use binius_field::{AESTowerField8b, BinaryField, PackedField};
use binius_ip::sumcheck::SumcheckOutput;
use binius_ip_prover::channel::IPProverChannel;
use binius_math::{FieldBuffer, multilinear::eq::eq_ind_partial_eval_scalars};
use binius_utils::{checked_arithmetics::log2_ceil_usize, rayon::prelude::*};
use binius_verifier::{
	config::{LOG_WORD_SIZE_BITS, WORD_SIZE_BITS},
	protocols::shift::{INTMUL_ARITY, SHIFT_VARIANT_COUNT},
};
use tracing::instrument;

use super::{
	key_collection::{KeyCollection, Operation},
	monster::{build_h_parts, build_monster_multilinear},
	phase_1::run_phase_1_sumcheck,
	phase_2::run_sumcheck,
	prove::{OperatorData, PreparedOperatorData},
};

/// Number of variables in the phase-1 `g` and `h` multilinears: bit index `j` plus shift amount
/// `s`.
const LOG_LEN: usize = LOG_WORD_SIZE_BITS + LOG_WORD_SIZE_BITS;

/// Proves the batched BitAnd shift reduction over `K = 2^k` instances of one circuit.
///
/// The reduction leaves one operand claim per operand `(a, b, c)`.
/// Each claim sits at a point `(r_zhat, r_x, r_kappa)`: bit challenge, local constraint, instance.
/// It reduces the claims to a single evaluation of the committed batch witness.
///
/// # Algorithm
///
/// - Fold the batch onto one field-valued virtual instance: `W_tilde(j, y) = sum_kappa eq(r_kappa,
///   kappa) * bit_j(w_kappa[y])`.
/// - Phase 1 over `(j, s)` reduces the batched operand claim to a point `(r_j, r_s)`.
/// - Phase 2 over the word index `y` reduces to the witness evaluation.
///
/// The `g` parts and the phase-2 fold are linear in the witness.
/// So building them from the folded witness is exact, with no per-instance error.
///
/// # Arguments
///
/// - `key_collection`: the per-instance shift structure, shared by every instance.
/// - `instances`: one committed-word slice per instance, in instance order.
/// - `r_kappa`: the instance challenge of length `k`, the high coordinates of the reduction point.
/// - `bitand_data`: the BitAnd operand claim, with bit challenge `r_zhat`, local `r_x`, and evals.
/// - `channel`: the prover channel.
///
/// # Returns
///
/// The reduced claim: challenges `[r_j, r_y]` and the witness evaluation `W_tilde(r_j, r_y)`.
/// The caller appends `r_kappa` to form the committed-witness point `(r_j, r_y, r_kappa)`.
///
/// # Panics
///
/// Panics if the instance count is not `2^{r_kappa.len()}`.
#[instrument(skip_all, name = "batch_shift_prove")]
pub fn prove_batch<F, P, Channel>(
	key_collection: &KeyCollection,
	instances: &[&[Word]],
	r_kappa: &[F],
	bitand_data: OperatorData<F>,
	channel: &mut Channel,
) -> SumcheckOutput<F>
where
	F: BinaryField + From<AESTowerField8b>,
	P: PackedField<Scalar = F>,
	Channel: IPProverChannel<F>,
{
	// The batch is a clean hypercube of instances: exactly 2^k of them.
	let n_instances = 1usize << r_kappa.len();
	assert_eq!(instances.len(), n_instances, "instance count must be 2^r_kappa.len()");

	// One key range per committed word of a single instance.
	let n_words = key_collection.key_ranges.len();

	// Sample the operand-batching coefficient for (a, b, c).
	// Only BitAnd is reduced here, so a single lambda suffices.
	let bitand_lambda = channel.sample();

	// The univariate bit challenge is shared by the fold and both phases.
	let r_zhat_prime = bitand_data.r_zhat_prime;
	let bitand_prep = PreparedOperatorData::new(bitand_data, bitand_lambda);

	// Fold the batch onto one virtual instance.
	//
	//     W_tilde[y][j] = sum_kappa eq(r_kappa, kappa) * bit_j(w_kappa[y])
	//
	// Each (bit, word) becomes one field element; the batch index is gone after this step.
	let w_tilde = fold_batch(instances, r_kappa, n_words);

	// Phase 1: the operand claim as a sum over (j, s) of g * h.
	//
	//     g_sigma(j, s) = sum_y W_tilde[y][j] * (constraint tensor weight of this word-key)
	//     h_sigma(j, s) = sum_i L_tilde_i(r_zhat) * shift_ind_sigma(i, j, s)   (public)
	let g_parts = build_batch_g_parts::<F, P>(&w_tilde, key_collection, &bitand_prep);
	let h_parts = build_h_parts::<F, P>(r_zhat_prime);

	let SumcheckOutput {
		challenges: mut r_jr_s,
		eval: gamma,
	} = run_phase_1_sumcheck(g_parts, h_parts, channel);

	// Split the phase-1 challenges: r_j is the low bit-index half, r_s the high shift-amount half.
	let r_s = r_jr_s.split_off(LOG_WORD_SIZE_BITS);
	let r_j = r_jr_s;

	// Phase 2: pair the r_j-folded witness against the monster over the word index y.
	// The monster is instance-independent, so it is the single-instance monster verbatim.
	// A neutral IntMul operand contributes nothing: there are no IntMul keys to read its scalars.
	let neutral_intmul = PreparedOperatorData::new(
		OperatorData {
			evals: vec![F::ZERO; INTMUL_ARITY],
			r_zhat_prime,
			r_x_prime: Vec::new(),
		},
		F::ZERO,
	);
	let monster = build_monster_multilinear::<F, P>(
		key_collection,
		&bitand_prep,
		&neutral_intmul,
		&r_j,
		&r_s,
	);

	// Fold W_tilde at r_j: r_j_witness(y) = sum_j eq(r_j, j) * W_tilde[y][j].
	let mut r_j_witness = fold_witness_at_r_j::<F, P>(&w_tilde, &r_j);

	// Match the monster's variable count so the bivariate product sumcheck lines up.
	// The monster rounds the word count up to at least one committed field element.
	if r_j_witness.log_len() < monster.log_len() {
		r_j_witness.zero_extend(monster.log_len());
	}

	run_sumcheck(r_j_witness, monster, r_j, gamma, channel)
}

/// Folds the batch onto one field-valued virtual instance.
///
/// The result is indexed `[word][bit]`:
///
/// ```text
///   W_tilde[y][j] = sum_kappa eq(r_kappa, kappa) * bit_j(w_kappa[y])
/// ```
///
/// The words are independent, so the fold parallelizes over the word index.
fn fold_batch<F: BinaryField>(
	instances: &[&[Word]],
	r_kappa: &[F],
	n_words: usize,
) -> Vec<[F; WORD_SIZE_BITS]> {
	// eq(r_kappa, .) expanded over the 2^k instances: one weight per instance.
	let eps = eq_ind_partial_eval_scalars(r_kappa);

	let mut w_tilde = vec![[F::ZERO; WORD_SIZE_BITS]; n_words];
	w_tilde.par_iter_mut().enumerate().for_each(|(y, row)| {
		// Accumulate every instance's contribution to word y at its set bit positions.
		for (words, &eps_k) in instances.iter().zip(&eps) {
			let bits = words[y].0;
			for (j, slot) in row.iter_mut().enumerate() {
				// A set bit contributes the instance weight; a clear bit contributes nothing.
				if (bits >> j) & 1 == 1 {
					*slot += eps_k;
				}
			}
		}
	});
	w_tilde
}

/// Builds the phase-1 `g` multilinears from the folded witness.
///
/// A key's accumulated weight `acc` batches its operands by the lambda powers.
/// It lands in the `(variant, amount)` block chosen by the key, spread over the bit index `j`:
///
/// ```text
///   g[variant, amount][j] += W_tilde[y][j] * acc     for each word y and each key on it
/// ```
///
/// The scalar layout is `id * WORD_SIZE_BITS + j`, the same as the single-instance builder.
/// The block is `id = variant * WORD_SIZE_BITS + amount`, then bit index `j` in the low bits.
fn build_batch_g_parts<F, P>(
	w_tilde: &[[F; WORD_SIZE_BITS]],
	key_collection: &KeyCollection,
	bitand_prep: &PreparedOperatorData<F>,
) -> [FieldBuffer<P>; SHIFT_VARIANT_COUNT]
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
{
	// One flat scalar buffer holding all 8 variant blocks of the (s, j) hypercube.
	// Heap-allocated on purpose: 8 * 2^12 field elements is far too large for the stack.
	#[allow(clippy::useless_vec)]
	let mut g_flat = vec![F::ZERO; SHIFT_VARIANT_COUNT << LOG_LEN];

	for (range, row) in key_collection.key_ranges.iter().zip(w_tilde) {
		let keys = &key_collection.keys[range.start as usize..range.end as usize];
		for key in keys {
			// IntMul is out of scope; a prepared BitAnd-only system has only BitAnd keys anyway.
			if key.operation != Operation::BitwiseAnd {
				continue;
			}

			// acc = sum_operand lambda^operand * (r_x tensor summed over this key's constraints).
			let acc = key.accumulate(&key_collection.constraint_indices, bitand_prep);

			// key.id selects the (variant, amount) block; j indexes the bit within the word.
			let base = key.id as usize * WORD_SIZE_BITS;
			for (j, &w) in row.iter().enumerate() {
				g_flat[base + j] += w * acc;
			}
		}
	}

	// Split the flat buffer into one multilinear per shift variant.
	g_flat
		.chunks(1 << LOG_LEN)
		.map(FieldBuffer::from_values)
		.collect::<Vec<_>>()
		.try_into()
		.expect("g_flat has SHIFT_VARIANT_COUNT blocks of 2^LOG_LEN scalars")
}

/// Folds the field-valued witness at the bit-index challenge `r_j`.
///
/// The result is a multilinear over the word index:
///
/// ```text
///   folded[y] = sum_j eq(r_j, j) * W_tilde[y][j]
/// ```
fn fold_witness_at_r_j<F, P>(w_tilde: &[[F; WORD_SIZE_BITS]], r_j: &[F]) -> FieldBuffer<P>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
{
	// eq(r_j, .) over the 64 bit positions.
	let r_j_tensor = eq_ind_partial_eval_scalars(r_j);

	// The word count need not be a power of two; the high words round up to zero.
	let n_words = w_tilde.len();
	let log_len = log2_ceil_usize(n_words);

	// One inner product per word collapses the bit dimension; padded words stay zero.
	let mut folded = vec![F::ZERO; 1 << log_len];
	folded[..n_words]
		.par_iter_mut()
		.zip(w_tilde.par_iter())
		.for_each(|(out, row)| {
			*out = row.iter().zip(&r_j_tensor).map(|(&w, &e)| w * e).sum();
		});

	FieldBuffer::from_values(&folded)
}
