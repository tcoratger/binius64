// Copyright 2025 Irreducible Inc.
//! # NTT Lookup Table Module
//!
//! This module provides a precomputed lookup table implementation for fast Number Theoretic
//! Transform (NTT) operations on 64-bit binary field elements. The implementation is specifically
//! optimized for the Binius64 protocol's constraint system.
//!
//! ## Overview
//!
//! The NTT lookup table achieves significant performance improvements by precomputing all possible
//! NTT evaluations for 8-bit input chunks. This allows the full 64-bit NTT to be computed by:
//!
//! 1. Splitting the 64 input bits into eight 8-bit chunks
//! 2. Looking up precomputed NTT values for each chunk
//! 3. Adding the results together (exploiting the linearity of the NTT)
//!
//! ## Algorithm
//!
//! The implementation uses additive NTT over binary fields, which is a linear transformation that
//! converts between coefficient and evaluation representations of polynomials. The specific
//! approach:
//!
//! - **Input**: 64 1-bit coefficients representing a polynomial in the Lagrange basis
//! - **Output**: 64 evaluations of the polynomial at specified domain points
//! - **Optimization**: Precomputes all 256 possible evaluations for each 8-bit position
//!
//! ## Striding the output domain
//!
//! - The 64 output evaluations are split into 4 limbs of 16 evaluations.
//! - One limb is exactly the 16 scalars held by one 128-bit packed field.
//! - The table is laid out limb-major.
//! - A single limb's slab spans every byte position and value: `8 * 256 * 16 = 32 KB`.
//! - That slab is contiguous in memory.
//!
//! The hot loop walks the output domain in 4 passes, one limb per pass.
//! Each pass reads only its own 32 KB slab.
//! So the per-pass working set stays in L1, instead of streaming the full `4 * 32 = 128 KB` table.
//! On the benchmark M4 processors that full table is the entire 128 KB L1 data cache.
//! The total arithmetic and byte traffic are identical to a single full-width pass.
//! The win is purely cache residency.

use std::array;

#[cfg(test)]
use binius_field::Divisible;
use binius_field::{BinaryField, BinaryField1b as B1, PackedField, util::expand_subset_sums_array};
use binius_math::{
	BinarySubspace, FieldBuffer,
	ntt::{AdditiveNTT, NeighborsLastReference, domain_context::GenericOnTheFly},
};
use binius_verifier::protocols::bitand::{ROWS_PER_HYPERCUBE_VERTEX, SKIPPED_VARS};

/// Number of evaluations produced per pass, equal to the width of one 128-bit packed field.
pub(crate) const PASS_WIDTH: usize = 16;

/// Number of width-`PASS_WIDTH` limbs that tile the `ROWS_PER_HYPERCUBE_VERTEX` output evaluations.
pub(crate) const N_LIMBS: usize = ROWS_PER_HYPERCUBE_VERTEX / PASS_WIDTH;

/// Number of 8-bit chunks in a 64-bit input word.
const N_BYTES: usize = ROWS_PER_HYPERCUBE_VERTEX / 8;

/// A precomputed lookup table for fast NTT operations on 64-bit binary field elements.
///
/// This structure stores precomputed NTT evaluations for all possible 8-bit input combinations,
/// enabling fast computation of the full 64-bit NTT through table lookups and additions.
///
/// ## Structure
///
/// The internal data is a boxed 3-dimensional array `Box<[[[P; 256]; 8]; N_LIMBS]>` where:
/// - **First dimension** (`N_LIMBS = 4`): which limb of 16 output evaluations the entry holds
/// - **Second dimension** (`8`): the byte position of the chunk within the 64-bit input
/// - **Third dimension** (`256`): the 8-bit value of the chunk
///
/// ## Memory layout
///
/// - The limb index is the outermost axis.
/// - Each limb is a contiguous 32 KB slab covering every byte position and value.
/// - The hot loop reads one slab per pass, keeping the per-pass working set in L1.
/// - See the module docs for the cache-residency argument.
///
/// ## Type Parameters
///
/// - `P`: packed field storing the precomputed values.
/// - Its scalar must be a binary field and its width must equal the per-pass width.
#[derive(Debug, Clone)]
pub struct NTTLookup<P>(Box<[[[P; 256]; 8]; N_LIMBS]>);

impl<F, PNTTDomain> NTTLookup<PNTTDomain>
where
	F: BinaryField,
	PNTTDomain: PackedField<Scalar = F>,
{
	/// Precomputes the per-byte NTT contributions for every byte position and value.
	///
	/// ## Parameters
	///
	/// - `subspace`: binary subspace of dimension 7.
	/// - Its first half is the NTT input domain.
	/// - Its second half is the NTT output coset.
	///
	/// ## Constraints
	///
	/// - The packed field width must equal the per-pass width (16).
	/// - The subspace dimension must be one above the number of skipped variables (7).
	pub fn new(subspace: &BinarySubspace<F>) -> Self {
		assert_eq!(PNTTDomain::WIDTH, PASS_WIDTH);
		assert_eq!(subspace.dim(), SKIPPED_VARS + 1);

		let lde = LowDegreeExtension::<PNTTDomain>::new(subspace);

		// Extend each single coefficient on its own, indexed by byte position and bit within byte.
		// The transform buffer holds 128 scalars as two halves of 64:
		//
		//     [ input domain S | output coset Λ ]
		//
		// As width-16 packing that is 8 elements, and the output coset is the second half.
		// Keep those limbs, one per pass.
		let single_bit = array::from_fn::<_, N_BYTES, _>(|b| {
			array::from_fn::<_, 8, _>(|i| {
				// Set exactly the coefficient at bit `8*b + i`, then take its low-degree extension.
				let output = lde.transform(1 << (8 * b + i));
				assert_eq!(output.log_len(), SKIPPED_VARS + 1);
				// The output coset occupies the second half of the packed elements.
				let limbs: [PNTTDomain; N_LIMBS] = output.as_ref()[N_LIMBS..2 * N_LIMBS]
					.try_into()
					.expect("output buffer has 2 * N_LIMBS packed elements");
				limbs
			})
		});

		// A byte is the XOR of its set bits, and the NTT is linear.
		// So each byte's extension is the sum of its set bits' single-bit images.
		// Expanding the 8 single-bit images gives all 256 byte values at once.
		// Iterating limb-major writes each 32 KB slab as one contiguous block.
		let table = array::from_fn::<_, N_LIMBS, _>(|limb| {
			array::from_fn::<_, N_BYTES, _>(|b| {
				// Gather this limb's image of each of the 8 set-bit positions in the byte.
				let basis: [PNTTDomain; 8] = array::from_fn(|i| single_bit[b][i][limb]);
				expand_subset_sums_array::<_, 8, 256>(basis)
			})
		});

		NTTLookup(Box::new(table))
	}

	/// Borrows the 32 KB lookup slab for a single output limb.
	///
	/// The hot loop reads exactly one slab per pass.
	/// See the module docs for why this keeps the per-pass working set inside L1.
	#[inline]
	pub(crate) fn limb(&self, limb: usize) -> &[[PNTTDomain; 256]; 8] {
		&self.0[limb]
	}

	/// Computes the NTT of 64 1-bit coefficients from the precomputed lookup table.
	///
	/// - The 64 coefficients arrive packed as a byte-divisible value.
	/// - Each byte's contribution is read from the table and the eight are summed.
	/// - Summing is valid because the NTT is linear over the byte decomposition.
	///
	/// For bytes B_0, ..., B_7 the result is NTT(B_0) + ... + NTT(B_7).
	///
	/// This reference path materializes all limbs at once.
	/// The hot loop instead walks one limb per pass for cache locality.
	///
	/// ## Returns
	///
	/// The packed limbs of the NTT evaluations over the output coset.
	#[cfg(test)]
	#[inline]
	pub fn ntt<T: Divisible<u8>>(&self, input: T) -> [PNTTDomain; N_LIMBS] {
		// One accumulator per output limb.
		let mut out = [PNTTDomain::zero(); N_LIMBS];
		// Walk the input low byte first, matching the table's byte positions.
		for (b, byte) in Divisible::value_iter(input).enumerate() {
			// Add this byte's contribution into every limb.
			for (limb, acc) in out.iter_mut().enumerate() {
				*acc += self.0[limb][b][byte as usize];
			}
		}
		out
	}
}

struct LowDegreeExtension<P: PackedField> {
	interpolation: NeighborsLastReference<GenericOnTheFly<P::Scalar>>,
	extrapolation: NeighborsLastReference<GenericOnTheFly<P::Scalar>>,
	_marker: std::marker::PhantomData<P>,
}

impl<F, P> LowDegreeExtension<P>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
{
	fn new(subspace: &BinarySubspace<F>) -> Self {
		assert_eq!(subspace.dim(), SKIPPED_VARS + 1);

		let input_subspace = subspace.reduce_dim(SKIPPED_VARS);
		let input_domain_context = GenericOnTheFly::generate_from_subspace(&input_subspace);
		let output_domain_context = GenericOnTheFly::generate_from_subspace(subspace);

		Self {
			interpolation: NeighborsLastReference {
				domain_context: input_domain_context,
			},
			extrapolation: NeighborsLastReference {
				domain_context: output_domain_context,
			},
			_marker: std::marker::PhantomData,
		}
	}

	fn transform(&self, input: u64) -> FieldBuffer<P> {
		let mut values = FieldBuffer::<P>::zeros(SKIPPED_VARS + 1);

		// Inverse NTT the inputs in the first half of the buffer.
		{
			let mut values_split = values.split_half_mut();
			let (mut input_elems, _) = values_split.halves();

			for i in 0..ROWS_PER_HYPERCUBE_VERTEX {
				input_elems.set(i, F::from(B1::from((input >> i) & 1 == 1)));
			}
			self.interpolation.inverse_transform(input_elems, 0, 0);
		}

		// Forward NTT the zero-padded coefficients.
		self.extrapolation.forward_transform(values.to_mut(), 0, 0);

		values
	}
}

#[cfg(test)]
mod test {
	use binius_field::{AESTowerField8b, PackedAESBinaryField16x8b};
	use binius_math::BinarySubspace;
	use rand::prelude::*;

	use super::*;

	#[test]
	fn test_against_ntt() {
		// Invariant: the lookup table reproduces a from-scratch low-degree extension.
		//
		// Reference: a width-1 (scalar) transform that interpolates then extrapolates each input.
		// Under test: the width-16 lookup table read limb by limb.
		let subspace = BinarySubspace::with_dim(SKIPPED_VARS + 1);
		let lde = LowDegreeExtension::<AESTowerField8b>::new(&subspace);
		let ntt_lookup = NTTLookup::<PackedAESBinaryField16x8b>::new(&subspace);

		// Cover 10 random 64-bit inputs from a fixed seed for reproducibility.
		let mut rng = StdRng::seed_from_u64(0);
		for _ in 0..10 {
			let input = rng.random::<u64>();

			// Reference evaluations on the full domain, and the table's 4 output limbs.
			let lde_result = lde.transform(input);
			let limbs = ntt_lookup.ntt(input);

			// Layout: limb `l` lane `j` is output evaluation `16*l + j`.
			//
			//     reference buffer:  [ input domain (64) | output coset (64) ]
			//     limb l, lane j   →  output coset index 16*l + j
			//                      →  buffer index 64 + 16*l + j
			for (l, limb) in limbs.iter().enumerate() {
				for j in 0..PASS_WIDTH {
					assert_eq!(
						limb.get(j),
						lde_result.get(ROWS_PER_HYPERCUBE_VERTEX + PASS_WIDTH * l + j)
					);
				}
			}
		}
	}
}
