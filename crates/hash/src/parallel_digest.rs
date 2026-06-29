// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::{marker::PhantomData, mem::MaybeUninit};

use binius_utils::{
	FixedSizeSerializeBytes, SerializeBytes,
	rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator},
};
use digest::{Digest, FixedOutputReset, Output, block_api::BlockSizeUser};

use crate::HashBuffer;

pub trait ParallelDigest: Send {
	/// The corresponding non-parallelized hash function.
	type Digest: Digest;

	/// Create new hasher instance with empty state.
	fn new() -> Self;

	/// Calculate the digest of multiple hashes by processing a parallel iterator of iterators.
	///
	/// The source parameter provides a parallel iterator where:
	/// - Each element of the outer iterator maps to one leaf/digest in the output
	/// - Each element contains an inner iterator of items that will be serialized and concatenated
	///   to form that leaf's content
	///
	/// # Panics
	/// All items must be able to serialize with SerializationMode::Native without error, or this
	/// method will panic.
	fn digest<I: IntoIterator<Item: SerializeBytes>>(
		&self,
		source: impl IndexedParallelIterator<Item = I>,
		out: &mut [MaybeUninit<Output<Self::Digest>>],
	);

	/// Like [`digest`](Self::digest), but specialized for the case where every leaf is built from
	/// exactly `n_items_per_input` items of a [`FixedSizeSerializeBytes`] type, so that each leaf
	/// has the same, compile-time-derivable byte length.
	///
	/// This extra structure lets implementations skip per-leaf length bookkeeping (and, for short
	/// leaves, the message padding) that [`digest`](Self::digest) must redo every time. The default
	/// implementation simply forwards to [`digest`](Self::digest).
	///
	/// # Panics
	/// Each iterator in `source` must yield exactly `n_items_per_input` items, and all items must
	/// serialize without error, or this method may panic.
	fn digest_with_const_len<I: IntoIterator<Item: FixedSizeSerializeBytes>>(
		&self,
		n_items_per_input: usize,
		source: impl IndexedParallelIterator<Item = I>,
		out: &mut [MaybeUninit<Output<Self::Digest>>],
	) {
		let _ = n_items_per_input;
		self.digest(source, out);
	}
}

/// Adapts a sequential [`Digest`] into a [`ParallelDigest`] that hashes one leaf per element of a
/// parallel iterator.
///
/// Each Rayon work-item is seeded with a single hasher (via `for_each_with`) which is recycled in
/// place with `finalize_reset` between leaves, rather than cloning a fresh hasher per leaf. This
/// requires `D: FixedOutputReset`.
pub struct ParallelDigestAdapter<D>(PhantomData<D>);

impl<D> Default for ParallelDigestAdapter<D> {
	fn default() -> Self {
		Self(PhantomData)
	}
}

impl<D> ParallelDigest for ParallelDigestAdapter<D>
where
	D: Digest + FixedOutputReset + BlockSizeUser + Send + Sync + Clone,
{
	type Digest = D;

	fn new() -> Self {
		Self(PhantomData)
	}

	fn digest<I: IntoIterator<Item: SerializeBytes>>(
		&self,
		source: impl IndexedParallelIterator<Item = I>,
		out: &mut [MaybeUninit<Output<Self::Digest>>],
	) {
		source
			.zip(out.par_iter_mut())
			.for_each_with(D::new(), |hasher, (items, out)| {
				{
					let mut buffer = HashBuffer::new(hasher);
					for item in items {
						item.serialize(&mut buffer)
							.expect("pre-condition: items must serialize without error")
					}
				}
				out.write(hasher.finalize_reset());
			});
	}
}

#[cfg(test)]
mod tests {
	use std::iter::repeat_with;

	use binius_utils::rayon::iter::IntoParallelRefIterator;
	use rand::prelude::*;

	use super::*;

	fn generate_mock_data(n_hashes: usize, chunk_size: usize) -> Vec<Vec<u8>> {
		let mut rng = StdRng::seed_from_u64(0);

		(0..n_hashes)
			.map(|_| {
				let mut chunk = vec![0; chunk_size];
				rng.fill_bytes(&mut chunk);
				chunk
			})
			.collect()
	}

	#[test]
	fn test_adapter_matches_serial_sha256() {
		use sha2::Sha256;

		for n_hashes in [0, 1, 2, 4, 8, 9, 100] {
			let data = generate_mock_data(n_hashes, 16);

			let adapter = ParallelDigestAdapter::<Sha256>::new();
			let mut results = repeat_with(MaybeUninit::<Output<Sha256>>::uninit)
				.take(data.len())
				.collect::<Vec<_>>();
			adapter.digest(data.par_iter(), &mut results);

			for (result, leaf) in results.into_iter().zip(&data) {
				assert_eq!(unsafe { result.assume_init() }, <Sha256 as Digest>::digest(leaf));
			}
		}
	}
}
