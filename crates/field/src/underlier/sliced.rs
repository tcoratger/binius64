// Copyright 2026 The Binius Developers

//! An underlier whose subdivisions read in transposed order, for sliced packed extension fields.

use std::{
	array,
	fmt::{self, Debug},
	hash::{Hash, Hasher},
	marker::PhantomData,
	ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not},
};

use binius_utils::{
	DeserializeBytes, SerializationError, SerializeBytes,
	bytes::{Buf, BufMut},
	checked_arithmetics::checked_log_2,
};
use bytemuck::{Pod, Zeroable};
use rand::{
	Rng,
	distr::{Distribution, StandardUniform},
};

use super::{Divisible, UnderlierType, mapget};
use crate::Random;

/// An underlier of `N` limbs whose subdivisions read in *transposed* order.
///
/// In memory this is `[U; N]`, identical to a scaled underlier; only the read order differs.
/// The `SubU`-sized element at index `i` is limb `i mod N`, sub-position `i div N`.
/// A scaled underlier instead reads one limb fully before the next.
///
/// This is the layout of a sliced (struct-of-arrays) packed extension field.
/// Limb `j` holds coordinate `j` of every lane, so `N` consecutive reads reassemble one element.
///
/// Degree-two extension, 512-bit limbs of four 128-bit coordinates (`N = 2`):
///
/// ```text
///     limb 0 (coord 0):  a0  a1  a2  a3
///     limb 1 (coord 1):  b0  b1  b2  b3
///     read order:        a0  b0  a1  b1  a2  b2  a3  b3
/// ```
///
/// Index 3 is `b1` — the second coordinate of the second element — not `a3`.
///
/// `SubU` marks the transpose granularity only; it is a zero-sized [`PhantomData`].
#[repr(transparent)]
pub struct SlicedUnderlier<U, SubU, const N: usize>(pub [U; N], PhantomData<SubU>);

impl<U, SubU, const N: usize> SlicedUnderlier<U, SubU, N> {
	/// Wraps `N` limbs into a sliced underlier.
	#[inline]
	pub const fn new(limbs: [U; N]) -> Self {
		Self(limbs, PhantomData)
	}
}

impl<U: Copy, SubU, const N: usize> Clone for SlicedUnderlier<U, SubU, N> {
	#[inline]
	fn clone(&self) -> Self {
		*self
	}
}

impl<U: Copy, SubU, const N: usize> Copy for SlicedUnderlier<U, SubU, N> {}

impl<U: Debug, SubU, const N: usize> Debug for SlicedUnderlier<U, SubU, N> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		// Print only the limbs; the marker carries no runtime data.
		write!(f, "SlicedUnderlier({:?})", self.0)
	}
}

impl<U: PartialEq, SubU, const N: usize> PartialEq for SlicedUnderlier<U, SubU, N> {
	#[inline]
	fn eq(&self, other: &Self) -> bool {
		self.0 == other.0
	}
}

impl<U: Eq, SubU, const N: usize> Eq for SlicedUnderlier<U, SubU, N> {}

impl<U: PartialOrd, SubU, const N: usize> PartialOrd for SlicedUnderlier<U, SubU, N> {
	#[inline]
	fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
		self.0.partial_cmp(&other.0)
	}
}

impl<U: Ord, SubU, const N: usize> Ord for SlicedUnderlier<U, SubU, N> {
	#[inline]
	fn cmp(&self, other: &Self) -> std::cmp::Ordering {
		self.0.cmp(&other.0)
	}
}

impl<U: Hash, SubU, const N: usize> Hash for SlicedUnderlier<U, SubU, N> {
	#[inline]
	fn hash<H: Hasher>(&self, state: &mut H) {
		self.0.hash(state);
	}
}

impl<U: Default, SubU, const N: usize> Default for SlicedUnderlier<U, SubU, N> {
	#[inline]
	fn default() -> Self {
		Self::new(array::from_fn(|_| U::default()))
	}
}

impl<U: Random, SubU, const N: usize> Distribution<SlicedUnderlier<U, SubU, N>>
	for StandardUniform
{
	#[inline]
	fn sample<R: Rng + ?Sized>(&self, mut rng: &mut R) -> SlicedUnderlier<U, SubU, N> {
		// Every limb is independent, so sample them one at a time.
		SlicedUnderlier::new(array::from_fn(|_| U::random(&mut rng)))
	}
}

impl<U, SubU, const N: usize> From<SlicedUnderlier<U, SubU, N>> for [U; N] {
	#[inline]
	fn from(val: SlicedUnderlier<U, SubU, N>) -> Self {
		val.0
	}
}

impl<U, SubU, const N: usize> From<[U; N]> for SlicedUnderlier<U, SubU, N> {
	#[inline]
	fn from(limbs: [U; N]) -> Self {
		Self::new(limbs)
	}
}

// SAFETY: the value is `[U; N]` plus a zero-sized marker, so the all-zero bit pattern is valid
// whenever `U: Zeroable`.
unsafe impl<U: Zeroable, SubU, const N: usize> Zeroable for SlicedUnderlier<U, SubU, N> {}
// SAFETY: `#[repr(transparent)]` over `[U; N]` (the marker is zero-sized), so the layout is exactly
// `[U; N]` with no padding; it is plain-old-data whenever `U` is and the marker outlives every use.
unsafe impl<U: Pod, SubU: 'static, const N: usize> Pod for SlicedUnderlier<U, SubU, N> {}

impl<U: BitAnd<Output = U> + Copy, SubU, const N: usize> BitAnd for SlicedUnderlier<U, SubU, N> {
	type Output = Self;

	#[inline]
	fn bitand(self, rhs: Self) -> Self::Output {
		// Bitwise ops are limb-local, so the transpose never enters here.
		Self::new(array::from_fn(|i| self.0[i] & rhs.0[i]))
	}
}

impl<U: BitAndAssign + Copy, SubU, const N: usize> BitAndAssign for SlicedUnderlier<U, SubU, N> {
	#[inline]
	fn bitand_assign(&mut self, rhs: Self) {
		for i in 0..N {
			self.0[i] &= rhs.0[i];
		}
	}
}

impl<U: BitOr<Output = U> + Copy, SubU, const N: usize> BitOr for SlicedUnderlier<U, SubU, N> {
	type Output = Self;

	#[inline]
	fn bitor(self, rhs: Self) -> Self::Output {
		Self::new(array::from_fn(|i| self.0[i] | rhs.0[i]))
	}
}

impl<U: BitOrAssign + Copy, SubU, const N: usize> BitOrAssign for SlicedUnderlier<U, SubU, N> {
	#[inline]
	fn bitor_assign(&mut self, rhs: Self) {
		for i in 0..N {
			self.0[i] |= rhs.0[i];
		}
	}
}

impl<U: BitXor<Output = U> + Copy, SubU, const N: usize> BitXor for SlicedUnderlier<U, SubU, N> {
	type Output = Self;

	#[inline]
	fn bitxor(self, rhs: Self) -> Self::Output {
		Self::new(array::from_fn(|i| self.0[i] ^ rhs.0[i]))
	}
}

impl<U: BitXorAssign + Copy, SubU, const N: usize> BitXorAssign for SlicedUnderlier<U, SubU, N> {
	#[inline]
	fn bitxor_assign(&mut self, rhs: Self) {
		for i in 0..N {
			self.0[i] ^= rhs.0[i];
		}
	}
}

impl<U: Not<Output = U>, SubU, const N: usize> Not for SlicedUnderlier<U, SubU, N> {
	type Output = Self;

	#[inline]
	fn not(self) -> Self::Output {
		Self::new(self.0.map(U::not))
	}
}

impl<U, SubU, const N: usize> UnderlierType for SlicedUnderlier<U, SubU, N>
where
	U: UnderlierType + Divisible<SubU> + Pod,
	SubU: UnderlierType,
{
	const LOG_BITS: usize = U::LOG_BITS + checked_log_2(N);

	const ZERO: Self = Self([U::ZERO; N], PhantomData);
	// Index 0 maps to limb `0 mod N`, so bit 0 of the value is bit 0 of limb 0.
	// The multiplicative identity therefore lives entirely in the low limb.
	const ONE: Self = {
		let mut arr = [U::ZERO; N];
		arr[0] = U::ONE;
		Self(arr, PhantomData)
	};
	const ONES: Self = Self([U::ONES; N], PhantomData);

	fn interleave(self, other: Self, log_block_len: usize) -> (Self, Self) {
		assert!(log_block_len < Self::LOG_BITS);

		let sub_log_bits = SubU::LOG_BITS;
		if log_block_len < sub_log_bits {
			// Blocks narrower than a `SubU` limb never straddle a limb.
			// The limbs at a fixed sub-position then pair up exactly as a per-limb interleave,
			// so interleaving each limb independently is the whole operation.
			let mut lo = [U::ZERO; N];
			let mut hi = [U::ZERO; N];
			for c in 0..N {
				(lo[c], hi[c]) = self.0[c].interleave(other.0[c], log_block_len);
			}
			(Self::new(lo), Self::new(hi))
		} else {
			// Blocks span whole `SubU` units, so this is a permutation of those units.
			// Riffle them in sliced order, the block transpose the primitive interleave performs:
			//     out0.block[t] = (t even ? self : other).block[2*(t/2)]
			//     out1.block[t] = (t even ? self : other).block[2*(t/2) + 1]
			let s = 1usize << (log_block_len - sub_log_bits);
			let total_units = N << <U as Divisible<SubU>>::LOG_N;

			// Sliced unit `p` lives in limb `p mod N` at `SubU` sub-position `p div N`.
			let unit = |src: &Self, p: usize| -> SubU {
				// Safety: every `p` here is `< total_units`, so `p / N < <U as Divisible<SubU>>::N`
				// and `p % N < N`.
				unsafe { <U as Divisible<SubU>>::get_unchecked(&src.0[p % N], p / N) }
			};

			let mut lo = [U::ZERO; N];
			let mut hi = [U::ZERO; N];
			for p in 0..total_units {
				let (t, o) = (p / s, p % s);
				// Even block draws from `self`, odd from `other`; low output keeps the block, high
				// takes the next.
				let (u_lo, u_hi) = if t % 2 == 0 {
					(unit(&self, t * s + o), unit(&self, (t + 1) * s + o))
				} else {
					(unit(&other, (t - 1) * s + o), unit(&other, t * s + o))
				};
				// Safety: `p < total_units`, so both writes are in bounds, as in `unit`.
				unsafe {
					<U as Divisible<SubU>>::set_unchecked(&mut lo[p % N], p / N, u_lo);
					<U as Divisible<SubU>>::set_unchecked(&mut hi[p % N], p / N, u_hi);
				}
			}
			(Self::new(lo), Self::new(hi))
		}
	}
}

impl<U, SubU, T, const N: usize> Divisible<T> for SlicedUnderlier<U, SubU, N>
where
	U: Divisible<T> + Divisible<SubU> + Pod + Send + Sync,
	SubU: Divisible<T> + Send + Sync + 'static,
	T: Send + 'static,
{
	// A transpose is a permutation, so the count of `T` elements matches a scaled underlier.
	const LOG_N: usize = <U as Divisible<T>>::LOG_N + checked_log_2(N);

	#[inline]
	fn value_iter(value: Self) -> impl ExactSizeIterator<Item = T> + Send + Clone {
		// The index -> position mapping lives in `get`, so mapping over indices yields sliced
		// order.
		mapget::value_iter(value)
	}

	#[inline]
	fn ref_iter(value: &Self) -> impl ExactSizeIterator<Item = T> + Send + Clone + '_ {
		mapget::value_iter(*value)
	}

	#[inline]
	fn slice_iter(slice: &[Self]) -> impl ExactSizeIterator<Item = T> + Send + Clone + '_ {
		mapget::slice_iter(slice)
	}

	#[inline]
	unsafe fn get_unchecked(&self, index: usize) -> T {
		// Split the index into the `T` offset inside a `SubU` and the `SubU` cell in sliced order.
		let k = <SubU as Divisible<T>>::N;
		let t_in_sub = index % k;
		let grid = index / k;
		// Transpose the cell: it sits in limb `grid mod N` at sub-position `grid div N`.
		let (limb, row) = (grid % N, grid / N);
		// Safety: `index < Self::N == <U as Divisible<T>>::N * N`, so `limb < N` and
		// `row * k + t_in_sub < <U as Divisible<T>>::N`.
		unsafe { Divisible::<T>::get_unchecked(self.0.get_unchecked(limb), row * k + t_in_sub) }
	}

	#[inline]
	unsafe fn set_unchecked(&mut self, index: usize, val: T) {
		// Mirror of `get_unchecked`: same index decomposition, writing instead of reading.
		let k = <SubU as Divisible<T>>::N;
		let t_in_sub = index % k;
		let grid = index / k;
		let (limb, row) = (grid % N, grid / N);
		// Safety: as in `get_unchecked`.
		unsafe {
			Divisible::<T>::set_unchecked(self.0.get_unchecked_mut(limb), row * k + t_in_sub, val);
		}
	}

	#[inline]
	fn broadcast(val: T) -> Self {
		// Every position ends up holding `val`, so the ordering is irrelevant; fill each limb.
		Self::new([Divisible::<T>::broadcast(val); N])
	}

	#[inline]
	fn from_iter(mut iter: impl Iterator<Item = T>) -> Self {
		// Placement follows sliced order, so write through `set`; `U: Pod` zeroes any unwritten
		// tail.
		let mut result: Self = Zeroable::zeroed();
		for i in 0..Self::N {
			match iter.next() {
				// Safety: `i < Self::N`.
				Some(val) => unsafe { result.set_unchecked(i, val) },
				None => break,
			}
		}
		result
	}
}

impl<U: SerializeBytes, SubU, const N: usize> SerializeBytes for SlicedUnderlier<U, SubU, N> {
	fn serialize(&self, write_buf: impl BufMut) -> Result<(), SerializationError> {
		// Serialize the raw limbs; deserialization reads them back into the same slots.
		self.0.serialize(write_buf)
	}
}

impl<U: DeserializeBytes, SubU, const N: usize> DeserializeBytes for SlicedUnderlier<U, SubU, N> {
	fn deserialize(read_buf: impl Buf) -> Result<Self, SerializationError> {
		<[U; N]>::deserialize(read_buf).map(Self::new)
	}
}

#[cfg(test)]
mod tests {
	use proptest::prelude::*;

	use super::*;
	use crate::underlier::U1;

	// The stand-in packing throughout: `U = u32` limbs sliced at `SubU = u8` granularity.
	// A `u32` splits into 4 bytes, so `N` limbs expose `4 * N` bytes in sliced order.
	type S2 = SlicedUnderlier<u32, u8, 2>;
	type S4 = SlicedUnderlier<u32, u8, 4>;

	// Reference sliced byte order, computed straight from the spec rather than the implementation.
	// Byte `i` comes from limb `i mod N` at byte-position `i div N`.
	fn ref_bytes(limbs: &[u32]) -> Vec<u8> {
		let n = limbs.len();
		(0..n * 4)
			.map(|i| (limbs[i % n] >> ((i / n) * 8)) as u8)
			.collect()
	}

	// Standard block-transpose interleave over a flat sequence, the semantics the primitive
	// `interleave` implements:
	//   out0.block[t] = (t even ? a : b).block[2*(t/2)]
	//   out1.block[t] = (t even ? a : b).block[2*(t/2) + 1]
	// `which` selects out0 (0) or out1 (1).
	fn flat_interleave<T: Copy>(a: &[T], b: &[T], log_block: usize) -> (Vec<T>, Vec<T>) {
		let s = 1usize << log_block;
		let num_blocks = a.len() / s;
		let build = |which: usize| -> Vec<T> {
			let mut out = Vec::with_capacity(a.len());
			for t in 0..num_blocks {
				// Even block draws from the first operand, odd block from the second.
				// out0 keeps the same block index (shifted down by one on odd t); out1 takes the
				// next.
				let (src, blk) = if t % 2 == 0 {
					(a, t + which)
				} else {
					(b, t - 1 + which)
				};
				out.extend_from_slice(&src[blk * s..blk * s + s]);
			}
			out
		};
		(build(0), build(1))
	}

	#[test]
	fn get_reads_in_transposed_order() {
		// Invariant: index 3 reads limb 1, byte 1 — the transpose the issue calls out — where a
		// scaled underlier would read limb 0, byte 3.
		//
		//     limb 0 = 0x04030201  bytes -> [01, 02, 03, 04]
		//     limb 1 = 0x08070605  bytes -> [05, 06, 07, 08]
		//
		//     sliced order (index -> limb i%2, byte i/2):
		//       [01, 05, 02, 06, 03, 07, 04, 08]
		let x = S2::new([0x04030201, 0x08070605]);
		let got: Vec<u8> = (0..8).map(|i| Divisible::<u8>::get(&x, i)).collect();
		assert_eq!(got, [0x01, 0x05, 0x02, 0x06, 0x03, 0x07, 0x04, 0x08]);
		// The transposed pick differs from the scaled pick at index 3.
		assert_eq!(Divisible::<u8>::get(&x, 3), 0x06);
	}

	#[test]
	fn interleave_at_unit_boundary_groups_limbs() {
		// At a block of exactly one `SubU` unit (log = 8 bits), the byte-unit riffle regroups the
		// limbs: out0 = [a.limb0, b.limb0], out1 = [a.limb1, b.limb1].
		//
		//     a = [0x04030201, 0x08070605]
		//     b = [0x0C0B0A09, 0x100F0E0D]
		//     out0 = [0x04030201, 0x0C0B0A09]
		//     out1 = [0x08070605, 0x100F0E0D]
		let a = S2::new([0x04030201, 0x08070605]);
		let b = S2::new([0x0C0B0A09, 0x100F0E0D]);
		let (c, d) = a.interleave(b, 3);
		assert_eq!(c, S2::new([0x04030201, 0x0C0B0A09]));
		assert_eq!(d, S2::new([0x08070605, 0x100F0E0D]));
	}

	proptest! {
		#[test]
		fn value_iter_matches_reference_order(limbs in any::<[u32; 2]>()) {
			// The iterator must yield bytes in the spec's transposed order.
			let x = S2::new(limbs);
			let got: Vec<u8> = Divisible::<u8>::value_iter(x).collect();
			prop_assert_eq!(got, ref_bytes(&limbs));
		}

		#[test]
		fn value_iter_matches_reference_order_n4(limbs in any::<[u32; 4]>()) {
			// Same order property across four limbs.
			let x = S4::new(limbs);
			let got: Vec<u8> = Divisible::<u8>::value_iter(x).collect();
			prop_assert_eq!(got, ref_bytes(&limbs));
		}

		#[test]
		fn set_then_get_round_trips(limbs in any::<[u32; 2]>(), idx in 0usize..8, val in any::<u8>()) {
			// Writing then reading the same index returns the written byte and disturbs no other.
			let mut x = S2::new(limbs);
			let before: Vec<u8> = Divisible::<u8>::value_iter(x).collect();
			Divisible::<u8>::set(&mut x, idx, val);
			prop_assert_eq!(Divisible::<u8>::get(&x, idx), val);
			for i in (0..8).filter(|&i| i != idx) {
				prop_assert_eq!(Divisible::<u8>::get(&x, i), before[i]);
			}
		}

		#[test]
		fn from_iter_inverts_value_iter(limbs in any::<[u32; 4]>()) {
			// Collecting the sliced bytes then rebuilding reproduces the original value.
			let x = S4::new(limbs);
			let bytes: Vec<u8> = Divisible::<u8>::value_iter(x).collect();
			let rebuilt = <S4 as Divisible<u8>>::from_iter(bytes.into_iter());
			prop_assert_eq!(rebuilt, x);
		}

		#[test]
		fn broadcast_fills_every_lane(val in any::<u8>()) {
			// Every byte position holds the broadcast value regardless of ordering.
			let x = <S4 as Divisible<u8>>::broadcast(val);
			for i in 0..16 {
				prop_assert_eq!(Divisible::<u8>::get(&x, i), val);
			}
		}

		#[test]
		fn slice_iter_concatenates_element_orders(a in any::<[u32; 2]>(), b in any::<[u32; 2]>()) {
			// A slice iterates each element in sliced order, one element after the other.
			let xs = [S2::new(a), S2::new(b)];
			let got: Vec<u8> = Divisible::<u8>::slice_iter(&xs).collect();
			let mut expected = ref_bytes(&a);
			expected.extend(ref_bytes(&b));
			prop_assert_eq!(got, expected);
		}

		#[test]
		fn bit_division_follows_the_same_transpose(limbs in any::<[u32; 2]>()) {
			// Dividing to single bits transposes at the `SubU` unit too: the bits of one limb's byte
			// stay contiguous, and the bytes are picked in sliced order.
			let x = S2::new(limbs);
			let got: Vec<u8> = Divisible::<U1>::value_iter(x).map(|b| b.val()).collect();
			let expected: Vec<u8> = ref_bytes(&limbs)
				.iter()
				.flat_map(|byte| (0..8).map(move |bit| (byte >> bit) & 1))
				.collect();
			prop_assert_eq!(got, expected);
		}

		#[test]
		fn flat_interleave_models_primitive_interleave(a in any::<u32>(), b in any::<u32>()) {
			// Anchor the reference: on a plain `u32`, rebuilding the block-transpose of the bit
			// sequences reproduces the primitive `interleave` for every block length.
			for lbl in 0..<u32 as UnderlierType>::LOG_BITS {
				let a_bits: Vec<U1> = Divisible::<U1>::value_iter(a).collect();
				let b_bits: Vec<U1> = Divisible::<U1>::value_iter(b).collect();
				let (e0, e1) = flat_interleave(&a_bits, &b_bits, lbl);
				let exp0 = <u32 as Divisible<U1>>::from_iter(e0.into_iter());
				let exp1 = <u32 as Divisible<U1>>::from_iter(e1.into_iter());
				let (c, d) = a.interleave(b, lbl);
				prop_assert_eq!(c, exp0);
				prop_assert_eq!(d, exp1);
			}
		}

		#[test]
		fn interleave_matches_flat_transpose_of_sliced_bits(a in any::<[u32; 4]>(), b in any::<[u32; 4]>()) {
			// The real check: sliced `interleave` equals the block-transpose taken over the sliced
			// bit order, for every block length below the full width.
			let (xa, xb) = (S4::new(a), S4::new(b));
			let a_bits: Vec<U1> = Divisible::<U1>::value_iter(xa).collect();
			let b_bits: Vec<U1> = Divisible::<U1>::value_iter(xb).collect();
			for lbl in 0..<S4 as UnderlierType>::LOG_BITS {
				let (e0, e1) = flat_interleave(&a_bits, &b_bits, lbl);
				let exp0 = <S4 as Divisible<U1>>::from_iter(e0.into_iter());
				let exp1 = <S4 as Divisible<U1>>::from_iter(e1.into_iter());
				let (c, d) = xa.interleave(xb, lbl);
				prop_assert_eq!(c, exp0);
				prop_assert_eq!(d, exp1);
			}
		}

		#[test]
		fn serialize_round_trips(limbs in any::<[u32; 2]>()) {
			// Bytes written out deserialize back to the identical value.
			let x = S2::new(limbs);
			let mut buf = Vec::new();
			x.serialize(&mut buf).unwrap();
			let back = S2::deserialize(buf.as_slice()).unwrap();
			prop_assert_eq!(back, x);
		}
	}
}
