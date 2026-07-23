// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use crate::{
	aes_field::AESTowerField8b,
	arch::{
		AesInvert1x, AesInvert16x, AesInvert32x, AesInvert64x, AesSquare1x, AesSquare16x,
		AesSquare32x, AesSquare64x, AesWideMul1x, AesWideMul16x, AesWideMul32x, AesWideMul64x,
		M128, M256, M512, MulFromWideMul,
		portable::packed_macros::{portable_macros::*, *},
	},
};

define_packed_binary_field!(
	PackedAESBinaryField1x8b,
	AESTowerField8b,
	u8,
	(MulFromWideMul),
	(AesSquare1x),
	(AesInvert1x),
	(AesWideMul1x)
);
define_packed_binary_field!(
	PackedAESBinaryField16x8b,
	AESTowerField8b,
	M128,
	(MulFromWideMul),
	(AesSquare16x),
	(AesInvert16x),
	(AesWideMul16x)
);
define_packed_binary_field!(
	PackedAESBinaryField32x8b,
	AESTowerField8b,
	M256,
	(MulFromWideMul),
	(AesSquare32x),
	(AesInvert32x),
	(AesWideMul32x)
);
define_packed_binary_field!(
	PackedAESBinaryField64x8b,
	AESTowerField8b,
	M512,
	(MulFromWideMul),
	(AesSquare64x),
	(AesInvert64x),
	(AesWideMul64x)
);

#[cfg(test)]
mod test_utils {
	/// Test if `mult_func` operation is a valid multiply operation on the given values for
	/// all possible packed fields defined on 8-512 bits.
	macro_rules! define_multiply_tests {
		($mult_func:path, $constraint:ty) => {
			$crate::packed_binary_field::test_utils::define_check_packed_mul!(
				$mult_func,
				$constraint
			);

			proptest! {
				#[test]
				fn test_mul_packed_8(a_val in any::<u8>(), b_val in any::<u8>()) {
					TestMult::<$crate::PackedAESBinaryField1x8b>::test_mul(
						a_val.into(),
						b_val.into(),
					);
				}

				#[test]
				fn test_mul_packed_128(a_val in any::<u128>(), b_val in any::<u128>()) {
					TestMult::<$crate::PackedAESBinaryField16x8b>::test_mul(
						a_val.into(),
						b_val.into(),
					);
				}

				#[test]
				fn test_mul_packed_256(a_val in any::<[u128; 2]>(), b_val in any::<[u128; 2]>()) {
					TestMult::<$crate::PackedAESBinaryField32x8b>::test_mul(
						a_val.into(),
						b_val.into(),
					);
				}

				#[test]
				fn test_mul_packed_512(a_val in any::<[u128; 4]>(), b_val in any::<[u128; 4]>()) {
					TestMult::<$crate::PackedAESBinaryField64x8b>::test_mul(
						a_val.into(),
						b_val.into(),
					);
				}
			}
		};
	}

	/// Test if `square_func` operation is a valid square operation on the given value for
	/// all possible packed fields.
	macro_rules! define_square_tests {
		($square_func:path, $constraint:ident) => {
			$crate::packed_binary_field::test_utils::define_check_packed_square!(
				$square_func,
				$constraint
			);

			proptest! {
				#[test]
				fn test_square_packed_8(a_val in any::<u8>()) {
					TestSquare::<$crate::PackedAESBinaryField1x8b>::test_square(a_val.into());
				}

				#[test]
				fn test_square_packed_128(a_val in any::<u128>()) {
					TestSquare::<$crate::PackedAESBinaryField16x8b>::test_square(a_val.into());
				}

				#[test]
				fn test_square_packed_256(a_val in any::<[u128; 2]>()) {
					TestSquare::<$crate::PackedAESBinaryField32x8b>::test_square(a_val.into());
				}

				#[test]
				fn test_square_packed_512(a_val in any::<[u128; 4]>()) {
					TestSquare::<$crate::PackedAESBinaryField64x8b>::test_square(a_val.into());
				}
			}
		};
	}

	/// Test if `invert_func` operation is a valid invert operation on the given value for
	/// all possible packed fields.
	macro_rules! define_invert_tests {
		($invert_func:path, $constraint:ident) => {
			$crate::packed_binary_field::test_utils::define_check_packed_inverse!(
				$invert_func,
				$constraint
			);

			proptest! {
				#[test]
				fn test_invert_packed_8(a_val in any::<u8>()) {
					TestSquare::<$crate::PackedAESBinaryField1x8b>::test_invert(a_val.into());
				}

				#[test]
				fn test_invert_packed_128(a_val in any::<u128>()) {
					TestInvert::<$crate::PackedAESBinaryField16x8b>::test_invert(a_val.into());
				}

				#[test]
				fn test_invert_packed_256(a_val in any::<[u128; 2]>()) {
					TestInvert::<$crate::PackedAESBinaryField32x8b>::test_invert(a_val.into());
				}

				#[test]
				fn test_invert_packed_512(a_val in any::<[u128; 4]>()) {
					TestInvert::<$crate::PackedAESBinaryField64x8b>::test_invert(a_val.into());
				}
			}
		};
	}

	/// Test the widening multiply against the plain multiply for all AES packings.
	macro_rules! define_wide_mul_tests {
		() => {
			fn check_widening_correctness<P>(a: P::Underlier, b: P::Underlier)
			where
				P: $crate::PackedField<Scalar = $crate::AESTowerField8b>
					+ $crate::WideMul
					+ $crate::underlier::WithUnderlier,
			{
				let a = P::from_underlier(a);
				let b = P::from_underlier(b);
				// One deferred product, reduced immediately, must equal the plain multiply.
				let wide = P::wide_mul(a, b);
				let reduced = P::reduce(wide);
				assert_eq!(reduced, a * b);
			}

			fn check_widening_linearity<P>(
				a1: P::Underlier,
				b1: P::Underlier,
				a2: P::Underlier,
				b2: P::Underlier,
			) where
				P: $crate::PackedField<Scalar = $crate::AESTowerField8b>
					+ $crate::WideMul
					+ $crate::underlier::WithUnderlier,
			{
				let (a1, b1) = (P::from_underlier(a1), P::from_underlier(b1));
				let (a2, b2) = (P::from_underlier(a2), P::from_underlier(b2));
				// Accumulated products reduce once at the end.
				// The sum reaches wide values no single product produces, so this covers the
				// reduction's full accumulated domain, not just fresh products.
				let sum_reduced = P::reduce(P::wide_mul(a1, b1) + P::wide_mul(a2, b2));
				assert_eq!(sum_reduced, a1 * b1 + a2 * b2);
			}

			proptest! {
				#[test]
				fn test_wide_mul_correctness_8(a in any::<u8>(), b in any::<u8>()) {
					check_widening_correctness::<$crate::PackedAESBinaryField1x8b>(a, b);
				}

				#[test]
				fn test_wide_mul_correctness_128(a in any::<u128>(), b in any::<u128>()) {
					check_widening_correctness::<$crate::PackedAESBinaryField16x8b>(
						a.into(),
						b.into(),
					);
				}

				#[test]
				fn test_wide_mul_correctness_256(a in any::<[u128; 2]>(), b in any::<[u128; 2]>()) {
					check_widening_correctness::<$crate::PackedAESBinaryField32x8b>(
						a.into(),
						b.into(),
					);
				}

				#[test]
				fn test_wide_mul_correctness_512(a in any::<[u128; 4]>(), b in any::<[u128; 4]>()) {
					check_widening_correctness::<$crate::PackedAESBinaryField64x8b>(
						a.into(),
						b.into(),
					);
				}

				#[test]
				fn test_wide_mul_linearity_8(
					a1 in any::<u8>(), b1 in any::<u8>(),
					a2 in any::<u8>(), b2 in any::<u8>(),
				) {
					check_widening_linearity::<$crate::PackedAESBinaryField1x8b>(a1, b1, a2, b2);
				}

				#[test]
				fn test_wide_mul_linearity_128(
					a1 in any::<u128>(), b1 in any::<u128>(),
					a2 in any::<u128>(), b2 in any::<u128>(),
				) {
					check_widening_linearity::<$crate::PackedAESBinaryField16x8b>(
						a1.into(), b1.into(), a2.into(), b2.into(),
					);
				}

				#[test]
				fn test_wide_mul_linearity_256(
					a1 in any::<[u128; 2]>(), b1 in any::<[u128; 2]>(),
					a2 in any::<[u128; 2]>(), b2 in any::<[u128; 2]>(),
				) {
					check_widening_linearity::<$crate::PackedAESBinaryField32x8b>(
						a1.into(), b1.into(), a2.into(), b2.into(),
					);
				}

				#[test]
				fn test_wide_mul_linearity_512(
					a1 in any::<[u128; 4]>(), b1 in any::<[u128; 4]>(),
					a2 in any::<[u128; 4]>(), b2 in any::<[u128; 4]>(),
				) {
					check_widening_linearity::<$crate::PackedAESBinaryField64x8b>(
						a1.into(), b1.into(), a2.into(), b2.into(),
					);
				}
			}
		};
	}

	pub(crate) use define_invert_tests;
	pub(crate) use define_multiply_tests;
	pub(crate) use define_square_tests;
	pub(crate) use define_wide_mul_tests;
}

#[cfg(test)]
mod tests {
	use std::ops::Mul;

	use proptest::prelude::*;

	use super::test_utils::{
		define_invert_tests, define_multiply_tests, define_square_tests, define_wide_mul_tests,
	};
	use crate::{
		PackedField, WideMul,
		arithmetic_traits::{InvertOrZero, Square},
	};

	define_multiply_tests!(Mul::mul, PackedField);

	define_square_tests!(Square::square, PackedField);

	define_invert_tests!(InvertOrZero::invert_or_zero, PackedField);

	define_wide_mul_tests!();

	#[test]
	fn test_wide_mul_exhaustive_scalar_pairs() {
		// The scalar field has only 2^8 elements, so every product admits an exhaustive check.
		// Each byte pair is broadcast across the 128-bit packing and multiplied deferred.
		//
		//     reduce(wide_mul(a, b)) must equal the scalar product in every lane.
		//
		// The scalar multiply is an independent oracle: it runs the tower-field log/exp tables,
		// not the packed widening path under test.
		for a in 0..=u8::MAX {
			for b in 0..=u8::MAX {
				let expected = crate::AESTowerField8b::new(a) * crate::AESTowerField8b::new(b);

				let a_packed =
					crate::PackedAESBinaryField16x8b::broadcast(crate::AESTowerField8b::new(a));
				let b_packed =
					crate::PackedAESBinaryField16x8b::broadcast(crate::AESTowerField8b::new(b));
				let reduced = crate::PackedAESBinaryField16x8b::reduce(
					crate::PackedAESBinaryField16x8b::wide_mul(a_packed, b_packed),
				);

				assert_eq!(
					reduced,
					crate::PackedAESBinaryField16x8b::broadcast(expected),
					"a={a:#04x} b={b:#04x}"
				);
			}
		}
	}
}
