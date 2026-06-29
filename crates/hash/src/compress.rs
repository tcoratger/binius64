// Copyright 2024-2025 Irreducible Inc.
// Copyright (c) 2024 The Plonky3 Authors

//! These interfaces are taken from
//! [p3_symmetric](https://github.com/Plonky3/Plonky3/blob/main/symmetric/src/compression.rs) in
//! [Plonky3].
//!
//! Plonky3 is dual-licensed under MIT OR Apache 2.0. We use it under Apache 2.0.
//!
//! [Plonky3]: <https://github.com/plonky3/plonky3>

/// An `N`-to-1 compression function, collision-resistant in a hash-tree setting.
///
/// - It is not assumed collision-resistant for arbitrary inputs.
/// - Collision resistance holds only where every preimage is itself a compression output.
pub trait PseudoCompressionFunction<T, const N: usize>: Clone {
	fn compress(&self, input: [T; N]) -> T;
}
