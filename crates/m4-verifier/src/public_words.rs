// Copyright 2026 The Binius Developers

//! The public-word segment shared by every instance in an M4 batch.

use binius_core::{constraint_system::ConstraintSystem, word::Word};

/// Returns the public-word segment, zero-padded to its power-of-two layout length.
///
/// The shift reduction reads the public words as a multilinear over a power-of-two hypercube.
/// So the slice it receives must have exactly that length.
/// The raw constant bank is stored unpadded and is not directly usable.
///
/// The prover and verifier both call this, so the two sides pad identically.
/// The pad is a no-op once the constants already fill the segment.
///
/// # Panics
///
/// Panics if the constraint system has inout wires.
/// The batch setting forbids them, so the public segment is exactly the constants.
pub fn padded_public_words(cs: &ConstraintSystem) -> Vec<Word> {
	// The batch setting has no inout wires, so the public segment is exactly the constants.
	assert_eq!(
		cs.value_vec_layout.n_inout, 0,
		"M4 forbids inout wires; the public segment is exactly the constants"
	);

	// The layout rounds the public segment up to a power of two.
	// The witness offset is the start of the next segment, so it is that rounded length.
	let len = 1usize << cs.value_vec_layout.log_public_words();
	debug_assert!(
		cs.constants.len() <= len,
		"constants ({}) must fit in the padded public segment ({len})",
		cs.constants.len(),
	);

	// Constants sit at the low indices; the tail is the layout's zero padding.
	let mut words = vec![Word::ZERO; len];
	words[..cs.constants.len()].copy_from_slice(&cs.constants);
	words
}
