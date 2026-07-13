// Copyright 2026 The Binius Developers
//! End-to-end M4 proving throughput for independent BLAKE3 compressions.
//!
//! One BLAKE3 compression runs per instance, and the whole batch is proved together.
//! This is the batched counterpart to the single-instance BLAKE3 compression benchmark.
//! The input ranges match it, so both measure the same primitive.
//!
//! Environment overrides:
//! - `LOG_INSTANCES`: base-2 log of the compression count (default 13 = 8192).
//! - `LOG_INV_RATE`: base-2 log of the inverse Reed-Solomon rate (default 1 = rate 1/2).

#[path = "utils/m4_bench.rs"]
mod m4_bench;

use std::{array, env};

use binius_circuits::blake3::blake3_compress;
use binius_core::word::Word;
use binius_frontend::{Circuit, CircuitBuilder, Wire};
use binius_m4_prover::BatchWitnessFiller;
use criterion::{Criterion, criterion_group, criterion_main};
use m4_bench::bench_m4_proving;
use rand::prelude::*;

/// Base-2 logarithm of the instance count: 2^13 = 8192 compressions.
const DEFAULT_LOG_INSTANCES: usize = 13;

/// Base-2 logarithm of the inverse Reed-Solomon rate: rate 1/2, matching the hash benches.
const DEFAULT_LOG_INV_RATE: usize = 1;

/// The witness input wires of one BLAKE3 compression instance.
///
/// Every field is a witness input.
/// The compression output is force-committed, so the circuit has no inout wires.
#[derive(Clone, Copy)]
struct Blake3Inputs {
	/// The 8-word input chaining value, each a 32-bit value in its low bits.
	cv: [Wire; 8],
	/// The 16-word message block, each a 32-bit value in its low bits.
	block: [Wire; 16],
	/// The 64-bit block counter.
	counter: Wire,
	/// The block length in bytes.
	block_len: Wire,
	/// The domain-separation flags.
	flags: Wire,
}

/// Builds a circuit for one BLAKE3 compression, force-committing its output.
///
/// Force-committing the output keeps the compression alive under dead-code elimination.
/// The circuit has no inout wires, as the batch witness table requires.
fn build_blake3_circuit() -> (Circuit, Blake3Inputs) {
	let builder = CircuitBuilder::new();

	// Every compression input is a witness wire, filled per instance.
	let cv = array::from_fn(|_| builder.add_witness());
	let block = array::from_fn(|_| builder.add_witness());
	let counter = builder.add_witness();
	let block_len = builder.add_witness();
	let flags = builder.add_witness();

	// Force-commit each output word so the compression survives dead-code elimination.
	let out = blake3_compress(&builder, cv, block, counter, block_len, flags);
	for wire in out {
		builder.force_commit(wire);
	}

	(
		builder.build(),
		Blake3Inputs {
			cv,
			block,
			counter,
			block_len,
			flags,
		},
	)
}

/// Assigns one instance's BLAKE3 inputs from a per-instance seeded RNG.
///
/// The value ranges match the single-instance BLAKE3 benchmark.
/// The compression derives its output from these inputs, so any assignment is valid.
/// Seeding per instance keeps the data non-degenerate and reproducible.
fn fill_instance(inputs: &Blake3Inputs, i: usize, w: &mut BatchWitnessFiller<'_, '_>) {
	// Seed from the instance index so the batch is deterministic and instance-varying.
	let mut rng = StdRng::seed_from_u64(i as u64);

	// A 32-bit value per chaining-value word.
	for wire in inputs.cv {
		w[wire] = Word(rng.next_u32() as u64);
	}
	// A 32-bit value per message word.
	for wire in inputs.block {
		w[wire] = Word(rng.next_u32() as u64);
	}
	// A full 64-bit block counter.
	w[inputs.counter] = Word(rng.next_u64());
	// A byte length in 0..=64.
	w[inputs.block_len] = Word((rng.next_u32() % 65) as u64);
	// Arbitrary domain-separation flags.
	w[inputs.flags] = Word(rng.next_u32() as u64);
}

fn bench_blake3_compressions_m4(c: &mut Criterion) {
	// Batch size and code rate are environment-tunable for sweeping.
	let log_instances = env_usize("LOG_INSTANCES").unwrap_or(DEFAULT_LOG_INSTANCES);
	let log_inv_rate = env_usize("LOG_INV_RATE").unwrap_or(DEFAULT_LOG_INV_RATE);

	// One circuit, replicated across the batch by the shared driver.
	let (circuit, inputs) = build_blake3_circuit();

	bench_m4_proving(
		c,
		"blake3_compressions_m4_witness_generation_and_proving",
		&circuit,
		log_instances,
		log_inv_rate,
		|i, w| fill_instance(&inputs, i, w),
	);
}

/// Reads a `usize` environment variable, returning `None` when unset or not a number.
fn env_usize(key: &str) -> Option<usize> {
	env::var(key).ok().and_then(|s| s.parse().ok())
}

criterion_group!(blake3_compressions_m4, bench_blake3_compressions_m4);
criterion_main!(blake3_compressions_m4);
