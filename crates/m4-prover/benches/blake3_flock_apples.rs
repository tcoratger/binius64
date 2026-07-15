// Copyright 2026 The Binius Developers
//! Apples-to-apples BLAKE3 proving benchmark, methodology-matched to Flock.
//!
//! Flock is the reference prover at <https://github.com/succinctlabs/flock>.
//! This mirrors Flock's `blake3_proof` bench so the two numbers measure the same thing:
//!
//! - unit: N independent BLAKE3 compressions, throughput = N / best_time
//! - timed: witness generation + prove (the fast path), not verify
//! - best-of-3 minimum after one warm-up, a fresh witness per run
//! - threads: whatever `RAYON_NUM_THREADS` pins, set identically on both sides
//! - rate 1/2, matching Flock's "fast" config
//!
//! The two-lane core proves 2 compressions per instance.
//! So `2^h` total compressions use `LOG_INSTANCES = h - 1` (Flock: `BLAKE3_LOG2S = h`).
//!
//! Run:
//! ```text
//! BLAKE3_LOG2S="10 12 14" RAYON_NUM_THREADS=8 cargo bench -p binius-m4-prover --features rayon --bench blake3_flock_apples
//! ```

use std::{array, env, hint::black_box, time::Instant};

use binius_circuits::blake3::blake3_compress_2x;
use binius_core::word::Word;
use binius_frontend::{Circuit, CircuitBuilder, Wire};
use binius_m4_prover::{BatchWitnessFiller, Prover, ValueTable};
use binius_m4_verifier::Verifier;
use binius_prover::OptimalPackedB128;
use binius_transcript::ProverTranscript;
use binius_verifier::config::StdChallenger;
use rand::prelude::*;

/// Base-2 logarithm of the inverse Reed-Solomon rate: rate 1/2, matching Flock fast.
const LOG_INV_RATE: usize = 1;

/// Compressions per instance: the two-lane core proves two at once.
const COMPRESSIONS_PER_INSTANCE: usize = 2;

/// The witness input wires of one two-lane BLAKE3 compression instance.
#[derive(Clone, Copy)]
struct Blake3Inputs {
	cv: [Wire; 8],
	block: [Wire; 16],
	counter_lo: Wire,
	counter_hi: Wire,
	block_len: Wire,
	flags: Wire,
}

/// Builds the single-instance two-lane BLAKE3 circuit, output force-committed.
fn build_blake3_circuit() -> (Circuit, Blake3Inputs) {
	let builder = CircuitBuilder::new();
	let cv = array::from_fn(|_| builder.add_witness());
	let block = array::from_fn(|_| builder.add_witness());
	let counter_lo = builder.add_witness();
	let counter_hi = builder.add_witness();
	let block_len = builder.add_witness();
	let flags = builder.add_witness();
	let out = blake3_compress_2x(&builder, cv, block, counter_lo, counter_hi, block_len, flags);
	for wire in out {
		builder.force_commit(wire);
	}
	(
		builder.build(),
		Blake3Inputs {
			cv,
			block,
			counter_lo,
			counter_hi,
			block_len,
			flags,
		},
	)
}

/// Packs two independent 32-bit lane values into one 64-bit word.
const fn pack_lanes(lane0: u32, lane1: u32) -> Word {
	Word((lane0 as u64) | ((lane1 as u64) << 32))
}

/// Fills one instance's inputs from a per-(seed, instance) RNG.
fn fill_instance(inputs: &Blake3Inputs, seed: u64, i: usize, w: &mut BatchWitnessFiller<'_, '_>) {
	// Vary by both seed and instance so each run hits a fresh witness, as Flock does.
	let mut rng = StdRng::seed_from_u64(seed ^ (i as u64));
	for wire in inputs.cv {
		w[wire] = pack_lanes(rng.next_u32(), rng.next_u32());
	}
	for wire in inputs.block {
		w[wire] = pack_lanes(rng.next_u32(), rng.next_u32());
	}
	w[inputs.counter_lo] = pack_lanes(rng.next_u32(), rng.next_u32());
	w[inputs.counter_hi] = pack_lanes(rng.next_u32(), rng.next_u32());
	w[inputs.block_len] = pack_lanes(rng.next_u32() % 65, rng.next_u32() % 65);
	w[inputs.flags] = pack_lanes(rng.next_u32(), rng.next_u32());
}

/// Formats seconds the way Flock's bench does.
fn fmt_ms(s: f64) -> String {
	let ms = s * 1000.0;
	if ms < 1.0 {
		format!("{:>8.2} µs", s * 1e6)
	} else if ms < 1000.0 {
		format!("{:>8.2} ms", ms)
	} else {
		format!("{:>8.2} s ", s)
	}
}

/// Benches one target size: `n_compressions` total, best-of-`n_runs`.
fn bench_one(n_compressions: usize, n_runs: usize) {
	// Two compressions per instance, so the instance count is half the target.
	let log_instances = (n_compressions / COMPRESSIONS_PER_INSTANCE)
		.next_power_of_two()
		.trailing_zeros() as usize;
	let n_instances = 1usize << log_instances;
	let total = n_instances * COMPRESSIONS_PER_INSTANCE;

	println!("\n=== {total:>5} compressions  (log_instances = {log_instances}) ===");

	// One-time setup, outside timing, exactly as the real prover does it.
	let (circuit, inputs) = build_blake3_circuit();
	let mut cs = circuit.constraint_system().clone();
	cs.validate_and_prepare().unwrap();
	let verifier = Verifier::setup(&cs, log_instances, LOG_INV_RATE);
	let prover = Prover::<OptimalPackedB128>::setup(&verifier);

	// The timed unit: regenerate the batch witness, then prove it. No verify.
	let run_once = |seed: u64| {
		let table = ValueTable::populate_parallel(&circuit, log_instances, |i, w| {
			fill_instance(&inputs, seed, i, w)
		})
		.unwrap();
		let mut transcript = ProverTranscript::new(StdChallenger::default());
		prover.prove(&table, &mut transcript);
		transcript
	};

	// Warm-up.
	black_box(run_once(0xC0FFEE_BEEF ^ total as u64));

	// Best-of-n_runs, fresh witness (and thus transcript) each run.
	let mut best = f64::INFINITY;
	for run in 0..n_runs {
		let seed = 0xC0FFEE_BEEF ^ (total as u64) ^ ((run as u64) + 1);
		let t0 = Instant::now();
		let transcript = run_once(seed);
		let elapsed = t0.elapsed().as_secs_f64();
		best = best.min(elapsed);
		black_box(&transcript);
		println!("  [run {}/{}] prove: {}", run + 1, n_runs, fmt_ms(elapsed));
	}
	println!("  best prove: {}  ({:.0} compressions/sec)", fmt_ms(best), total as f64 / best);
}

fn main() {
	println!("BLAKE3 compression-function M4 proof timings (witness gen + prove, no verify).");
	println!("threads: {} (RAYON_NUM_THREADS)", binius_utils::rayon::current_num_threads());

	// Sizes: log2 of TOTAL compressions, matching Flock's BLAKE3_LOG2S semantics.
	let specs: Vec<(usize, usize)> = match env::var("BLAKE3_LOG2S") {
		Ok(s) => s
			.split([',', ' '])
			.filter(|t| !t.is_empty())
			.map(|t| (1usize << t.parse::<u32>().expect("integer log2"), 3usize))
			.collect(),
		Err(_) => vec![(1 << 10, 3), (1 << 12, 3), (1 << 14, 3), (1 << 16, 3)],
	};
	for &(n, n_runs) in &specs {
		bench_one(n, n_runs);
	}
}
