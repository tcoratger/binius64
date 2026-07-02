// Copyright 2026 The Binius Developers

use super::error::Error;

/// The value a sumcheck-style prover carries between two consecutive protocol phases.
///
/// A prover alternates between two phases, once per variable:
/// - producing the round polynomial(s) for the variable being bound,
/// - reducing them with the verifier challenge to the claim(s) for the next variable.
///
/// A prover holds exactly one of the two at any time.
/// Tracking which one lets the mandated call order be validated in a single place.
#[derive(Debug, Clone)]
pub enum RoundState<Coeffs, Claim> {
	/// Round polynomial(s) already produced for this variable, awaiting the reduction step.
	Coeffs(Coeffs),
	/// Claim(s) awaiting the next round polynomial(s): the initial claim, or a reduction result.
	Claim(Claim),
}

impl<Coeffs, Claim> RoundState<Coeffs, Claim> {
	/// Borrows the carried claim, needed to start producing this round's polynomial(s).
	///
	/// # Errors
	///
	/// Fails when the prover still holds an unreduced round polynomial.
	/// That means two round polynomials were requested with no reduction step between them.
	pub const fn claim(&self) -> Result<&Claim, Error> {
		match self {
			// A carried claim is the expected input to the round-polynomial phase.
			Self::Claim(claim) => Ok(claim),
			// Holding coefficients means the reduction step is still owed.
			Self::Coeffs(_) => Err(Error::ExpectedFold),
		}
	}

	/// Borrows this round's polynomial(s), needed to start the reduction step.
	///
	/// # Errors
	///
	/// Fails when the prover instead holds a claim.
	/// That means a reduction was requested before its round polynomial was produced.
	pub const fn coeffs(&self) -> Result<&Coeffs, Error> {
		match self {
			// The round polynomial is the expected input to the reduction phase.
			Self::Coeffs(coeffs) => Ok(coeffs),
			// Holding a claim means the round polynomial is still owed.
			Self::Claim(_) => Err(Error::ExpectedExecute),
		}
	}

	/// The error to raise when the protocol is told to finish while variables still remain.
	///
	/// # Returns
	///
	/// The call the prover is still waiting for, encoded as an error:
	/// - holding a round polynomial, the reduction step is owed,
	/// - holding a claim, the next round polynomial is owed.
	pub const fn unfinished_err(&self) -> Error {
		match self {
			Self::Coeffs(_) => Error::ExpectedFold,
			Self::Claim(_) => Error::ExpectedExecute,
		}
	}
}
