//! Normal (Gaussian) conditional distribution for GARCH residuals.
//!
//! The standardized normal provides the simplest log-likelihood case:
//!
//! ```text
//! log f(z) = -0.5 * (ln(2π) + z²)
//! ```
//!
//! where z = ε_t / σ_t is the standardized residual.

use super::ConditionalDistribution;

/// Precomputed constant: ln(2π)
const LN_2PI: f64 = 1.8378770664093453; // = (2.0 * PI).ln()

/// Standard Normal distribution for GARCH residuals.
///
/// No additional parameters beyond the model's own (μ, φ, θ, ω, α, β).
pub struct NormalDist;

impl NormalDist {
    pub fn new() -> Self {
        Self
    }
}

impl Default for NormalDist {
    fn default() -> Self {
        Self::new()
    }
}

impl ConditionalDistribution for NormalDist {
    /// Log-PDF of the standard Normal evaluated at z.
    ///
    /// `log f(z) = -0.5 * (ln(2π) + z²)`
    fn log_pdf(&self, z: f64, _params: &[f64]) -> f64 {
        -0.5 * (LN_2PI + z * z)
    }

    fn n_params(&self) -> usize {
        0 // Normal has no extra distribution parameters
    }

    fn name(&self) -> &str {
        "norm"
    }

    fn initial_params(&self) -> Vec<f64> {
        vec![] // No extra parameters
    }

    fn param_bounds(&self) -> (Vec<f64>, Vec<f64>) {
        (vec![], vec![])
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn assert_approx(a: f64, b: f64, tol: f64) {
        assert!(
            (a - b).abs() < tol,
            "expected ~{}, got {} (diff: {})",
            b, a, (a - b).abs()
        );
    }

    #[test]
    fn test_log_pdf_at_zero() {
        let d = NormalDist::new();
        // log f(0) = -0.5 * ln(2π) ≈ -0.9189385
        let expected = -0.5 * (2.0 * PI).ln();
        assert_approx(d.log_pdf(0.0, &[]), expected, 1e-12);
    }

    #[test]
    fn test_log_pdf_at_one() {
        let d = NormalDist::new();
        // log f(1) = -0.5 * (ln(2π) + 1) ≈ -1.4189385
        let expected = -0.5 * ((2.0 * PI).ln() + 1.0);
        assert_approx(d.log_pdf(1.0, &[]), expected, 1e-12);
    }

    #[test]
    fn test_log_pdf_symmetry() {
        let d = NormalDist::new();
        // f(-z) = f(z)
        assert_approx(d.log_pdf(2.5, &[]), d.log_pdf(-2.5, &[]), 1e-15);
    }

    #[test]
    fn test_pdf_integrates_correctly() {
        let d = NormalDist::new();
        // Verify pdf(0) = 1/√(2π) ≈ 0.3989422
        let pdf_val = d.pdf(0.0, &[]);
        let expected = 1.0 / (2.0 * PI).sqrt();
        assert_approx(pdf_val, expected, 1e-12);
    }

    #[test]
    fn test_n_params() {
        assert_eq!(NormalDist::new().n_params(), 0);
    }

    #[test]
    fn test_log_pdf_tail() {
        let d = NormalDist::new();
        // At z=5, log f should be very negative
        let val = d.log_pdf(5.0, &[]);
        assert!(val < -10.0);
        // Exact: -0.5*(ln(2π)+25) ≈ -13.4189
        assert_approx(val, -0.5 * (LN_2PI + 25.0), 1e-12);
    }
}
