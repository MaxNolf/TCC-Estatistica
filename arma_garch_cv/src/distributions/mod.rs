//! Conditional distributions for GARCH residuals.
//!
//! **Status: STUB** — to be implemented.
//!
//! ## Plan
//! - Define `trait ConditionalDistribution` with `log_pdf`, `pdf`, `cdf`
//! - Implement: Normal, StudentT, GED, SkewedNormal, SkewedStudentT, SkewedGED, JSU
//! - Each distribution contributes to the log-likelihood in `estimation::likelihood`

pub mod normal;

/// Trait for conditional distributions used in GARCH models.
///
/// Each distribution provides the log-PDF needed by the
/// log-likelihood function during MLE estimation.
pub trait ConditionalDistribution: Send + Sync {
    /// Log-PDF of the standardized distribution.
    ///
    /// # Arguments
    /// * `z` — Standardized residual (z = eps / sigma)
    /// * `params` — Distribution-specific parameters (e.g., df for Student-t)
    fn log_pdf(&self, z: f64, params: &[f64]) -> f64;

    /// PDF of the standardized distribution.
    fn pdf(&self, z: f64, params: &[f64]) -> f64 {
        self.log_pdf(z, params).exp()
    }

    /// Number of additional parameters for this distribution.
    ///
    /// Normal = 0, Student-t = 1 (df), GED = 1 (shape), etc.
    fn n_params(&self) -> usize;

    /// Returns the name of the distribution.
    fn name(&self) -> &str;

    /// Initial guesses for distribution parameters.
    fn initial_params(&self) -> Vec<f64>;

    /// Parameter bounds (lower, upper) for constrained optimization.
    fn param_bounds(&self) -> (Vec<f64>, Vec<f64>);
}
