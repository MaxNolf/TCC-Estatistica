//! Recursive forecasting for ARMA-GARCH models.
//!
//! This module provides convenience wrappers around `GarchFit::forecast()`.
//! The core forecast logic lives in `estimation::solver::GarchFit`.

use crate::estimation::solver::GarchFit;
use crate::spec::CvError;

/// Produces recursive n-step-ahead mean forecasts from a fitted model.
///
/// Delegates to [`GarchFit::forecast()`].
///
/// # Arguments
/// * `fit` — Fitted model with estimated parameters
/// * `n_ahead` — Number of steps to forecast
///
/// # Returns
/// Tuple of (mean_forecasts, variance_forecasts).
pub fn recursive_forecast(fit: &GarchFit, n_ahead: usize) -> Result<(Vec<f64>, Vec<f64>), CvError> {
    fit.forecast(n_ahead)
}
