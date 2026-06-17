//! # arma-garch-cv
//!
//! ARMA(p,q)-GARCH(m,s) model with purged temporal cross-validation.
//!
//! ## Modules
//! - `spec` — Type definitions: orders, params, results, errors
//! - `cv` — Cross-validation engine (purged walk-forward)
//! - `estimation` — Model fitting via MLE (Normal distribution)
//! - `distributions` — Conditional distributions for GARCH
//! - `forecast` — Recursive forecasting (delegates to GarchFit)
//! - `progress` — Progress bar with indicatif
//! - `r_bridge` — R integration via extendr (feature: r-bridge)
//! - `py_bridge` — Python integration via PyO3 (feature: python)

pub mod spec;
pub mod cv;
pub mod estimation;
pub mod distributions;
pub mod forecast;
pub mod progress;

#[cfg(feature = "r-bridge")]
pub mod r_bridge;

#[cfg(feature = "python")]
pub mod py_bridge;
