//! Python bindings for ARMA-GARCH cross-validation via PyO3.
//!
//! Exposes three main functions to Python:
//! - `fit_arma_garch` — single model fit
//! - `grid_search_parallel` — parallel grid search over ARMA×GARCH orders
//! - `rolling_variance_forecast` — rolling 1-step-ahead variance forecast

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
use pyo3::types::PyDict;

#[cfg(feature = "python")]
use rayon::prelude::*;

#[cfg(feature = "python")]
use crate::spec::*;

#[cfg(feature = "python")]
use crate::estimation::solver::garch_fit;

#[cfg(feature = "python")]
use crate::estimation::likelihood::SGarchLikelihood;

// =============================================================================
// Helper: compute AIC/BIC from neg-log-likelihood
// =============================================================================
#[cfg(feature = "python")]
fn compute_aic_bic(neg_log_lik: f64, n_params: usize, n_obs: usize) -> (f64, f64) {
    let k = n_params as f64;
    let n = n_obs as f64;
    let aic = 2.0 * neg_log_lik + 2.0 * k;
    let bic = 2.0 * neg_log_lik + k * n.ln();
    (aic, bic)
}

// =============================================================================
// Function 1: fit_arma_garch
// =============================================================================

/// Fit a single ARMA(p,q)-GARCH(m,s) model.
///
/// Args:
///     data: list[float] — time series (e.g. log returns)
///     arma_p: int — AR order
///     arma_q: int — MA order
///     garch_m: int — ARCH order
///     garch_s: int — GARCH order
///
/// Returns:
///     dict with keys: params, neg_log_lik, aic, bic,
///                     conditional_variances, residuals, converged
#[cfg(feature = "python")]
#[pyfunction]
fn fit_arma_garch(
    py: Python<'_>,
    data: Vec<f64>,
    arma_p: usize,
    arma_q: usize,
    garch_m: usize,
    garch_s: usize,
) -> PyResult<PyObject> {
    let spec = ModelSpec::new(
        ArmaOrder::new(arma_p, arma_q),
        GarchOrder::new(garch_m, garch_s),
        Distribution::Normal,
        Solver::Hybrid,
    );

    let dict = PyDict::new(py);

    match garch_fit(&spec, &data) {
        Ok(fit) => {
            let n_params = 1 + arma_p + arma_q + 1 + garch_m + garch_s;
            let (aic, bic) = compute_aic_bic(fit.neg_log_lik, n_params, data.len());

            dict.set_item("params", fit.params.clone())?;
            dict.set_item("neg_log_lik", fit.neg_log_lik)?;
            dict.set_item("aic", aic)?;
            dict.set_item("bic", bic)?;
            dict.set_item("conditional_variances", fit.conditional_variances.clone())?;
            dict.set_item("residuals", fit.residuals.clone())?;
            dict.set_item("converged", true)?;
        }
        Err(e) => {
            dict.set_item("converged", false)?;
            dict.set_item("error", format!("{}", e))?;
            dict.set_item("aic", f64::INFINITY)?;
            dict.set_item("bic", f64::INFINITY)?;
        }
    }

    Ok(dict.into())
}

// =============================================================================
// Function 2: grid_search_parallel
// =============================================================================

/// Result of a single grid search evaluation.
#[cfg(feature = "python")]
#[derive(Clone)]
struct GridResult {
    arma_p: usize,
    arma_q: usize,
    garch_m: usize,
    garch_s: usize,
    aic: f64,
    bic: f64,
    neg_log_lik: f64,
    converged: bool,
}

/// Parallel grid search over ARMA(p,q) × GARCH(m,s) parameter combinations.
///
/// Args:
///     data: list[float] — time series
///     p_max: int — max AR order (searches 0..=p_max)
///     q_max: int — max MA order (searches 0..=q_max)
///     gm_max: int — max ARCH order (searches 1..=gm_max)
///     gs_max: int — max GARCH order (searches 1..=gs_max)
///     criterion: str — "aic" or "bic"
///
/// Returns:
///     dict with keys:
///       best_arma_p, best_arma_q, best_garch_m, best_garch_s,
///       best_aic, best_bic, best_neg_log_lik,
///       all_results (list of dicts), n_converged, n_total
#[cfg(feature = "python")]
#[pyfunction]
fn grid_search_parallel(
    py: Python<'_>,
    data: Vec<f64>,
    p_max: usize,
    q_max: usize,
    gm_max: usize,
    gs_max: usize,
    criterion: String,
) -> PyResult<PyObject> {
    // Build all parameter combinations
    let mut combos: Vec<(usize, usize, usize, usize)> = Vec::new();
    for p in 0..=p_max {
        for q in 0..=q_max {
            for gm in 1..=gm_max {
                for gs in 1..=gs_max {
                    combos.push((p, q, gm, gs));
                }
            }
        }
    }

    let n_total = combos.len();

    // Run all fits in parallel via rayon
    let results: Vec<GridResult> = combos
        .par_iter()
        .map(|&(p, q, gm, gs)| {
            let spec = ModelSpec::new(
                ArmaOrder::new(p, q),
                GarchOrder::new(gm, gs),
                Distribution::Normal,
                Solver::Hybrid,
            );

            match garch_fit(&spec, &data) {
                Ok(fit) => {
                    let n_params = 1 + p + q + 1 + gm + gs;
                    let (aic, bic) = compute_aic_bic(fit.neg_log_lik, n_params, data.len());
                    GridResult {
                        arma_p: p,
                        arma_q: q,
                        garch_m: gm,
                        garch_s: gs,
                        aic,
                        bic,
                        neg_log_lik: fit.neg_log_lik,
                        converged: true,
                    }
                }
                Err(_) => GridResult {
                    arma_p: p,
                    arma_q: q,
                    garch_m: gm,
                    garch_s: gs,
                    aic: f64::INFINITY,
                    bic: f64::INFINITY,
                    neg_log_lik: f64::INFINITY,
                    converged: false,
                },
            }
        })
        .collect();

    // Find the best model
    let use_aic = criterion.to_lowercase() == "aic";
    let best = results
        .iter()
        .filter(|r| r.converged)
        .min_by(|a, b| {
            let va = if use_aic { a.aic } else { a.bic };
            let vb = if use_aic { b.aic } else { b.bic };
            va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
        });

    let n_converged = results.iter().filter(|r| r.converged).count();

    let dict = PyDict::new(py);

    if let Some(best) = best {
        dict.set_item("best_arma_p", best.arma_p)?;
        dict.set_item("best_arma_q", best.arma_q)?;
        dict.set_item("best_garch_m", best.garch_m)?;
        dict.set_item("best_garch_s", best.garch_s)?;
        dict.set_item("best_aic", best.aic)?;
        dict.set_item("best_bic", best.bic)?;
        dict.set_item("best_neg_log_lik", best.neg_log_lik)?;
    }

    // Build all_results list
    let all_results: Vec<PyObject> = results
        .iter()
        .filter(|r| r.converged)
        .map(|r| {
            let d = PyDict::new(py);
            let _ = d.set_item("arma_p", r.arma_p);
            let _ = d.set_item("arma_q", r.arma_q);
            let _ = d.set_item("garch_m", r.garch_m);
            let _ = d.set_item("garch_s", r.garch_s);
            let _ = d.set_item("aic", r.aic);
            let _ = d.set_item("bic", r.bic);
            d.into()
        })
        .collect();

    dict.set_item("all_results", all_results)?;
    dict.set_item("n_converged", n_converged)?;
    dict.set_item("n_total", n_total)?;

    Ok(dict.into())
}

// =============================================================================
// Function 3: rolling_variance_forecast
// =============================================================================

/// Rolling 1-step-ahead variance forecast using fixed ARMA-GARCH parameters.
///
/// The model is fitted ONCE on data[0..n_train], then the conditional variance
/// is computed recursively over the entire series using the fixed parameters.
/// Returns the conditional variance for the test period data[n_train..].
///
/// This is much faster than re-estimating at every step because:
/// - Fitting (MLE optimization) is O(iterations × T) and done only once
/// - The recursive variance computation is O(T) with tiny constant
///
/// Args:
///     data: list[float] — full time series (train + test concatenated)
///     arma_p, arma_q: int — ARMA orders
///     garch_m, garch_s: int — GARCH orders
///     n_train: int — number of training observations
///
/// Returns:
///     dict with keys:
///       forecast_variance: list[float] — conditional variance for test period
///       train_variance: list[float] — conditional variance for train period
///       converged: bool
#[cfg(feature = "python")]
#[pyfunction]
fn rolling_variance_forecast(
    py: Python<'_>,
    data: Vec<f64>,
    arma_p: usize,
    arma_q: usize,
    garch_m: usize,
    garch_s: usize,
    n_train: usize,
) -> PyResult<PyObject> {
    let dict = PyDict::new(py);

    if n_train >= data.len() || n_train < 10 {
        dict.set_item("converged", false)?;
        dict.set_item("error", "n_train inválido")?;
        return Ok(dict.into());
    }

    let train_data = &data[..n_train];

    let spec = ModelSpec::new(
        ArmaOrder::new(arma_p, arma_q),
        GarchOrder::new(garch_m, garch_s),
        Distribution::Normal,
        Solver::Hybrid,
    );

    // Fit model on training data
    let fit = match garch_fit(&spec, train_data) {
        Ok(f) => f,
        Err(e) => {
            dict.set_item("converged", false)?;
            dict.set_item("error", format!("{}", e))?;
            return Ok(dict.into());
        }
    };

    // Now compute residuals and variances on the FULL dataset using
    // the fixed parameters estimated from training data.
    let likelihood = SGarchLikelihood::new(spec);
    let (_, full_variances) = likelihood.compute_residuals_and_variances(&fit.params, &data);

    // Split into train and test variances
    let train_variance = full_variances[..n_train].to_vec();
    let forecast_variance = full_variances[n_train..].to_vec();

    let n_params = 1 + arma_p + arma_q + 1 + garch_m + garch_s;
    let (aic, bic) = compute_aic_bic(fit.neg_log_lik, n_params, n_train);

    dict.set_item("forecast_variance", forecast_variance)?;
    dict.set_item("train_variance", train_variance)?;
    dict.set_item("params", fit.params)?;
    dict.set_item("aic", aic)?;
    dict.set_item("bic", bic)?;
    dict.set_item("converged", true)?;

    Ok(dict.into())
}

// =============================================================================
// Function 4: batch grid search for CPCV folds (all folds in parallel)
// =============================================================================

/// Runs grid search across multiple CPCV folds in parallel.
///
/// Each fold gets its own train/test split. For each fold, all parameter
/// combinations are evaluated. Results are aggregated across folds.
///
/// Args:
///     data: list[float] — full training series
///     fold_train_indices: list[list[int]] — train indices per fold
///     fold_test_indices: list[list[int]] — test indices per fold
///     p_max, q_max, gm_max, gs_max: int — max orders
///     criterion: str — "aic" or "bic"
///
/// Returns:
///     list of dicts (one per fold) with best model and metrics
#[cfg(feature = "python")]
#[pyfunction]
fn cpcv_grid_search(
    py: Python<'_>,
    data: Vec<f64>,
    fold_train_indices: Vec<Vec<usize>>,
    _fold_test_indices: Vec<Vec<usize>>,
    p_max: usize,
    q_max: usize,
    gm_max: usize,
    gs_max: usize,
    criterion: String,
) -> PyResult<PyObject> {
    let use_aic = criterion.to_lowercase() == "aic";
    let n_folds = fold_train_indices.len();

    // Build parameter grid once
    let mut combos: Vec<(usize, usize, usize, usize)> = Vec::new();
    for p in 0..=p_max {
        for q in 0..=q_max {
            for gm in 1..=gm_max {
                for gs in 1..=gs_max {
                    combos.push((p, q, gm, gs));
                }
            }
        }
    }

    // Process all folds in parallel
    let fold_results: Vec<Option<GridResult>> = (0..n_folds)
        .into_par_iter()
        .map(|fold_idx| {
            // Extract fold training data
            let train_idx = &fold_train_indices[fold_idx];
            let fold_data: Vec<f64> = train_idx.iter().map(|&i| data[i]).collect();

            if fold_data.len() < 10 {
                return None;
            }

            // Run grid search for this fold
            let fold_grid: Vec<GridResult> = combos
                .iter()
                .filter_map(|&(p, q, gm, gs)| {
                    let spec = ModelSpec::new(
                        ArmaOrder::new(p, q),
                        GarchOrder::new(gm, gs),
                        Distribution::Normal,
                        Solver::Hybrid,
                    );

                    match garch_fit(&spec, &fold_data) {
                        Ok(fit) => {
                            let n_params = 1 + p + q + 1 + gm + gs;
                            let (aic, bic) = compute_aic_bic(
                                fit.neg_log_lik, n_params, fold_data.len()
                            );
                            Some(GridResult {
                                arma_p: p,
                                arma_q: q,
                                garch_m: gm,
                                garch_s: gs,
                                aic,
                                bic,
                                neg_log_lik: fit.neg_log_lik,
                                converged: true,
                            })
                        }
                        Err(_) => None,
                    }
                })
                .collect();

            // Find best for this fold
            fold_grid
                .into_iter()
                .min_by(|a, b| {
                    let va = if use_aic { a.aic } else { a.bic };
                    let vb = if use_aic { b.aic } else { b.bic };
                    va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
                })
        })
        .collect();

    // Convert to Python list of dicts
    let py_results: Vec<PyObject> = fold_results
        .iter()
        .enumerate()
        .map(|(i, opt)| {
            let d = PyDict::new(py);
            let _ = d.set_item("fold", i);
            match opt {
                Some(r) => {
                    let _ = d.set_item("arma_p", r.arma_p);
                    let _ = d.set_item("arma_q", r.arma_q);
                    let _ = d.set_item("garch_m", r.garch_m);
                    let _ = d.set_item("garch_s", r.garch_s);
                    let _ = d.set_item("aic", r.aic);
                    let _ = d.set_item("bic", r.bic);
                    let _ = d.set_item("converged", true);
                }
                None => {
                    let _ = d.set_item("converged", false);
                }
            }
            d.into()
        })
        .collect();

    Ok(py_results.into_pyobject(py)?.into())
}

// =============================================================================
// Module registration
// =============================================================================

/// Python module: arma_garch_cv
#[cfg(feature = "python")]
#[pymodule]
fn arma_garch_cv(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fit_arma_garch, m)?)?;
    m.add_function(wrap_pyfunction!(grid_search_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_variance_forecast, m)?)?;
    m.add_function(wrap_pyfunction!(cpcv_grid_search, m)?)?;
    Ok(())
}
