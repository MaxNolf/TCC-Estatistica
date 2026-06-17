//! Purged temporal cross-validation (walk-forward).
//!
//! Implements the CV loop with:
//! - Vectorized fold index computation
//! - Sequential execution with progress bar
//! - Parallel execution with `rayon`

use std::time::Instant;

use rayon::prelude::*;

use crate::cv::metrics::compute_metrics;
use crate::estimation::solver::garch_fit;
use crate::progress::{create_progress_bar, print_header, print_summary};
use crate::spec::*;

// =============================================================================
// Fold Index Computation
// =============================================================================

/// Computes all fold indices for purged walk-forward CV.
///
/// Mirrors the R implementation (lines 103–133):
/// - `train_start[i] = i * step_size`
/// - `train_end[i] = train_start[i] + window_size - 1`
/// - `test_start[i] = train_end[i] + purge_size + 1`
/// - `test_end[i] = test_start[i] + test_size - 1`
///
/// Only returns folds where `test_end < n`.
pub fn compute_fold_indices(n: usize, params: &CvParams) -> Result<Vec<FoldIndices>, CvError> {
    let max_train_start = n
        .checked_sub(params.window_size + params.purge_size + params.test_size)
        .ok_or(CvError::NoFoldsPossible)?;

    let start_indices: Vec<usize> = (0..=max_train_start)
        .step_by(params.step_size)
        .collect();

    if start_indices.is_empty() {
        return Err(CvError::NoFoldsPossible);
    }

    let folds: Vec<FoldIndices> = start_indices
        .iter()
        .enumerate()
        .filter_map(|(id, &ts)| {
            let te = ts + params.window_size - 1;
            let tts = te + params.purge_size + 1;
            let tte = tts + params.test_size - 1;

            if tte < n {
                Some(FoldIndices {
                    fold_id: id,
                    train_start: ts,
                    train_end: te,
                    test_start: tts,
                    test_end: tte,
                })
            } else {
                None
            }
        })
        .collect();

    if folds.is_empty() {
        return Err(CvError::NoFoldsPossible);
    }

    Ok(folds)
}

// =============================================================================
// Single Fold Execution
// =============================================================================

/// Fits a single fold: train → fit → forecast → compare.
fn fit_single_fold(
    data: &[f64],
    fold: &FoldIndices,
    spec: &ModelSpec,
) -> FoldResult {
    let train_data = &data[fold.train_start..=fold.train_end];
    let test_data = &data[fold.test_start..=fold.test_end];
    let time_indices: Vec<usize> = (fold.test_start..=fold.test_end).collect();

    // Attempt model fit
    let fit = match garch_fit(spec, train_data) {
        Ok(f) => f,
        Err(_) => return FoldResult::failed(test_data.to_vec(), time_indices),
    };

    // Attempt forecast
    let n_ahead = test_data.len();
    let (predictions, vars) = match fit.forecast(n_ahead) {
        Ok((preds, vs)) => (preds, vs),
        Err(_) => return FoldResult::failed(test_data.to_vec(), time_indices),
    };
    
    let sigmas = vars.into_iter().map(|v| v.sqrt()).collect();

    FoldResult::success(predictions, sigmas, test_data.to_vec(), time_indices)
}

// =============================================================================
// CV Execution
// =============================================================================

/// Main entry point: runs purged temporal cross-validation.
///
/// This is the Rust equivalent of R's `arma_garch_purged_cv()`.
///
/// # Arguments
/// * `data` — Time series as a contiguous f64 slice
/// * `spec` — Model specification (ARMA + GARCH orders, distribution, solver)
/// * `cv_params` — CV parameters (window, test, step, purge sizes)
/// * `parallel` — Whether to use rayon parallelization
/// * `verbosity` — Output verbosity level
///
/// # Returns
/// `CvResult` with predictions, metrics, and diagnostics.
pub fn run_purged_cv(
    data: &[f64],
    spec: &ModelSpec,
    cv_params: &CvParams,
    parallel: bool,
    verbosity: Verbosity,
) -> Result<CvResult, CvError> {
    // ── 1. Validate inputs ────────────────────────────────────────────────
    validate_inputs(data, cv_params)?;

    // ── 2. Compute fold indices ───────────────────────────────────────────
    let folds = compute_fold_indices(data.len(), cv_params)?;
    let n_folds = folds.len();

    // ── 3. Print header ───────────────────────────────────────────────────
    if verbosity != Verbosity::Silent {
        print_header(
            spec.arma_order.p,
            spec.arma_order.q,
            spec.garch_order.m,
            spec.garch_order.s,
            &spec.distribution.to_string(),
            n_folds,
            &spec.solver.to_string(),
            parallel,
        );
    }

    // ── 4. Execute CV ─────────────────────────────────────────────────────
    let t_start = Instant::now();

    let (fold_results, fold_times_opt) = if parallel {
        // ── Parallel via rayon ──
        let results: Vec<FoldResult> = folds
            .par_iter()
            .map(|fold| fit_single_fold(data, fold, spec))
            .collect();
        (results, None)
    } else {
        // ── Sequential with progress ──
        let pb = create_progress_bar(n_folds, verbosity);
        let mut fold_times = Vec::with_capacity(n_folds);
        let mut n_ok: usize = 0;
        let mut n_fail: usize = 0;

        let results: Vec<FoldResult> = folds
            .iter()
            .map(|fold| {
                let t_fold = Instant::now();
                let res = fit_single_fold(data, fold, spec);
                let fold_elapsed = t_fold.elapsed().as_secs_f64();
                fold_times.push(fold_elapsed);

                if res.converged {
                    n_ok += 1;
                } else {
                    n_fail += 1;
                }

                let icon = if res.converged { "✔" } else { "✘" };
                pb.set_message(format!("{} OK:{} Falha:{}", icon, n_ok, n_fail));
                pb.inc(1);

                res
            })
            .collect();

        pb.finish_and_clear();
        (results, Some(fold_times))
    };

    let total_time = t_start.elapsed().as_secs_f64();

    // ── 5. Consolidate results ────────────────────────────────────────────
    let mut predictions: Vec<PredictionRow> = Vec::new();
    let mut all_actuals: Vec<f64> = Vec::new();
    let mut all_predicted: Vec<f64> = Vec::new();
    let mut n_converged: usize = 0;

    for (i, res) in fold_results.iter().enumerate() {
        if res.converged {
            n_converged += 1;
            for j in 0..res.predictions.len() {
                if !res.predictions[j].is_nan() {
                    predictions.push(PredictionRow {
                        fold: i,
                        time: res.time_indices[j],
                        actual: res.actuals[j],
                        predicted: res.predictions[j],
                        sigma: res.sigmas[j],
                    });
                    all_actuals.push(res.actuals[j]);
                    all_predicted.push(res.predictions[j]);
                }
            }
        }
    }

    if predictions.is_empty() {
        return Err(CvError::NoPredictions);
    }

    // ── 6. Compute metrics ────────────────────────────────────────────────
    let metrics = compute_metrics(&all_actuals, &all_predicted)?;

    // ── 7. Diagnostics ────────────────────────────────────────────────────
    let conv_rate = n_converged as f64 / n_folds as f64 * 100.0;
    let avg_fold_time = fold_times_opt
        .as_ref()
        .map(|ft| ft.iter().sum::<f64>() / ft.len() as f64);

    let diagnostics = Diagnostics {
        convergence_rate: conv_rate,
        total_time_secs: total_time,
        avg_time_per_fold: avg_fold_time,
        fold_times: fold_times_opt,
        n_folds,
        n_converged,
    };

    // ── 8. Print summary ──────────────────────────────────────────────────
    if verbosity != Verbosity::Silent {
        print_summary(
            spec.arma_order.p,
            spec.arma_order.q,
            spec.garch_order.m,
            spec.garch_order.s,
            &spec.distribution.to_string(),
            &spec.solver.to_string(),
            cv_params.window_size,
            cv_params.test_size,
            cv_params.purge_size,
            n_folds,
            n_converged,
            conv_rate,
            total_time,
            metrics.rmse,
            metrics.mae,
            metrics.medae,
            metrics.mape,
            metrics.me,
        );
    }

    // ── 9. Return ─────────────────────────────────────────────────────────
    Ok(CvResult {
        predictions,
        metrics,
        model_spec: spec.clone(),
        cv_params: cv_params.clone(),
        diagnostics,
    })
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_fold_indices_basic() {
        // n=20, window=10, test=1, step=1, purge=1
        let params = CvParams::new(10, 1, 1, 1);
        let folds = compute_fold_indices(20, &params).unwrap();

        // First fold: train [0..9], skip 10, test [11]
        assert_eq!(folds[0].train_start, 0);
        assert_eq!(folds[0].train_end, 9);
        assert_eq!(folds[0].test_start, 11);
        assert_eq!(folds[0].test_end, 11);

        // All folds should have test_end < 20
        for f in &folds {
            assert!(f.test_end < 20);
        }
    }

    #[test]
    fn test_compute_fold_indices_step_size() {
        let params = CvParams::new(10, 1, 3, 1);
        let folds = compute_fold_indices(30, &params).unwrap();

        // Check step spacing
        if folds.len() >= 2 {
            let delta = folds[1].train_start - folds[0].train_start;
            assert_eq!(delta, 3);
        }
    }

    #[test]
    fn test_compute_fold_indices_purge() {
        let params = CvParams::new(10, 1, 1, 5);
        let folds = compute_fold_indices(30, &params).unwrap();

        // Gap between train_end and test_start should be purge_size + 1
        for f in &folds {
            assert_eq!(f.test_start - f.train_end, 6); // purge=5, so gap = 5+1
        }
    }

    #[test]
    fn test_compute_fold_indices_test_size() {
        let params = CvParams::new(10, 3, 1, 1);
        let folds = compute_fold_indices(30, &params).unwrap();

        for f in &folds {
            let test_len = f.test_end - f.test_start + 1;
            assert_eq!(test_len, 3);
        }
    }

    #[test]
    fn test_compute_fold_indices_insufficient() {
        // Too small for any fold
        let params = CvParams::new(10, 5, 1, 5);
        let result = compute_fold_indices(15, &params);
        assert!(result.is_err());
    }

    #[test]
    fn test_compute_fold_indices_exact_fit() {
        // n=13, window=10, test=1, step=1, purge=1
        // max_train_start = 13 - 10 - 1 - 1 = 1
        // start_indices = [0, 1]
        // fold 0: train [0..9], test [11], test_end=11 < 13 ✓
        // fold 1: train [1..10], test [12], test_end=12 < 13 ✓
        let params = CvParams::new(10, 1, 1, 1);
        let folds = compute_fold_indices(13, &params).unwrap();
        assert_eq!(folds.len(), 2);
    }
}
