//! Error metrics for cross-validation results.
//!
//! Implements RMSE, MAE, MedAE, MAPE, and ME (bias) using
//! vectorized iterator operations that compile to SIMD instructions.

use crate::spec::{CvError, Metrics};

/// Computes all error metrics from actual vs predicted vectors.
///
/// # Arguments
/// * `actual` — Observed values
/// * `predicted` — Model predictions (must be same length as `actual`)
///
/// # Returns
/// `Metrics` struct with RMSE, MAE, MedAE, MAPE, ME.
///
/// # Errors
/// Returns `CvError::NoPredictions` if the input slices are empty.
pub fn compute_metrics(actual: &[f64], predicted: &[f64]) -> Result<Metrics, CvError> {
    let n = actual.len();
    if n == 0 || predicted.len() != n {
        return Err(CvError::NoPredictions);
    }

    let n_f = n as f64;

    // ── Errors ────────────────────────────────────────────────────────────
    let errors: Vec<f64> = actual
        .iter()
        .zip(predicted.iter())
        .map(|(a, p)| a - p)
        .collect();

    // ── RMSE ──────────────────────────────────────────────────────────────
    let mse = errors.iter().map(|e| e * e).sum::<f64>() / n_f;
    let rmse = mse.sqrt();

    // ── MAE ───────────────────────────────────────────────────────────────
    let abs_errors: Vec<f64> = errors.iter().map(|e| e.abs()).collect();
    let mae = abs_errors.iter().sum::<f64>() / n_f;

    // ── MedAE ─────────────────────────────────────────────────────────────
    let medae = median(&abs_errors);

    // ── MAPE ──────────────────────────────────────────────────────────────
    // Protected against division by zero: only uses observations where
    // |actual| > machine epsilon.
    let mape = {
        let eps = f64::EPSILON;
        let valid_pairs: Vec<(f64, f64)> = actual
            .iter()
            .zip(errors.iter())
            .filter(|(a, _)| a.abs() > eps)
            .map(|(a, e)| (*a, *e))
            .collect();

        if valid_pairs.is_empty() {
            None
        } else {
            let sum_ape: f64 = valid_pairs
                .iter()
                .map(|(a, e)| (e.abs()) / a.abs())
                .sum();
            Some(sum_ape / valid_pairs.len() as f64 * 100.0)
        }
    };

    // ── ME (bias) ─────────────────────────────────────────────────────────
    let me = errors.iter().sum::<f64>() / n_f;

    Ok(Metrics {
        rmse,
        mae,
        medae,
        mape,
        me,
    })
}

/// Computes the median of a slice. Returns 0.0 for empty slices.
fn median(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: assert floats are approximately equal.
    fn assert_approx(a: f64, b: f64, tol: f64) {
        assert!(
            (a - b).abs() < tol,
            "expected ~{}, got {} (diff: {})",
            b,
            a,
            (a - b).abs()
        );
    }

    #[test]
    fn test_perfect_predictions() {
        let actual = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let predicted = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let m = compute_metrics(&actual, &predicted).unwrap();
        assert_approx(m.rmse, 0.0, 1e-12);
        assert_approx(m.mae, 0.0, 1e-12);
        assert_approx(m.medae, 0.0, 1e-12);
        assert_approx(m.me, 0.0, 1e-12);
        assert_approx(m.mape.unwrap(), 0.0, 1e-12);
    }

    #[test]
    fn test_known_errors() {
        // actual = [10, 20, 30], predicted = [12, 18, 33]
        // errors = [-2, 2, -3]
        let actual = vec![10.0, 20.0, 30.0];
        let predicted = vec![12.0, 18.0, 33.0];
        let m = compute_metrics(&actual, &predicted).unwrap();

        // MSE = (4 + 4 + 9) / 3 = 17/3
        assert_approx(m.rmse, (17.0_f64 / 3.0).sqrt(), 1e-10);

        // MAE = (2 + 2 + 3) / 3 = 7/3
        assert_approx(m.mae, 7.0 / 3.0, 1e-10);

        // MedAE = median([2, 2, 3]) = 2.0
        assert_approx(m.medae, 2.0, 1e-10);

        // ME = (-2 + 2 + -3) / 3 = -1.0
        assert_approx(m.me, -1.0, 1e-10);

        // MAPE = mean(|e/a|) * 100 = mean(0.2, 0.1, 0.1) * 100 = 13.33...%
        assert_approx(m.mape.unwrap(), 100.0 * (0.2 + 0.1 + 0.1) / 3.0, 1e-8);
    }

    #[test]
    fn test_mape_with_zero_actuals() {
        let actual = vec![0.0, 0.0, 0.0];
        let predicted = vec![1.0, 2.0, 3.0];
        let m = compute_metrics(&actual, &predicted).unwrap();
        // All actuals are zero -> MAPE should be None
        assert!(m.mape.is_none());
    }

    #[test]
    fn test_empty_input() {
        let result = compute_metrics(&[], &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_median_odd() {
        assert_approx(median(&[3.0, 1.0, 2.0]), 2.0, 1e-12);
    }

    #[test]
    fn test_median_even() {
        assert_approx(median(&[4.0, 1.0, 3.0, 2.0]), 2.5, 1e-12);
    }

    #[test]
    fn test_median_single() {
        assert_approx(median(&[42.0]), 42.0, 1e-12);
    }

    #[test]
    fn test_single_observation() {
        let m = compute_metrics(&[5.0], &[3.0]).unwrap();
        assert_approx(m.rmse, 2.0, 1e-12);
        assert_approx(m.mae, 2.0, 1e-12);
        assert_approx(m.me, 2.0, 1e-12);
    }
}
