//! Core type definitions for the ARMA-GARCH cross-validation system.

use serde::{Deserialize, Serialize};
use std::fmt;
use thiserror::Error;

// =============================================================================
// Errors
// =============================================================================

/// Errors that can occur during cross-validation.
#[derive(Debug, Error)]
pub enum CvError {
    #[error("Dados insuficientes: {0} observações (mínimo: 10)")]
    InsufficientData(usize),

    #[error("'{name}' deve ser vetor de 2 elementos >= 0 (recebido: {value:?})")]
    InvalidOrder { name: String, value: (usize, usize) },

    #[error("window_size ({window}) deve ser > 0 e < n ({n})")]
    InvalidWindowSize { window: usize, n: usize },

    #[error("test_size deve ser > 0 (recebido: {0})")]
    InvalidTestSize(usize),

    #[error("step_size deve ser > 0 (recebido: {0})")]
    InvalidStepSize(usize),

    #[error("Nenhum fold possível com os parâmetros fornecidos")]
    NoFoldsPossible,

    #[error("Nenhuma previsão válida gerada. Verifique os parâmetros ou a convergência.")]
    NoPredictions,

    #[error("Erro na estimação do modelo: {0}")]
    EstimationError(String),

    #[error("Erro no forecast: {0}")]
    ForecastError(String),

    #[error("Erro no gráfico: {0}")]
    PlotError(String),
}

// =============================================================================
// Model Orders
// =============================================================================

/// ARMA order (p, q) — AR and MA orders.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct ArmaOrder {
    /// AR order (p)
    pub p: usize,
    /// MA order (q)
    pub q: usize,
}

impl ArmaOrder {
    pub fn new(p: usize, q: usize) -> Self {
        Self { p, q }
    }
}

impl fmt::Display for ArmaOrder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ARMA({},{})", self.p, self.q)
    }
}

/// GARCH order (m, s) — ARCH and GARCH orders.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct GarchOrder {
    /// ARCH order (m)
    pub m: usize,
    /// GARCH order (s)
    pub s: usize,
}

impl GarchOrder {
    pub fn new(m: usize, s: usize) -> Self {
        Self { m, s }
    }
}

impl fmt::Display for GarchOrder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GARCH({},{})", self.m, self.s)
    }
}

// =============================================================================
// Enums
// =============================================================================

/// Conditional distribution for the GARCH residuals.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Distribution {
    /// Normal (Gaussian)
    Normal,
    /// Student-t
    StudentT,
    /// Generalized Error Distribution
    GED,
    /// Skewed Normal
    SkewedNormal,
    /// Skewed Student-t
    SkewedStudentT,
    /// Skewed GED
    SkewedGED,
    /// Johnson's SU
    JSU,
}

impl fmt::Display for Distribution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Distribution::Normal => write!(f, "norm"),
            Distribution::StudentT => write!(f, "std"),
            Distribution::GED => write!(f, "ged"),
            Distribution::SkewedNormal => write!(f, "snorm"),
            Distribution::SkewedStudentT => write!(f, "sstd"),
            Distribution::SkewedGED => write!(f, "sged"),
            Distribution::JSU => write!(f, "jsu"),
        }
    }
}

impl Default for Distribution {
    fn default() -> Self {
        Distribution::Normal
    }
}

/// Optimization solver for model fitting.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Solver {
    /// Combination of multiple solvers
    Hybrid,
    /// Sequential quadratic programming
    Solnp,
    /// Global solver (stochastic)
    Gosolnp,
    /// Quasi-Newton method
    Nlminb,
    /// Limited-memory BFGS
    LBFGS,
}

impl fmt::Display for Solver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Solver::Hybrid => write!(f, "hybrid"),
            Solver::Solnp => write!(f, "solnp"),
            Solver::Gosolnp => write!(f, "gosolnp"),
            Solver::Nlminb => write!(f, "nlminb"),
            Solver::LBFGS => write!(f, "lbfgs"),
        }
    }
}

impl Default for Solver {
    fn default() -> Self {
        Solver::Hybrid
    }
}

/// Verbosity level for progress output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verbosity {
    /// No output
    Silent,
    /// Progress bar (compact)
    ProgressBar,
    /// Detailed output per fold
    Detailed,
}

impl From<u8> for Verbosity {
    fn from(v: u8) -> Self {
        match v {
            0 => Verbosity::Silent,
            1 => Verbosity::ProgressBar,
            _ => Verbosity::Detailed,
        }
    }
}

// =============================================================================
// Cross-Validation Parameters
// =============================================================================

/// Parameters for purged temporal cross-validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CvParams {
    /// Training window size
    pub window_size: usize,
    /// Forecast horizon per fold
    pub test_size: usize,
    /// Step between folds
    pub step_size: usize,
    /// Gap between train and test (purge)
    pub purge_size: usize,
}

impl CvParams {
    pub fn new(window_size: usize, test_size: usize, step_size: usize, purge_size: usize) -> Self {
        Self {
            window_size,
            test_size,
            step_size,
            purge_size,
        }
    }

    /// Creates default CvParams for a given data length.
    /// window_size = floor(n / 3), test/step/purge = 1.
    pub fn default_for_length(n: usize) -> Self {
        Self {
            window_size: n / 3,
            test_size: 1,
            step_size: 1,
            purge_size: 1,
        }
    }
}

// =============================================================================
// Model Specification
// =============================================================================

/// Full model specification for ARMA-GARCH.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSpec {
    pub arma_order: ArmaOrder,
    pub garch_order: GarchOrder,
    pub distribution: Distribution,
    pub solver: Solver,
    pub include_mean: bool,
}

impl ModelSpec {
    pub fn new(
        arma_order: ArmaOrder,
        garch_order: GarchOrder,
        distribution: Distribution,
        solver: Solver,
    ) -> Self {
        Self {
            arma_order,
            garch_order,
            distribution,
            solver,
            include_mean: true,
        }
    }
}

impl fmt::Display for ModelSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}-{} | Dist: {} | Solver: {}",
            self.arma_order, self.garch_order, self.distribution, self.solver
        )
    }
}

// =============================================================================
// Fold Indices
// =============================================================================

/// Pre-computed indices for a single CV fold.
#[derive(Debug, Clone)]
pub struct FoldIndices {
    pub fold_id: usize,
    pub train_start: usize,
    pub train_end: usize,
    pub test_start: usize,
    pub test_end: usize,
}

// =============================================================================
// Fold Result
// =============================================================================

/// Result of fitting and forecasting a single fold.
#[derive(Debug, Clone)]
pub struct FoldResult {
    /// Predicted values
    pub predictions: Vec<f64>,
    /// Predicted sigmas (conditional volatilities)
    pub sigmas: Vec<f64>,
    /// Actual values (test set)
    pub actuals: Vec<f64>,
    /// Time indices of test set
    pub time_indices: Vec<usize>,
    /// Whether the model converged
    pub converged: bool,
}

impl FoldResult {
    /// Creates a successful fold result.
    pub fn success(predictions: Vec<f64>, sigmas: Vec<f64>, actuals: Vec<f64>, time_indices: Vec<usize>) -> Self {
        Self {
            predictions,
            sigmas,
            actuals,
            time_indices,
            converged: true,
        }
    }

    /// Creates a failed fold result (NaN predictions).
    pub fn failed(actuals: Vec<f64>, time_indices: Vec<usize>) -> Self {
        let n = actuals.len();
        Self {
            predictions: vec![f64::NAN; n],
            sigmas: vec![f64::NAN; n],
            actuals,
            time_indices,
            converged: false,
        }
    }
}

// =============================================================================
// Prediction Row
// =============================================================================

/// A single prediction row in the output DataFrame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionRow {
    pub fold: usize,
    pub time: usize,
    pub actual: f64,
    pub predicted: f64,
    pub sigma: f64,
}

// =============================================================================
// Metrics
// =============================================================================

/// Error metrics for the cross-validation results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metrics {
    /// Root Mean Squared Error
    pub rmse: f64,
    /// Mean Absolute Error
    pub mae: f64,
    /// Median Absolute Error
    pub medae: f64,
    /// Mean Absolute Percentage Error (%)
    pub mape: Option<f64>,
    /// Mean Error (bias)
    pub me: f64,
}

impl fmt::Display for Metrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "  RMSE:   {:.6}", self.rmse)?;
        writeln!(f, "  MAE:    {:.6}", self.mae)?;
        writeln!(f, "  MedAE:  {:.6}", self.medae)?;
        match self.mape {
            Some(mape) => writeln!(f, "  MAPE:   {:.2}%", mape)?,
            None => writeln!(f, "  MAPE:   N/A")?,
        }
        write!(f, "  ME:     {:.6} (viés)", self.me)
    }
}

// =============================================================================
// Diagnostics
// =============================================================================

/// Diagnostic information about the CV run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Diagnostics {
    /// Convergence rate (%)
    pub convergence_rate: f64,
    /// Total elapsed time (seconds)
    pub total_time_secs: f64,
    /// Average time per fold (seconds), None if parallel
    pub avg_time_per_fold: Option<f64>,
    /// Per-fold times (seconds), None if parallel
    pub fold_times: Option<Vec<f64>>,
    /// Total number of folds
    pub n_folds: usize,
    /// Number of converged folds
    pub n_converged: usize,
}

// =============================================================================
// CV Result (Final Output)
// =============================================================================

/// Complete result of the purged cross-validation.
#[derive(Debug, Clone)]
pub struct CvResult {
    /// Prediction rows (only converged folds)
    pub predictions: Vec<PredictionRow>,
    /// Error metrics
    pub metrics: Metrics,
    /// Model specification used
    pub model_spec: ModelSpec,
    /// CV parameters used
    pub cv_params: CvParams,
    /// Diagnostic information
    pub diagnostics: Diagnostics,
}

// =============================================================================
// Input validation
// =============================================================================

/// Validates all inputs before running CV.
pub fn validate_inputs(
    data: &[f64],
    cv_params: &CvParams,
) -> Result<(), CvError> {
    let n = data.len();

    // Min length
    if n < 10 {
        return Err(CvError::InsufficientData(n));
    }

    // Window size
    if cv_params.window_size == 0 || cv_params.window_size >= n {
        return Err(CvError::InvalidWindowSize {
            window: cv_params.window_size,
            n,
        });
    }

    // Test size
    if cv_params.test_size == 0 {
        return Err(CvError::InvalidTestSize(cv_params.test_size));
    }

    // Step size
    if cv_params.step_size == 0 {
        return Err(CvError::InvalidStepSize(cv_params.step_size));
    }

    Ok(())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arma_order_display() {
        let order = ArmaOrder::new(1, 1);
        assert_eq!(format!("{}", order), "ARMA(1,1)");
    }

    #[test]
    fn test_garch_order_display() {
        let order = GarchOrder::new(1, 1);
        assert_eq!(format!("{}", order), "GARCH(1,1)");
    }

    #[test]
    fn test_distribution_display() {
        assert_eq!(format!("{}", Distribution::Normal), "norm");
        assert_eq!(format!("{}", Distribution::StudentT), "std");
        assert_eq!(format!("{}", Distribution::JSU), "jsu");
    }

    #[test]
    fn test_model_spec_display() {
        let spec = ModelSpec::new(
            ArmaOrder::new(1, 1),
            GarchOrder::new(1, 1),
            Distribution::Normal,
            Solver::Hybrid,
        );
        assert_eq!(
            format!("{}", spec),
            "ARMA(1,1)-GARCH(1,1) | Dist: norm | Solver: hybrid"
        );
    }

    #[test]
    fn test_cv_params_default() {
        let params = CvParams::default_for_length(300);
        assert_eq!(params.window_size, 100);
        assert_eq!(params.test_size, 1);
        assert_eq!(params.step_size, 1);
        assert_eq!(params.purge_size, 1);
    }

    #[test]
    fn test_validate_inputs_insufficient_data() {
        let data = vec![1.0; 5];
        let params = CvParams::new(3, 1, 1, 1);
        assert!(matches!(
            validate_inputs(&data, &params),
            Err(CvError::InsufficientData(5))
        ));
    }

    #[test]
    fn test_validate_inputs_invalid_window() {
        let data = vec![1.0; 100];
        let params = CvParams::new(0, 1, 1, 1);
        assert!(matches!(
            validate_inputs(&data, &params),
            Err(CvError::InvalidWindowSize { .. })
        ));
    }

    #[test]
    fn test_validate_inputs_ok() {
        let data = vec![1.0; 100];
        let params = CvParams::new(30, 1, 1, 1);
        assert!(validate_inputs(&data, &params).is_ok());
    }

    #[test]
    fn test_fold_result_success() {
        let result = FoldResult::success(
            vec![1.0, 2.0],
            vec![0.1, 0.2],
            vec![1.1, 1.9],
            vec![50, 51],
        );
        assert!(result.converged);
        assert_eq!(result.predictions.len(), 2);
    }

    #[test]
    fn test_fold_result_failed() {
        let result = FoldResult::failed(vec![1.0, 2.0], vec![50, 51]);
        assert!(!result.converged);
        assert!(result.predictions.iter().all(|x| x.is_nan()));
    }
}
