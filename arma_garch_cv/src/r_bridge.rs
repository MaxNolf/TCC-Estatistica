//! R bridge module — exposes Rust functions to R via `extendr`.
//!
//! ## Funções exportadas para R:
//! - `arma_garch_purged_cv()` — Executa a CV completa e retorna uma lista R
//! - `arma_garch_fit_single()` — Ajusta um modelo ARMA-GARCH e retorna parâmetros

use extendr_api::prelude::*;
use crate::cv::purged::run_purged_cv;
use crate::estimation::solver::garch_fit;
use crate::spec::*;

/// Executa a validação cruzada temporal purged ARMA-GARCH.
///
/// @param data Vetor numérico de retornos (numeric vector)
/// @param arma_p Ordem AR (integer)
/// @param arma_q Ordem MA (integer)
/// @param garch_m Ordem ARCH (integer)
/// @param garch_s Ordem GARCH (integer)
/// @param window_size Tamanho da janela de treino (integer)
/// @param test_size Tamanho da janela de teste (integer)
/// @param step_size Passo entre folds (integer)
/// @param purge_size Gap de purga entre treino e teste (integer)
/// @param parallel Usar paralelismo via rayon? (logical)
/// @return Lista R com métricas, previsões e diagnósticos
/// @export
#[extendr]
fn arma_garch_purged_cv(
    data: &[f64],
    arma_p: i32,
    arma_q: i32,
    garch_m: i32,
    garch_s: i32,
    window_size: i32,
    test_size: i32,
    step_size: i32,
    purge_size: i32,
    parallel: bool,
) -> Robj {
    let spec = ModelSpec::new(
        ArmaOrder::new(arma_p as usize, arma_q as usize),
        GarchOrder::new(garch_m as usize, garch_s as usize),
        Distribution::Normal,
        Solver::LBFGS,
    );
    let cv_params = CvParams::new(
        window_size as usize,
        test_size as usize,
        step_size as usize,
        purge_size as usize,
    );

    match run_purged_cv(data, &spec, &cv_params, parallel, Verbosity::Silent) {
        Ok(result) => {
            // Extrair vetores de previsões
            let folds: Vec<i32> = result.predictions.iter().map(|r| r.fold as i32).collect();
            let times: Vec<i32> = result.predictions.iter().map(|r| r.time as i32).collect();
            let actuals: Vec<f64> = result.predictions.iter().map(|r| r.actual).collect();
            let predicted: Vec<f64> = result.predictions.iter().map(|r| r.predicted).collect();

            // Construir a lista R
            list!(
                predictions = list!(
                    fold = folds,
                    time = times,
                    actual = actuals,
                    predicted = predicted
                ),
                metrics = list!(
                    rmse = result.metrics.rmse,
                    mae = result.metrics.mae,
                    medae = result.metrics.medae,
                    mape = result.metrics.mape.unwrap_or(f64::NAN),
                    me = result.metrics.me
                ),
                diagnostics = list!(
                    n_folds = result.diagnostics.n_folds as i32,
                    n_converged = result.diagnostics.n_converged as i32,
                    convergence_rate = result.diagnostics.convergence_rate,
                    total_time_secs = result.diagnostics.total_time_secs
                ),
                model = format!("{}", result.model_spec),
                error = Robj::from(NULL)
            )
            .into()
        }
        Err(e) => {
            list!(
                predictions = Robj::from(NULL),
                metrics = Robj::from(NULL),
                diagnostics = Robj::from(NULL),
                model = Robj::from(NULL),
                error = format!("{}", e)
            )
            .into()
        }
    }
}

/// Ajusta um único modelo ARMA-GARCH e retorna os parâmetros estimados.
///
/// @param data Vetor numérico de retornos (numeric vector)
/// @param arma_p Ordem AR (integer)
/// @param arma_q Ordem MA (integer)
/// @param garch_m Ordem ARCH (integer)
/// @param garch_s Ordem GARCH (integer)
/// @return Lista R com parâmetros estimados, log-likelihood, resíduos e variâncias
/// @export
#[extendr]
fn arma_garch_fit_single(
    data: &[f64],
    arma_p: i32,
    arma_q: i32,
    garch_m: i32,
    garch_s: i32,
) -> Robj {
    let spec = ModelSpec::new(
        ArmaOrder::new(arma_p as usize, arma_q as usize),
        GarchOrder::new(garch_m as usize, garch_s as usize),
        Distribution::Normal,
        Solver::LBFGS,
    );

    match garch_fit(&spec, data) {
        Ok(fit) => {
            // Nomear os parâmetros
            let p = spec.arma_order.p;
            let q = spec.arma_order.q;
            let m = spec.garch_order.m;
            let s = spec.garch_order.s;

            let mut names = Vec::new();
            names.push("mu".to_string());
            for i in 1..=p { names.push(format!("ar{}", i)); }
            for j in 1..=q { names.push(format!("ma{}", j)); }
            names.push("omega".to_string());
            for i in 1..=m { names.push(format!("alpha{}", i)); }
            for j in 1..=s { names.push(format!("beta{}", j)); }

            list!(
                params = fit.params.clone(),
                param_names = names,
                neg_log_lik = fit.neg_log_lik,
                residuals = fit.residuals.clone(),
                conditional_variances = fit.conditional_variances.clone(),
                sigma = fit.conditional_variances.iter().map(|v| v.sqrt()).collect::<Vec<f64>>(),
                error = Robj::from(NULL)
            )
            .into()
        }
        Err(e) => {
            list!(
                params = Robj::from(NULL),
                param_names = Robj::from(NULL),
                neg_log_lik = Robj::from(NULL),
                residuals = Robj::from(NULL),
                conditional_variances = Robj::from(NULL),
                sigma = Robj::from(NULL),
                error = format!("{}", e)
            )
            .into()
        }
    }
}

/// Gera previsões n passos à frente a partir de dados de treino.
///
/// @param data Vetor numérico de retornos do treino (numeric vector)
/// @param arma_p Ordem AR (integer)
/// @param arma_q Ordem MA (integer)
/// @param garch_m Ordem ARCH (integer)
/// @param garch_s Ordem GARCH (integer)
/// @param n_ahead Número de passos à frente (integer)
/// @return Lista R com forecasts de média e variância
/// @export
#[extendr]
fn arma_garch_forecast(
    data: &[f64],
    arma_p: i32,
    arma_q: i32,
    garch_m: i32,
    garch_s: i32,
    n_ahead: i32,
) -> Robj {
    let spec = ModelSpec::new(
        ArmaOrder::new(arma_p as usize, arma_q as usize),
        GarchOrder::new(garch_m as usize, garch_s as usize),
        Distribution::Normal,
        Solver::LBFGS,
    );

    match garch_fit(&spec, data) {
        Ok(fit) => {
            match fit.forecast(n_ahead as usize) {
                Ok(forecasts) => {
                    list!(
                        mean_forecast = forecasts,
                        error = Robj::from(NULL)
                    )
                    .into()
                }
                Err(e) => {
                    list!(
                        mean_forecast = Robj::from(NULL),
                        error = format!("{}", e)
                    )
                    .into()
                }
            }
        }
        Err(e) => {
            list!(
                mean_forecast = Robj::from(NULL),
                error = format!("{}", e)
            )
            .into()
        }
    }
}

// Registrar módulo para R
extendr_module! {
    mod armgarchcv;
    fn arma_garch_purged_cv;
    fn arma_garch_fit_single;
    fn arma_garch_forecast;
}
