//! Model solver — fits ARMA-GARCH parameters via numerical optimization.
//!
//! ## Visão Geral
//!
//! O solver recebe uma série temporal e uma especificação de modelo, e retorna
//! os parâmetros estimados por MLE. O processo é:
//!
//! ```text
//! Série y_t → Chutes Iniciais θ₀ → Otimizador (L-BFGS / Nelder-Mead) → θ̂ (MLE)
//! ```
//!
//! ## Integração com `argmin`
//!
//! O crate `argmin` exige que o problema implemente o trait `CostFunction`:
//!
//! ```text
//! impl CostFunction for GarchProblem {
//!     type Param = Vec<f64>;    // vetor de parâmetros
//!     type Output = f64;        // valor da função custo (neg log-lik)
//!
//!     fn cost(&self, params: &Vec<f64>) -> Result<f64, argmin::core::Error> {
//!         Ok(self.likelihood.neg_log_likelihood(params, &self.data))
//!     }
//! }
//! ```
//!
//! ## Escolha do Otimizador
//!
//! - **L-BFGS** (`argmin::solver::linesearch::...`): rápido, precisa de gradiente
//!   - Gradiente pode ser numérico (diferenças finitas) via `argmin`
//!   - Recomendado para GARCH por ser rápido e robusto
//! - **Nelder-Mead** (`argmin::solver::neldermead::NelderMead`): sem gradiente
//!   - Mais lento, mas funciona quando L-BFGS falha
//!   - Usado como fallback no modo "hybrid"

use crate::spec::{CvError, ModelSpec};
use crate::estimation::likelihood::{LogLikelihood, SGarchLikelihood};
use argmin::core::{CostFunction, Error, Executor, Gradient, State};
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;
use argmin::solver::neldermead::NelderMead;

/// Result of a successful GARCH model fit.
///
/// Contains estimated parameters and can produce forecasts.
#[derive(Debug, Clone)]
pub struct GarchFit {
    /// The model specification used
    pub spec: ModelSpec,
    /// Estimated parameters: [μ, φ..., θ..., ω, α..., β...]
    pub params: Vec<f64>,
    /// Final negative log-likelihood value
    pub neg_log_lik: f64,
    /// Conditional variances (σ²_t) from the in-sample fit
    pub conditional_variances: Vec<f64>,
    /// Fitted residuals (ε_t)
    pub residuals: Vec<f64>,
    /// Last values of the training data (for forecasting AR component)
    pub last_y: Vec<f64>,
}

impl GarchFit {
    /// Produz previsões n passos à frente via recursão.
    ///
    /// ## Instrução de Implementação
    ///
    /// ### Forecast da Média (ARMA):
    /// ```text
    /// Para h = 1..n_ahead:
    ///   ŷ_{T+h} = μ̂ + Σ_{i=1}^{p} φ̂_i * (y_{T+h-i} - μ̂)
    ///                + Σ_{j=1}^{q} θ̂_j * ε_{T+h-j}
    ///
    /// onde:
    ///   - y_{T+h-i} = valor observado se T+h-i ≤ T, senão ŷ_{T+h-i} (previsão anterior)
    ///   - ε_{T+h-j} = resíduo observado se T+h-j ≤ T, senão 0 (E[ε] = 0)
    /// ```
    ///
    /// ### Forecast da Variância (GARCH):
    /// ```text
    /// Para h = 1..n_ahead:
    ///   σ̂²_{T+h} = ω̂ + Σ_{i=1}^{m} α̂_i * ε̂²_{T+h-i}
    ///                 + Σ_{j=1}^{s} β̂_j * σ̂²_{T+h-j}
    ///
    /// onde:
    ///   - ε̂²_{T+h-i} = ε² observado se T+h-i ≤ T, senão σ̂²_{T+h-i} (E[ε²] = σ²)
    ///   - σ̂²_{T+h-j} = σ² observado se T+h-j ≤ T, senão σ̂²_{T+h-j} (recursão)
    /// ```
    ///
    /// ### Passos da implementação:
    /// 1. Parsear `self.params` para obter μ̂, φ̂, θ̂, ω̂, α̂, β̂
    /// 2. Usar `self.residuals` e `self.conditional_variances` como histórico
    /// 3. O vetor de train_data original é necessário → armazenar em `GarchFit`
    ///    ou recebê-lo como argumento
    /// 4. Loop h = 1..n_ahead calculando ŷ_{T+h}
    /// 5. Retornar `(Vec<f64>, Vec<f64>)` com as previsões de MÉDIA e VARIÂNCIA
    pub fn forecast(&self, n_ahead: usize) -> Result<(Vec<f64>, Vec<f64>), CvError> {
        let p = self.spec.arma_order.p;
        let q = self.spec.arma_order.q;
        let m = self.spec.garch_order.m;
        let s = self.spec.garch_order.s;

        let mu = self.params[0];
        let ar_coeffs = &self.params[1..1+p];
        let ma_coeffs = &self.params[1+p..1+p+q];
        let omega = self.params[1+p+q];
        let alpha_coeffs = &self.params[2+p+q..2+p+q+m];
        let beta_coeffs = &self.params[2+p+q+m..2+p+q+m+s];

        let mut y_buffer = self.last_y.clone();
        if y_buffer.len() < p {
            let mut pad = vec![mu; p - y_buffer.len()];
            pad.extend(y_buffer);
            y_buffer = pad;
        }

        let mut eps_buffer = self.residuals.iter().copied().rev().take(q).collect::<Vec<f64>>();
        eps_buffer.reverse();
        if eps_buffer.len() < q {
            let mut pad = vec![0.0; q - eps_buffer.len()];
            pad.extend(eps_buffer);
            eps_buffer = pad;
        }

        let mut var_buffer = self.conditional_variances.iter().copied().rev().take(m.max(s)).collect::<Vec<f64>>();
        var_buffer.reverse();
        if var_buffer.len() < m.max(s) {
            let prev_var = var_buffer.last().copied().unwrap_or(omega);
            let mut pad = vec![prev_var; m.max(s) - var_buffer.len()];
            pad.extend(var_buffer);
            var_buffer = pad;
        }

        let mut eps2_buffer: Vec<f64> = self.residuals.iter().copied().rev().take(m).map(|e| e*e).collect();
        eps2_buffer.reverse();
        if eps2_buffer.len() < m {
            let mut pad = vec![omega; m - eps2_buffer.len()];
            pad.extend(eps2_buffer);
            eps2_buffer = pad;
        }

        let mut forecasts = Vec::with_capacity(n_ahead);
        let mut var_forecasts = Vec::with_capacity(n_ahead);

        for _ in 0..n_ahead {
            let mut mean_h = mu;
            for i in 0..p {
                mean_h += ar_coeffs[i] * (y_buffer[y_buffer.len() - 1 - i] - mu);
            }
            for j in 0..q {
                mean_h += ma_coeffs[j] * eps_buffer[eps_buffer.len() - 1 - j];
            }

            let mut var_h = omega;
            for i in 0..m {
                var_h += alpha_coeffs[i] * eps2_buffer[eps2_buffer.len() - 1 - i];
            }
            for j in 0..s {
                var_h += beta_coeffs[j] * var_buffer[var_buffer.len() - 1 - j];
            }
            
            forecasts.push(mean_h);
            var_forecasts.push(var_h);

            y_buffer.push(mean_h);
            eps_buffer.push(0.0);
            eps2_buffer.push(var_h);
            var_buffer.push(var_h);
        }

        Ok((forecasts, var_forecasts))
    }
}

/// Fits an ARMA-GARCH model to the training data.
///
/// ## Instrução de Implementação
///
/// ### Passo 1 — Preparar o problema de otimização:
/// ```text
/// struct GarchProblem {
///     likelihood: SGarchLikelihood,   // de likelihood.rs
///     data: Vec<f64>,                 // série de treino
/// }
///
/// impl CostFunction for GarchProblem {
///     type Param = Vec<f64>;
///     type Output = f64;
///     fn cost(&self, p: &Vec<f64>) -> Result<f64, Error> {
///         Ok(self.likelihood.neg_log_likelihood(p, &self.data))
///     }
/// }
/// ```
///
/// ### Passo 2 — Chutes iniciais + bounds:
/// ```text
/// let x0 = likelihood.initial_params(train_data);
/// let (lb, ub) = likelihood.param_bounds();
/// ```
///
/// ### Passo 3 — Executar L-BFGS (tentativa primária):
/// ```text
/// use argmin::solver::linesearch::MoreThuenteLineSearch;
/// use argmin::solver::quasinewton::LBFGS;
///
/// let linesearch = MoreThuenteLineSearch::new();
/// let solver = LBFGS::new(linesearch, 10);  // 10 = histórico
///
/// let result = Executor::new(problem, solver)
///     .configure(|state| state
///         .param(x0.clone())
///         .max_iters(500)
///     )
///     .run();
/// ```
///
/// **Nota:** L-BFGS precisa de `Gradient` trait. Usar diferenças finitas:
/// ```text
/// impl Gradient for GarchProblem {
///     type Param = Vec<f64>;
///     type Gradient = Vec<f64>;
///     fn gradient(&self, p: &Vec<f64>) -> Result<Vec<f64>, Error> {
///         // Diferenças finitas centrais:
///         let h = 1e-8;
///         let grad = (0..p.len()).map(|i| {
///             let mut p_plus = p.clone();
///             let mut p_minus = p.clone();
///             p_plus[i] += h;
///             p_minus[i] -= h;
///             (self.cost(&p_plus)? - self.cost(&p_minus)?) / (2.0 * h)
///         }).collect();
///         Ok(grad)
///     }
/// }
/// ```
///
/// ### Passo 4 — Fallback para Nelder-Mead (se L-BFGS falhar):
/// ```text
/// use argmin::solver::neldermead::NelderMead;
///
/// // Gerar simplex inicial: x0 + perturbações
/// let simplex: Vec<Vec<f64>> = gerar_simplex(x0, n_params);
///
/// let solver = NelderMead::new(simplex);
/// let result = Executor::new(problem, solver)
///     .configure(|state| state.max_iters(2000))
///     .run();
/// ```
///
/// ### Passo 5 — Construir GarchFit:
/// ```text
/// let best_params = result.state.best_param;
/// let (residuals, variances) = likelihood.compute_residuals_and_variances(&best_params, data);
///
/// Ok(GarchFit {
///     spec: spec.clone(),
///     params: best_params,
///     neg_log_lik: result.state.best_cost,
///     conditional_variances: variances,
///     residuals: residuals,
/// })
/// ```
///
/// ### Passo 6 — Verificar convergência:
/// ```text
/// - Se result.state.best_cost == f64::INFINITY → Err(EstimationError)
/// - Se Σα + Σβ ≥ 1 → warning (não-estacionário) ou Err
/// - Se ω ≤ 0 → Err (variância negativa)
/// ```
///
/// # Arguments
/// * `spec` — Model specification (orders, distribution, solver)
/// * `train_data` — Training time series slice
///
/// # Returns
/// `GarchFit` on success, `CvError::EstimationError` on failure.
#[derive(Clone)]
struct GarchProblem {
    likelihood: SGarchLikelihood,
    data: Vec<f64>,
}

impl CostFunction for GarchProblem {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        Ok(self.likelihood.neg_log_likelihood(p, &self.data))
    }
}

impl Gradient for GarchProblem {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, p: &Self::Param) -> Result<Self::Gradient, Error> {
        let h = 1e-8;
        let mut grad = Vec::with_capacity(p.len());
        for i in 0..p.len() {
            let mut p_plus = p.clone();
            let mut p_minus = p.clone();
            p_plus[i] += h;
            p_minus[i] -= h;
            
            let c_plus = self.cost(&p_plus)?;
            let c_minus = self.cost(&p_minus)?;
            
            grad.push((c_plus - c_minus) / (2.0 * h));
        }
        Ok(grad)
    }
}

pub fn garch_fit(spec: &ModelSpec, train_data: &[f64]) -> Result<GarchFit, CvError> {
    let likelihood = SGarchLikelihood::new(spec.clone());
    let x0 = likelihood.initial_params(train_data);

    let problem = GarchProblem {
        likelihood: SGarchLikelihood::new(spec.clone()),
        data: train_data.to_vec(),
    };

    let mut best_params = x0.clone();
    let mut best_cost = f64::INFINITY;
    let mut success = false;

    // Try L-BFGS
    let linesearch = MoreThuenteLineSearch::new();
    let solver = LBFGS::new(linesearch, 10);
    
    if let Ok(res) = Executor::new(problem.clone(), solver)
        .configure(|state| state.param(x0.clone()).max_iters(500))
        .run()
    {
        if let Some(bp) = res.state.get_best_param() {
            if res.state.get_best_cost() < f64::INFINITY {
                best_params = bp.clone();
                best_cost = res.state.get_best_cost();
                success = true;
            }
        }
    }

    // Fallback to NelderMead
    if !success || best_cost == f64::INFINITY || best_cost.is_nan() {
        let mut simplex = vec![x0.clone()];
        for i in 0..x0.len() {
            let mut pt = x0.clone();
            if pt[i] == 0.0 {
                pt[i] = 0.00025;
            } else {
                pt[i] *= 1.05;
            }
            simplex.push(pt);
        }

        if let Ok(nm_solver) = NelderMead::new(simplex).with_sd_tolerance(1e-6) {
            if let Ok(res) = Executor::new(problem.clone(), nm_solver)
                .configure(|state| state.max_iters(2000))
                .run()
            {
                if let Some(bp) = res.state.get_best_param() {
                    let cost = res.state.get_best_cost();
                    if cost < best_cost {
                        best_params = bp.clone();
                        best_cost = cost;
                        success = true;
                    }
                }
            }
        }
    }

    if !success || best_cost == f64::INFINITY || best_cost.is_nan() {
        return Err(CvError::EstimationError("Optimization failed to converge".to_string()));
    }

    let (residuals, variances) = problem.likelihood.compute_residuals_and_variances(&best_params, train_data);
    
    let p = spec.arma_order.p;
    let last_y: Vec<f64> = train_data.iter().rev().take(p.max(1)).copied().collect::<Vec<f64>>().into_iter().rev().collect();

    Ok(GarchFit {
        spec: spec.clone(),
        params: best_params,
        neg_log_lik: best_cost,
        conditional_variances: variances,
        residuals,
        last_y,
    })
}
