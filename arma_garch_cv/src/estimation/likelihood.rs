//! Log-likelihood functions for ARMA-GARCH models.
//!
//! ## Visão Geral
//!
//! A estimação de um modelo ARMA(p,q)-GARCH(m,s) é feita por **Maximum Likelihood
//! Estimation (MLE)**. O objetivo é encontrar o vetor de parâmetros θ que maximiza
//! a log-verossimilhança da série observada, ou equivalentemente, minimiza a
//! **negativa** da log-verossimilhança.
//!
//! ## Modelo ARMA(p,q)-GARCH(m,s) com Normal
//!
//! ### Equação da Média (ARMA):
//! ```text
//! y_t = μ + Σ_{i=1}^{p} φ_i * (y_{t-i} - μ) + ε_t + Σ_{j=1}^{q} θ_j * ε_{t-j}
//! ```
//! onde:
//! - μ = intercepto (média incondicional)
//! - φ_i = coeficientes AR
//! - θ_j = coeficientes MA
//! - ε_t = resíduo (inovação)
//!
//! ### Equação da Variância (sGARCH):
//! ```text
//! σ²_t = ω + Σ_{i=1}^{m} α_i * ε²_{t-i} + Σ_{j=1}^{s} β_j * σ²_{t-j}
//! ```
//! onde:
//! - ω > 0 = constante de variância
//! - α_i ≥ 0 = coeficientes ARCH (reação a choques)
//! - β_j ≥ 0 = coeficientes GARCH (persistência)
//! - Restrição de estacionariedade: Σα_i + Σβ_j < 1
//!
//! ### Log-Verossimilhança (Normal):
//! ```text
//! ℓ(θ) = Σ_{t=1}^{T} log f(z_t)
//!       = Σ_{t=1}^{T} [-0.5 * (ln(2π) + ln(σ²_t) + z²_t)]
//!
//! onde z_t = ε_t / σ_t  (resíduo padronizado)
//! ```
//!
//! **Nota:** A contribuição de cada observação t inclui o termo `-0.5 * ln(σ²_t)`
//! que diferencia a GARCH likelihood de uma regressão simples — a variância
//! muda a cada passo.
//!
//! ## Vetor de Parâmetros θ
//!
//! O vetor é organizado como:
//! ```text
//! θ = [μ, φ_1, ..., φ_p, θ_1, ..., θ_q, ω, α_1, ..., α_m, β_1, ..., β_s]
//!      ↑       AR          ↑     MA       ↑        GARCH
//!   índice 0          índice p+1      índice p+q+1
//! ```
//!
//! Total de parâmetros: 1 + p + q + 1 + m + s
//!
//! ## Dependências
//! - `statrs` — não necessário para Normal (fórmula direta)
//! - `ndarray` — para operações vetorizadas na likelihood
//! - `argmin` — `CostFunction` trait para integração com otimizadores

use crate::spec::ModelSpec;

/// Precomputed ln(2π) ≈ 1.8378770664093453
const LN_2PI: f64 = 1.8378770664093453;

pub struct ParsedParams<'a> {
    pub mu: f64,
    pub ar_coeffs: &'a [f64],
    pub ma_coeffs: &'a [f64],
    pub omega: f64,
    pub alpha_coeffs: &'a [f64],
    pub beta_coeffs: &'a [f64],
}


/// Trait for computing the log-likelihood of a GARCH model.
///
/// Types implementing this trait can be used as cost functions
/// for `argmin` optimizers.
pub trait LogLikelihood {
    /// Computes the negative log-likelihood (for minimization).
    ///
    /// # Arguments
    /// * `params` — Parameter vector [μ, φ..., θ..., ω, α..., β...]
    /// * `data` — Observed time series
    ///
    /// # Returns
    /// Negative log-likelihood value (lower is better).
    fn neg_log_likelihood(&self, params: &[f64], data: &[f64]) -> f64;

    /// Returns the number of parameters for this model specification.
    fn n_params(&self) -> usize;

    /// Returns initial parameter guesses based on the data.
    fn initial_params(&self, data: &[f64]) -> Vec<f64>;

    /// Returns (lower_bounds, upper_bounds) for constrained optimization.
    fn param_bounds(&self) -> (Vec<f64>, Vec<f64>);
}

#[derive(Clone)]
pub struct SGarchLikelihood {
    pub spec: ModelSpec,
}

impl SGarchLikelihood {
    pub fn new(spec: ModelSpec) -> Self {
        Self { spec }
    }

    /// Extrai os sub-vetores de parâmetros do vetor θ completo.
    ///
    /// ## Instrução de Implementação
    ///
    /// Dado o vetor `params` com layout:
    /// ```text
    /// [μ, φ_1..φ_p, θ_1..θ_q, ω, α_1..α_m, β_1..β_s]
    /// ```
    ///
    /// Retornar uma struct/tupla com os slices:
    /// - `mu = params[0]`
    /// - `ar_coeffs = &params[1..1+p]`
    /// - `ma_coeffs = &params[1+p..1+p+q]`
    /// - `omega = params[1+p+q]`
    /// - `alpha_coeffs = &params[2+p+q..2+p+q+m]`
    /// - `beta_coeffs = &params[2+p+q+m..2+p+q+m+s]`
    fn parse_params<'a>(&self, params: &'a [f64]) -> ParsedParams<'a> {
        let p = self.spec.arma_order.p;
        let q = self.spec.arma_order.q;
        let m = self.spec.garch_order.m;
        let s = self.spec.garch_order.s;
        
        let mu = params[0];
        let ar_coeffs = &params[1..1+p];
        let ma_coeffs = &params[1+p..1+p+q];
        let omega = params[1+p+q];
        let alpha_coeffs = &params[2+p+q..2+p+q+m];
        let beta_coeffs = &params[2+p+q+m..2+p+q+m+s];
        
        ParsedParams {
            mu,
            ar_coeffs,
            ma_coeffs,
            omega,
            alpha_coeffs,
            beta_coeffs,
        }
    }

    /// Calcula os resíduos ε_t e variâncias condicionais σ²_t.
    ///
    /// ## Instrução de Implementação
    ///
    /// Esta é a função CENTRAL. Itera sobre t = 0..T e calcula:
    ///
    /// ### Passo 1 — Inicialização (t < max(p, q, m, s)):
    /// ```text
    /// ε_t = y_t - μ                (resíduo inicial, sem termos AR/MA)
    /// σ²_t = variância amostral    (warm-up com var(data))
    /// ```
    ///
    /// ### Passo 2 — Recursão para t ≥ max(p, q, m, s):
    /// ```text
    /// // Média condicional (ARMA):
    /// μ_t = μ + Σ_{i=1}^{p} φ_i * (y_{t-i} - μ)
    ///         + Σ_{j=1}^{q} θ_j * ε_{t-j}
    ///
    /// // Resíduo:
    /// ε_t = y_t - μ_t
    ///
    /// // Variância condicional (GARCH):
    /// σ²_t = ω + Σ_{i=1}^{m} α_i * ε²_{t-i}
    ///          + Σ_{j=1}^{s} β_j * σ²_{t-j}
    ///
    /// // Proteção: σ²_t = max(σ²_t, 1e-12) para evitar log(0)
    /// ```
    ///
    /// ### Retorno:
    /// - `residuals: Vec<f64>` de tamanho T (ε_t para cada t)
    /// - `variances: Vec<f64>` de tamanho T (σ²_t para cada t)
    pub fn compute_residuals_and_variances(
        &self,
        params: &[f64],
        data: &[f64],
    ) -> (Vec<f64>, Vec<f64>) {
        let parsed = self.parse_params(params);
        let t_len = data.len();
        let mut residuals = vec![0.0; t_len];
        let mut variances = vec![0.0; t_len];

        let p = self.spec.arma_order.p;
        let q = self.spec.arma_order.q;
        let m = self.spec.garch_order.m;
        let s = self.spec.garch_order.s;

        let warmup_len = p.max(q).max(m).max(s);

        let var_amostral = if t_len > 1 {
            let sum_sq: f64 = data.iter().map(|&y| (y - parsed.mu).powi(2)).sum();
            sum_sq / (t_len as f64 - 1.0)
        } else {
            1.0
        };

        for t in 0..t_len {
            if t < warmup_len {
                residuals[t] = data[t] - parsed.mu;
                variances[t] = var_amostral;
            } else {
                let mut mu_t = parsed.mu;
                for i in 0..p {
                    mu_t += parsed.ar_coeffs[i] * (data[t - 1 - i] - parsed.mu);
                }
                for j in 0..q {
                    mu_t += parsed.ma_coeffs[j] * residuals[t - 1 - j];
                }

                let eps_t = data[t] - mu_t;
                residuals[t] = eps_t;

                let mut sigma2_t = parsed.omega;
                for i in 0..m {
                    sigma2_t += parsed.alpha_coeffs[i] * residuals[t - 1 - i].powi(2);
                }
                for j in 0..s {
                    sigma2_t += parsed.beta_coeffs[j] * variances[t - 1 - j];
                }

                variances[t] = sigma2_t.max(1e-12);
            }
        }

        (residuals, variances)
    }
}

impl LogLikelihood for SGarchLikelihood {
    /// Calcula a negativa da log-verossimilhança.
    ///
    /// ## Instrução de Implementação
    ///
    /// ```text
    /// neg_ℓ(θ) = -Σ_{t=0}^{T-1} [-0.5 * (ln(2π) + ln(σ²_t) + ε²_t / σ²_t)]
    ///          =  0.5 * Σ_{t=0}^{T-1} [ln(2π) + ln(σ²_t) + ε²_t / σ²_t]
    /// ```
    ///
    /// ### Passos:
    /// 1. Chamar `compute_residuals_and_variances(params, data)` → (ε, σ²)
    /// 2. Somar sobre t: `0.5 * (LN_2PI + σ²_t.ln() + ε_t² / σ²_t)`
    /// 3. Se qualquer σ²_t ≤ 0 ou NaN → retornar `f64::INFINITY` (rejeita params)
    /// 4. Retornar a soma (valor a ser MINIMIZADO pelo otimizador)
    fn neg_log_likelihood(&self, params: &[f64], data: &[f64]) -> f64 {
        let parsed = self.parse_params(params);

        let sum_alpha_beta: f64 = parsed.alpha_coeffs.iter().sum::<f64>() + parsed.beta_coeffs.iter().sum::<f64>();
        if parsed.omega <= 0.0 || sum_alpha_beta >= 1.0 {
            return f64::INFINITY;
        }

        let (resid, var) = self.compute_residuals_and_variances(params, data);
        let mut nll: f64 = 0.0;
        let ln_2pi = LN_2PI;
        for t in 0..resid.len() {
            let s2 = var[t];
            let e = resid[t];
            if s2 <= 0.0 || s2.is_nan() {
                return f64::INFINITY;
            }
            nll += 0.5 * (ln_2pi + s2.ln() + e * e / s2);
        }

        if nll.is_nan() || nll.is_infinite() {
            f64::INFINITY
        } else {
            nll
        }
    }

    fn n_params(&self) -> usize {
        // μ + AR(p) + MA(q) + ω + ARCH(m) + GARCH(s)
        1 + self.spec.arma_order.p + self.spec.arma_order.q
            + 1 + self.spec.garch_order.m + self.spec.garch_order.s
    }

    /// Gera estimativas iniciais de parâmetros.
    ///
    /// ## Instrução de Implementação
    ///
    /// ### Valores iniciais recomendados:
    /// ```text
    /// μ₀ = mean(data)                      → média amostral
    /// φ_i₀ = 0.1 para todo i              → AR perto de zero
    /// θ_j₀ = 0.1 para todo j              → MA perto de zero
    /// ω₀ = var(data) * 0.1                 → 10% da variância amostral
    /// α_i₀ = 0.05                          → ARCH pequeno
    /// β_j₀ = 0.85                          → GARCH com alta persistência
    /// ```
    ///
    /// A ideia é: α + β ≈ 0.9 (alta persistência, mas estacionário).
    fn initial_params(&self, data: &[f64]) -> Vec<f64> {
        let t_len = data.len();
        let mu = if t_len > 0 {
            data.iter().sum::<f64>() / (t_len as f64)
        } else {
            0.0
        };

        let var = if t_len > 1 {
            data.iter().map(|&y| (y - mu).powi(2)).sum::<f64>() / (t_len as f64 - 1.0)
        } else {
            1.0
        };

        let p = self.spec.arma_order.p;
        let q = self.spec.arma_order.q;
        let m = self.spec.garch_order.m;
        let s = self.spec.garch_order.s;

        let mut params = Vec::with_capacity(1 + p + q + 1 + m + s);
        params.push(mu);
        for _ in 0..p { params.push(0.1); }
        for _ in 0..q { params.push(0.1); }
        params.push(var * 0.1);
        for _ in 0..m { params.push(0.05); }
        let beta0 = if s > 0 { 0.85 / (s as f64) } else { 0.0 };
        for _ in 0..s { params.push(beta0); }

        params
    }

    /// Retorna limites inferiores e superiores dos parâmetros.
    ///
    /// ## Instrução de Implementação
    ///
    /// ```text
    /// Parâmetro     | Lower       | Upper
    /// ──────────────|─────────────|────────────
    /// μ             | -∞ (f64::MIN) | +∞ (f64::MAX)
    /// φ_i (AR)      | -0.9999     | 0.9999      (estacionariedade)
    /// θ_j (MA)      | -0.9999     | 0.9999      (invertibilidade)
    /// ω             | 1e-8        | f64::MAX    (positividade estrita)
    /// α_i (ARCH)    | 1e-8        | 0.9999      (positividade)
    /// β_j (GARCH)   | 1e-8        | 0.9999      (positividade)
    /// ```
    ///
    /// **Nota:** A restrição Σα + Σβ < 1 NÃO é imposta aqui diretamente—
    /// ela é verificada penalizando a likelihood (retornar +∞ se violada)
    /// ou usando um otimizador com suporte a restrições não-lineares.
    fn param_bounds(&self) -> (Vec<f64>, Vec<f64>) {
        let n = self.n_params();
        let mut lb = vec![f64::MIN; n];
        let mut ub = vec![f64::MAX; n];

        let p = self.spec.arma_order.p;
        let q = self.spec.arma_order.q;
        let m = self.spec.garch_order.m;
        let s = self.spec.garch_order.s;

        let mut idx = 1;
        for _ in 0..p {
            lb[idx] = -0.9999;
            ub[idx] = 0.9999;
            idx += 1;
        }
        for _ in 0..q {
            lb[idx] = -0.9999;
            ub[idx] = 0.9999;
            idx += 1;
        }
        
        lb[idx] = 1e-8;
        idx += 1;
        
        for _ in 0..m {
            lb[idx] = 1e-8;
            ub[idx] = 0.9999;
            idx += 1;
        }
        for _ in 0..s {
            lb[idx] = 1e-8;
            ub[idx] = 0.9999;
            idx += 1;
        }

        (lb, ub)
    }
}
