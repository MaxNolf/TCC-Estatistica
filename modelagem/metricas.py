"""
metricas.py — Métricas de avaliação de previsão de variância.

Implementa métricas robustas para comparar o poder preditivo da variância
condicional estimada pelo GARCH contra a variância realizada (proxy Yang-Zhang).

Referências:
  - Patton, A.J. (2011). "Volatility forecast comparison using imperfect
    volatility proxies." Journal of Econometrics, 160(1), 246-256.
  - Mincer, J. & Zarnowitz, V. (1969). "The evaluation of economic forecasts."
  - Diebold, F.X. & Mariano, R.S. (1995). "Comparing predictive accuracy."
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
import statsmodels.api as sm


def mse(forecast_var, realized_var):
    """
    Mean Squared Error entre variância prevista e realizada.

    Pertence à classe de funções de perda robustas de Patton (2011),
    garantindo que o ranking de modelos é consistente mesmo quando
    se usa um proxy ruidoso da volatilidade verdadeira.

    Parameters
    ----------
    forecast_var : array-like
        Variância condicional prevista pelo modelo.
    realized_var : array-like
        Variância realizada (proxy Yang-Zhang).

    Returns
    -------
    float
        MSE.
    """
    f = np.asarray(forecast_var)
    r = np.asarray(realized_var)
    return np.mean((f - r) ** 2)


def mae(forecast_var, realized_var):
    """
    Mean Absolute Error entre variância prevista e realizada.

    Parameters
    ----------
    forecast_var : array-like
        Variância condicional prevista.
    realized_var : array-like
        Variância realizada.

    Returns
    -------
    float
        MAE.
    """
    f = np.asarray(forecast_var)
    r = np.asarray(realized_var)
    return np.mean(np.abs(f - r))


def qlike(forecast_var, realized_var):
    """
    Quasi-Likelihood loss function.

    QLIKE = mean( log(σ²_forecast) + RV / σ²_forecast )

    Métrica robusta de Patton (2011): o ranking de modelos é preservado
    mesmo quando a proxy de volatilidade contém ruído de estimação.
    QLIKE penaliza assimetricamente sub e sobre-estimação, o que é
    economicamente mais realista.

    Parameters
    ----------
    forecast_var : array-like
        Variância condicional prevista.
    realized_var : array-like
        Variância realizada.

    Returns
    -------
    float
        QLIKE loss.
    """
    f = np.asarray(forecast_var, dtype=np.float64)
    r = np.asarray(realized_var, dtype=np.float64)

    # Evitar log(0) e divisão por zero
    f = np.maximum(f, 1e-20)

    return np.mean(np.log(f) + r / f)


def r_squared(forecast_var, realized_var):
    """
    R² da regressão de Mincer-Zarnowitz.

    RV_t = α + β * σ²_forecast_t + ε_t

    Um forecast ótimo e não-viesado implica α=0, β=1.
    O R² mede a proporção da variação da RV explicada pelo forecast.

    Parameters
    ----------
    forecast_var : array-like
        Variância prevista.
    realized_var : array-like
        Variância realizada.

    Returns
    -------
    float
        R² da regressão MZ.
    """
    _, result = mincer_zarnowitz(forecast_var, realized_var)
    return result['r_squared']


def mincer_zarnowitz(forecast_var, realized_var):
    """
    Regressão de Mincer-Zarnowitz para avaliação de viés do forecast.

    RV_t = α + β * σ²_forecast_t + ε_t

    Testa H0: α=0, β=1 (forecast ótimo e não-viesado).

    Parameters
    ----------
    forecast_var : array-like
        Variância prevista.
    realized_var : array-like
        Variância realizada.

    Returns
    -------
    ols_result : statsmodels RegressionResults
        Resultado completo da regressão OLS.
    summary : dict
        Dicionário com alpha, beta, r_squared, f_stat, f_pvalue.
    """
    f = np.asarray(forecast_var, dtype=np.float64)
    r = np.asarray(realized_var, dtype=np.float64)

    X = sm.add_constant(f)
    ols = sm.OLS(r, X).fit()

    # Teste F conjunto: H0: α=0, β=1
    r_matrix = np.array([[1, 0], [0, 1]])  # Restrições em α e β
    q_vector = np.array([0, 1])             # H0: α=0, β=1
    try:
        f_test = ols.f_test(np.column_stack([r_matrix, q_vector]))
        f_stat = float(f_test.fvalue)
        f_pvalue = float(f_test.pvalue)
    except Exception:
        f_stat = np.nan
        f_pvalue = np.nan

    summary = {
        'alpha': ols.params[0],
        'beta': ols.params[1],
        'alpha_pvalue': ols.pvalues[0],
        'beta_pvalue': ols.pvalues[1],
        'r_squared': ols.rsquared,
        'f_stat_joint': f_stat,
        'f_pvalue_joint': f_pvalue,
    }

    return ols, summary


def diebold_mariano(loss1, loss2, h=1):
    """
    Teste de Diebold-Mariano para igualdade de poder preditivo.

    Testa H0: E[d_t] = 0, onde d_t = L(e1_t) - L(e2_t)
    (a diferença nas funções de perda dos dois modelos).

    Um DM negativo e significante indica que o modelo 1 é melhor.
    Um DM positivo e significante indica que o modelo 2 é melhor.

    Parameters
    ----------
    loss1 : array-like
        Série de perdas (e.g., (forecast1 - RV)²) do modelo 1.
    loss2 : array-like
        Série de perdas do modelo 2.
    h : int
        Horizonte de previsão (para cálculo de HAC variance).

    Returns
    -------
    dict com:
        'dm_stat': estatística DM
        'p_value': p-valor bilateral
        'conclusion': string descritiva
    """
    d = np.asarray(loss1) - np.asarray(loss2)
    n = len(d)
    d_mean = np.mean(d)

    # Variância HAC (Newey-West) para horizonte h
    gamma_0 = np.var(d, ddof=1)
    gamma_sum = 0
    for k in range(1, h):
        gamma_k = np.cov(d[k:], d[:-k])[0, 1]
        gamma_sum += gamma_k
    var_d = (gamma_0 + 2 * gamma_sum) / n

    if var_d <= 0:
        var_d = gamma_0 / n

    dm_stat = d_mean / np.sqrt(var_d)
    p_value = 2 * (1 - scipy_stats.norm.cdf(abs(dm_stat)))

    if p_value < 0.05:
        if dm_stat < 0:
            conclusion = "Modelo 1 é significativamente melhor (p < 0.05)"
        else:
            conclusion = "Modelo 2 é significativamente melhor (p < 0.05)"
    else:
        conclusion = "Não há diferença significativa entre os modelos (p >= 0.05)"

    return {
        'dm_stat': dm_stat,
        'p_value': p_value,
        'conclusion': conclusion,
    }


def compute_all_metrics(forecast_var, realized_var, model_name="Modelo"):
    """
    Calcula todas as métricas de avaliação para um único modelo.

    Parameters
    ----------
    forecast_var : array-like
        Variância prevista.
    realized_var : array-like
        Variância realizada (proxy Yang-Zhang).
    model_name : str
        Nome do modelo (para exibição).

    Returns
    -------
    dict
        Dicionário com todas as métricas calculadas.
    """
    _, mz_summary = mincer_zarnowitz(forecast_var, realized_var)

    metrics = {
        'model': model_name,
        'mse': mse(forecast_var, realized_var),
        'mae': mae(forecast_var, realized_var),
        'qlike': qlike(forecast_var, realized_var),
        'r_squared': mz_summary['r_squared'],
        'mz_alpha': mz_summary['alpha'],
        'mz_beta': mz_summary['beta'],
        'mz_alpha_pvalue': mz_summary['alpha_pvalue'],
        'mz_beta_pvalue': mz_summary['beta_pvalue'],
    }

    return metrics


def comparison_table(metrics_dollar, metrics_time, forecast_var_dollar,
                     forecast_var_time, realized_var_dollar, realized_var_time):
    """
    Gera uma tabela comparativa completa entre Dollar Bars e Time Bars,
    incluindo o teste de Diebold-Mariano.

    Parameters
    ----------
    metrics_dollar : dict
        Métricas do modelo Dollar Bars.
    metrics_time : dict
        Métricas do modelo Time Bars.
    forecast_var_dollar, forecast_var_time : array-like
        Variâncias previstas.
    realized_var_dollar, realized_var_time : array-like
        Variâncias realizadas.

    Returns
    -------
    pd.DataFrame
        Tabela comparativa.
    dict
        Resultado do teste Diebold-Mariano (usando MSE loss).
    """
    # Tabela comparativa
    df = pd.DataFrame([metrics_dollar, metrics_time])
    df.set_index('model', inplace=True)

    # Diebold-Mariano: comparar as perdas squared error
    f_d = np.asarray(forecast_var_dollar)
    r_d = np.asarray(realized_var_dollar)
    f_t = np.asarray(forecast_var_time)
    r_t = np.asarray(realized_var_time)

    # Para DM, precisamos do mesmo comprimento — usar o mínimo
    min_len = min(len(f_d), len(f_t))
    loss_dollar = (f_d[:min_len] - r_d[:min_len]) ** 2
    loss_time = (f_t[:min_len] - r_t[:min_len]) ** 2

    dm_result = diebold_mariano(loss_dollar, loss_time, h=1)

    return df, dm_result
