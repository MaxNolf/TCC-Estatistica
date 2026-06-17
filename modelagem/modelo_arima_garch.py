"""
modelo_arima_garch.py — Fitting e previsão ARIMA-GARCH via backend Rust.

O motor pesado (MLE fitting, grid search, forecast) é executado em Rust nativo
via PyO3 bindings do crate `arma_garch_cv`. O Python cuida apenas da orquestração
e formatação dos resultados.

Fallback: se o módulo Rust não estiver disponível, usa `arch` (Python puro).
"""

import warnings
import numpy as np
import pandas as pd

# Tentar importar o backend Rust
try:
    import arma_garch_cv as rust_backend
    RUST_AVAILABLE = True
    print("[INFO] Backend Rust (arma_garch_cv) carregado com sucesso.")
except ImportError:
    RUST_AVAILABLE = False
    print("[AVISO] Backend Rust não disponível. Tentando fallback Python (arch)...")
    try:
        from arch import arch_model
        from statsmodels.tsa.arima.model import ARIMA
        ARCH_AVAILABLE = True
    except ImportError:
        ARCH_AVAILABLE = False
        print("[ERRO] Nem Rust nem 'arch' estão disponíveis. Instale um dos dois.")


# ==============================================================================
# RUST BACKEND
# ==============================================================================

def fit_arima_garch_rust(returns, arma_p=1, arma_q=1, garch_m=1, garch_s=1):
    """
    Ajusta ARMA(p,q)-GARCH(m,s) via Rust.

    Note: o motor Rust implementa ARMA (sem diferenciação). Para d>0,
    diferencie os retornos em Python antes de chamar esta função.
    """
    data = np.asarray(returns, dtype=np.float64).tolist()
    result = rust_backend.fit_arma_garch(data, arma_p, arma_q, garch_m, garch_s)
    return result


def grid_search_rust(returns, p_max=5, q_max=5, garch_m_max=2, garch_s_max=2,
                     criterion='aic'):
    """
    Grid search paralelo sobre ARMA(p,q) × GARCH(m,s) via Rust + rayon.
    """
    data = np.asarray(returns, dtype=np.float64).tolist()
    result = rust_backend.grid_search_parallel(
        data, p_max, q_max, garch_m_max, garch_s_max, criterion
    )
    return result


def forecast_variance_rust(returns_full, n_train, arma_p=1, arma_q=1,
                           garch_m=1, garch_s=1):
    """
    Previsão da variância condicional no período de teste via Rust.

    O modelo é ajustado UMA VEZ no treino e a variância condicional é
    calculada recursivamente sobre toda a série com parâmetros fixos.
    Isso é O(n) em vez de O(n_test × fit_time).
    """
    data = np.asarray(returns_full, dtype=np.float64).tolist()
    result = rust_backend.rolling_variance_forecast(
        data, arma_p, arma_q, garch_m, garch_s, n_train
    )
    return result


def cpcv_grid_search_rust(returns, fold_train_indices, fold_test_indices,
                          p_max=5, q_max=5, garch_m_max=2, garch_s_max=2,
                          criterion='aic', _returns_as_list=None):
    """
    Grid search com CPCV — todos os folds processados em paralelo via Rust.

    Parameters
    ----------
    returns : array-like
        Série completa de retornos (treino). Ignorado se _returns_as_list
        for fornecido.
    fold_train_indices : list of list of int
        Índices de treino para cada fold CPCV. Aceita listas Python
        pré-convertidas (evita reconversão numpy→list).
    fold_test_indices : list of list of int
        Índices de teste para cada fold CPCV.
    _returns_as_list : list of float, optional
        Se fornecido, usa diretamente sem reconverter `returns`.
        Otimização para evitar `tolist()` a cada batch.

    Returns
    -------
    list of dict
        Resultado por fold (melhor modelo, AIC, BIC).
    """
    if _returns_as_list is not None:
        data = _returns_as_list
    else:
        data = np.asarray(returns, dtype=np.float64).tolist()

    # Converter arrays numpy para listas de inteiros Python (se necessário)
    train_idx = [idx.tolist() if hasattr(idx, 'tolist') else idx
                 for idx in fold_train_indices]
    test_idx = [idx.tolist() if hasattr(idx, 'tolist') else idx
                for idx in fold_test_indices]

    results = rust_backend.cpcv_grid_search(
        data, train_idx, test_idx,
        p_max, q_max, garch_m_max, garch_s_max, criterion
    )
    return results


# ==============================================================================
# PYTHON FALLBACK (usa arch)
# ==============================================================================

def fit_arima_garch_python(returns, arima_order=(1, 0, 1), garch_p=1, garch_q=1):
    """Fallback: ajusta ARIMA-GARCH via statsmodels + arch."""
    if not ARCH_AVAILABLE:
        raise ImportError("Biblioteca 'arch' não instalada.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # ARIMA para a média
        arima_model = ARIMA(returns, order=arima_order)
        arima_result = arima_model.fit()
        residuals = arima_result.resid.values

        # GARCH nos resíduos
        garch_mod = arch_model(
            residuals, vol='Garch', p=garch_p, q=garch_q,
            dist='normal', mean='Zero', rescale=False
        )
        garch_result = garch_mod.fit(disp='off', show_warning=False)

    combined_aic = arima_result.aic + garch_result.aic
    combined_bic = arima_result.bic + garch_result.bic

    return {
        'arima_order': arima_order,
        'garch_order': (garch_p, garch_q),
        'aic': combined_aic,
        'bic': combined_bic,
        'conditional_variance': garch_result.conditional_volatility ** 2,
        'residuals': residuals,
        'converged': True,
    }


# ==============================================================================
# INTERFACE UNIFICADA
# ==============================================================================

def fit_model(returns, arma_p=1, arma_q=1, garch_m=1, garch_s=1):
    """
    Interface unificada: usa Rust se disponível, senão Python.
    """
    if RUST_AVAILABLE:
        return fit_arima_garch_rust(returns, arma_p, arma_q, garch_m, garch_s)
    else:
        return fit_arima_garch_python(
            returns,
            arima_order=(arma_p, 0, arma_q),
            garch_p=garch_m,
            garch_q=garch_s
        )


def grid_search(returns, p_max=5, q_max=5, garch_m_max=2, garch_s_max=2,
                criterion='aic'):
    """
    Interface unificada para grid search.
    """
    if RUST_AVAILABLE:
        return grid_search_rust(returns, p_max, q_max, garch_m_max,
                                garch_s_max, criterion)
    else:
        raise NotImplementedError(
            "Grid search Python puro não implementado. Instale o backend Rust."
        )


def forecast_variance(returns_full, n_train, arma_p=1, arma_q=1,
                      garch_m=1, garch_s=1):
    """
    Interface unificada para previsão de variância.
    """
    if RUST_AVAILABLE:
        return forecast_variance_rust(
            returns_full, n_train, arma_p, arma_q, garch_m, garch_s
        )
    else:
        raise NotImplementedError(
            "Forecast Python puro não implementado. Instale o backend Rust."
        )
