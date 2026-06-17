"""
main.py — Orquestrador principal do pipeline ARIMA-GARCH (Rust backend).

Fluxo:
  1. Carregar dados (dollar_bars_5m, time_bars_5m) e calcular retornos logarítmicos
  2. Split temporal: 80% treino / 20% teste
  3. CPCV no treino: grid search ARMA(p,q) × GARCH(m,s) via Rust (seleção por AIC/BIC)
     C(30,5) = 142.506 folds — backend Rust obrigatório.
  4. Retreinar modelo final com melhores parâmetros em TODO o treino
  5. Previsão da variância condicional no teste (Rust, parâmetros fixos)
  6. Agregar variância prevista intraday → diário (soma por dia)
  7. Calcular métricas vs RV Benchmark Consenso (MSE, MAE, QLIKE, MZ, Diebold-Mariano)
  8. Análise de distribuição: ajuste t-Student/Normal, QQ-Plot, IC bootstrap
  9. Gerar tabelas comparativas e gráficos
 10. Salvar resultados em 'resultados 2/'

Uso:
  python main.py
"""

import os
import sys
import time
import math
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats as sp_stats
from scipy.special import gammaln
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# tqdm com fallback silencioso se não instalado
try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False
    print("[AVISO] tqdm não instalado. Instale com: pip install tqdm")

def _pbar(iterable, **kw):
    """Retorna tqdm decorado se disponível, senão o iterável puro."""
    return _tqdm(iterable, **kw) if _HAS_TQDM else iterable

warnings.filterwarnings("ignore")

# Importações internas do pipeline
from config import (
    FILE_DOLLAR_5M, FILE_TIME_5M, FILE_RV_BENCHMARK, RESULTS_DIR,
    TRAIN_RATIO,
    CPCV_N_GROUPS, CPCV_K_TEST, EMBARGO_BARS,
    ARIMA_P_MAX, ARIMA_Q_MAX,
    GARCH_P_MAX, GARCH_Q_MAX,
    SELECTION_CRITERION, RANDOM_SEED
)
from cpcv import CombPurgedKFold
from modelo_arima_garch import (
    fit_model, grid_search, forecast_variance, cpcv_grid_search_rust,
    RUST_AVAILABLE
)
from metricas import compute_all_metrics, comparison_table


# ==============================================================================
# LOGGING
# ==============================================================================
class Logger:
    """Redireciona stdout para terminal + arquivo simultaneamente."""

    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# ==============================================================================
# ANÁLISE DE DISTRIBUIÇÃO — QQ-PLOT, AJUSTE, INTERVALO DE CONFIANÇA
# ==============================================================================




def plot_residual_analysis(residuals, fitted_values, date_index, series_label, bar_type_name, filename_prefix):
    """
    Gera 8 gráficos INDEPENDENTES de análise de resíduos:
      1. Resíduos vs. Tempo
      2. Histograma dos Resíduos
      3. QQ-Plot Normal
      4. Resíduos vs. Valores Ajustados
      5. ACF dos Resíduos
      6. PACF dos Resíduos
      7. ACF dos Resíduos ao Quadrado
      8. Previsão vs Observado (scatter)

    Todos os arquivos são salvos em analise_residuos/<bar_type_name>/.
    """
    data = np.asarray(residuals, dtype=np.float64)
    fitted = np.asarray(fitted_values, dtype=np.float64)
    finite_mask = np.isfinite(data)
    data   = data[finite_mask]
    fitted = fitted[finite_mask]
    n = len(data)
    observed = fitted + data   # RV_t = σ̂²_t − ê_t + ê_t = fitted + residual

    # Pasta de destino: analise_residuos/<bar_type_name>/
    out_dir = os.path.join(RESULTS_DIR, 'analise_residuos', bar_type_name)
    os.makedirs(out_dir, exist_ok=True)

    norm_params = sp_stats.norm.fit(data)
    x_grid = np.linspace(np.percentile(data, 0.1), np.percentile(data, 99.9), 500)
    title_suffix = f'{series_label} ({bar_type_name})'

    def _savefig(fig, suffix, label):
        p = os.path.join(out_dir, f'{filename_prefix}_{bar_type_name}_{suffix}.png')
        fig.savefig(p, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"    [resid {label}] Salvo: {p}")
        return p

    # ── 1. Resíduos vs. Tempo ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(date_index, data, color='steelblue', linewidth=1)
    ax.axhline(0, color='black', linewidth=1)
    ax.set_title(f'Resíduos vs. Tempo\n{title_suffix}', fontsize=12)
    ax.set_xlabel('Data')
    ax.set_ylabel('Resíduo')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _savefig(fig, '01_residuos_tempo', '1/8')

    # ── 2. Histograma dos Resíduos ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(data, bins=min(100, int(np.sqrt(n) * 2)),
            density=True, alpha=0.45, color='steelblue', label='Empírico')
    pdf_norm = sp_stats.norm.pdf(x_grid, *norm_params)
    ax.plot(x_grid, pdf_norm, 'k--', linewidth=1.5, label='Normal ajustada')
    ax.set_title(f'Histograma dos Resíduos\n{title_suffix}', fontsize=12)
    ax.set_xlabel('Resíduo')
    ax.set_ylabel('Densidade')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _savefig(fig, '02_histograma', '2/8')

    # ── 3. QQ-Plot Normal ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 7))
    probs = (np.arange(1, n + 1) - 0.5) / n
    q_emp = np.sort(data)
    q_teo = sp_stats.norm.ppf(probs, *norm_params)
    ax.scatter(q_teo, q_emp, s=4, alpha=0.5, color='steelblue', label='Dados')
    lo = min(q_teo.min(), q_emp.min())
    hi = max(q_teo.max(), q_emp.max())
    ax.plot([lo, hi], [lo, hi], 'r-', linewidth=1.5, label='Linha ideal (y=x)')
    ax.set_title(f'QQ-Plot (Normal)\n{title_suffix}', fontsize=11)
    ax.set_xlabel('Quantis teóricos (Normal)')
    ax.set_ylabel('Quantis empíricos')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _savefig(fig, '03_qqplot_normal', '3/8')

    # ── 4. Resíduos vs. Valores Ajustados ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(fitted, data, s=4, alpha=0.5, color='steelblue')
    ax.axhline(0, color='black', linewidth=1)
    ax.set_title(f'Resíduos vs. Valores Ajustados\n{title_suffix}', fontsize=12)
    ax.set_xlabel('Valores Ajustados (Previsão)')
    ax.set_ylabel('Resíduos')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _savefig(fig, '04_residuos_vs_ajustados', '4/8')

    # ── 5. ACF dos Resíduos ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_acf(data, ax=ax, lags=40, title=f'ACF dos Resíduos\n{title_suffix}')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _savefig(fig, '05_acf_residuos', '5/8')

    # ── 6. PACF dos Resíduos ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_pacf(data, ax=ax, lags=40,
              title=f'PACF dos Resíduos\n{title_suffix}', method='ywm')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _savefig(fig, '06_pacf_residuos', '6/8')

    # ── 7. ACF dos Resíduos ao Quadrado ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_acf(data**2, ax=ax, lags=40,
             title=f'ACF dos Resíduos ao Quadrado\n{title_suffix}')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _savefig(fig, '07_acf_residuos_quadrado', '7/8')

    # ── 8. Previsão vs Observado ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 8))
    lim_lo = min(fitted.min(), observed.min())
    lim_hi = max(fitted.max(), observed.max())
    ax.scatter(fitted, observed, s=6, alpha=0.4, color='steelblue', label='Dias')
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], 'r-', linewidth=1.5,
            label='Linha ideal (y=x)')
    corr = np.corrcoef(fitted, observed)[0, 1]
    ax.set_title(
        f'Previsão vs Observado  (r={corr:.4f})\n{title_suffix}',
        fontsize=12
    )
    ax.set_xlabel('Variância Prevista (σ̂²_t)')
    ax.set_ylabel('Variância Realizada (RV_t)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _savefig(fig, '08_previsao_vs_observado', '8/8')

    print(f"  Análise de resíduos concluída (8 gráficos) → {out_dir}")


def bootstrap_metric_ci(forecast_var, realized_var, n_boot=2000,
                         alpha=0.05, seed=42):
    """
    Intervalo de confiança (1-α) para MSE, MAE e QLIKE via bootstrap
    percentil (não-paramétrico).

    O bootstrap é preferível ao IC t-Student aqui porque:
      - As perdas L_t = (σ̂²_t - ξ_t)² são assimetricas e de cauda pesada
        (herdam a não-normalidade dos retornos financeiros).
      - Nenhuma hipótese distribucional é assumida.
      - Consistente sob heteroscedasticidade serial (comum em séries de perdas).

    Parameters
    ----------
    forecast_var, realized_var : array-like
        Variâncias previstas e realizadas (alinhadas).
    n_boot : int
        Número de reamostras bootstrap.
    alpha : float
        Nível de significância (default 0,05 → IC 95%).

    Returns
    -------
    dict com 'mse', 'mae', 'qlike', cada um contendo
        {'point': float, 'lower': float, 'upper': float, 'se': float}
    """
    rng = np.random.default_rng(seed)
    f = np.asarray(forecast_var, dtype=np.float64)
    r = np.asarray(realized_var, dtype=np.float64)
    n = len(f)

    mse_boot  = np.empty(n_boot)
    mae_boot  = np.empty(n_boot)
    qlike_boot = np.empty(n_boot)

    for b in _pbar(range(n_boot), desc="  Bootstrap CI", unit="it",
                   ncols=90, leave=False):
        idx = rng.integers(0, n, size=n)
        fb, rb = f[idx], r[idx]
        mse_boot[b]   = np.mean((fb - rb) ** 2)
        mae_boot[b]   = np.mean(np.abs(fb - rb))
        fb_safe = np.maximum(fb, 1e-20)
        qlike_boot[b] = np.mean(np.log(fb_safe) + rb / fb_safe)

    lo, hi = alpha / 2, 1 - alpha / 2
    result = {}
    for name, boot, point_fn in [
        ('mse',   mse_boot,   lambda: np.mean((f - r) ** 2)),
        ('mae',   mae_boot,   lambda: np.mean(np.abs(f - r))),
        ('qlike', qlike_boot, lambda: np.mean(np.log(np.maximum(f, 1e-20)) + r / np.maximum(f, 1e-20))),
    ]:
        result[name] = {
            'point': point_fn(),
            'lower': np.quantile(boot, lo),
            'upper': np.quantile(boot, hi),
            'se':    boot.std(ddof=1),
            'dist_boot': boot,          # guardado para histograma do bootstrap
        }
    return result


def plot_forecast_with_ci(forecast_var, date_index, realized_var,
                           bar_type_name, arma_order, garch_order,
                           ci_result, alpha=0.05):
    """
    Gera 2 gráficos INDEPENDENTES de análise de previsão:
      1. Série temporal: Prevista vs RV Benchmark com IC empírico
      2. Erro de previsão ao longo do tempo

    Os histogramas bootstrap são salvos separadamente via _save_bootstrap_plots().

    Todos os arquivos são salvos em analise_previsao/<bar_type_name>/.
    """
    f = np.asarray(forecast_var)
    r = np.asarray(realized_var)
    error = f - r

    window = min(20, len(f) // 4)
    err_series = pd.Series(error, index=date_index)
    roll_std = err_series.rolling(window, min_periods=5).std()
    z = sp_stats.norm.ppf(1 - alpha / 2)
    ci_upper = f + z * roll_std.values
    ci_lower = f - z * roll_std.values

    # Pasta de destino: analise_previsao/<bar_type_name>/
    out_dir = os.path.join(RESULTS_DIR, 'analise_previsao', bar_type_name)
    os.makedirs(out_dir, exist_ok=True)

    # ── 1. Série temporal com IC ─────────────────────────────────────────────
    fig, ax_ts = plt.subplots(figsize=(16, 6))
    ax_ts.plot(date_index, r, color='navy', linewidth=0.9, alpha=0.85,
               label='RV Benchmark')
    ax_ts.plot(date_index, f, color='crimson', linewidth=0.9, alpha=0.85,
               label=f'Previsão ARMA{arma_order}-GARCH{garch_order}')
    ax_ts.fill_between(date_index, ci_lower, ci_upper, alpha=0.18, color='crimson',
                       label=f'IC {int((1-alpha)*100)}% empírico (rolling σ, w={window}d)')
    ax_ts.set_title(
        f'Variância Diária: Prevista vs RV Benchmark\n{bar_type_name}  |  '
        f'ARMA{arma_order}-GARCH{garch_order}',
        fontsize=12
    )
    ax_ts.set_xlabel('Data')
    ax_ts.set_ylabel('Variância')
    ax_ts.legend(fontsize=9)
    ax_ts.grid(True, alpha=0.3)
    plt.tight_layout()
    p1 = os.path.join(out_dir, f'{bar_type_name}_01_serie_temporal_ic.png')
    plt.savefig(p1, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    [prev 1/2] Série temporal com IC salva: {p1}")

    # ── 2. Erro de previsão ──────────────────────────────────────────────────
    fig, ax_err = plt.subplots(figsize=(16, 5))
    ax_err.fill_between(date_index, error, 0,
                        where=error >= 0, color='seagreen', alpha=0.45,
                        label='Sobre-estimação')
    ax_err.fill_between(date_index, error, 0,
                        where=error < 0, color='firebrick', alpha=0.45,
                        label='Sub-estimação')
    ax_err.axhline(y=0, color='black', linewidth=0.6)
    ax_err.set_title(
        f'Erro de Previsão (Forecast − RV Benchmark)\n{bar_type_name}',
        fontsize=12
    )
    ax_err.set_ylabel('Erro')
    ax_err.set_xlabel('Data')
    ax_err.legend(fontsize=9)
    ax_err.grid(True, alpha=0.3)
    plt.tight_layout()
    p2 = os.path.join(out_dir, f'{bar_type_name}_02_erro_previsao.png')
    plt.savefig(p2, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    [prev 2/2] Erro de previsão salvo: {p2}")

    # ── Histogramas Bootstrap (pasta própria) ────────────────────────────────
    _save_bootstrap_plots(ci_result, bar_type_name)

    print(f"  Gráficos de previsão concluídos → {out_dir}")


def _save_bootstrap_plots(ci_result, bar_type_name):
    """
    Gera 3 histogramas INDEPENDENTES de distribuição bootstrap:
      1. Bootstrap MSE
      2. Bootstrap MAE
      3. Bootstrap QLIKE

    Todos os arquivos são salvos em bootstrap_ic/<bar_type_name>/.
    """
    out_dir = os.path.join(RESULTS_DIR, 'bootstrap_ic', bar_type_name)
    os.makedirs(out_dir, exist_ok=True)

    specs = [
        ('mse',   'MSE',   'steelblue',    '01'),
        ('mae',   'MAE',   'darkorange',   '02'),
        ('qlike', 'QLIKE', 'mediumpurple', '03'),
    ]

    for name, label, color, idx in specs:
        info = ci_result[name]
        boot = info['dist_boot']
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(boot, bins=50, density=True, color=color, alpha=0.7, label='Bootstrap')
        ax.axvline(info['point'], color='black', linewidth=1.8,
                   label=f'Pontual = {info["point"]:.3e}')
        ax.axvline(info['lower'], color='red', linewidth=1.4, linestyle='--',
                   label=f'IC 95% lower = {info["lower"]:.3e}')
        ax.axvline(info['upper'], color='red', linewidth=1.4, linestyle='--',
                   label=f'IC 95% upper = {info["upper"]:.3e}')
        ax.set_title(
            f'Distribuição Bootstrap — {label}\n'
            f'{bar_type_name}  |  n_boot=2000',
            fontsize=12, fontweight='bold'
        )
        ax.set_xlabel(label)
        ax.set_ylabel('Densidade')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        p = os.path.join(out_dir, f'{bar_type_name}_{idx}_bootstrap_{name}.png')
        plt.savefig(p, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"    [boot {idx}/03] Bootstrap {label} salvo: {p}")

    print(f"  Bootstrap IC concluído (3 gráficos) → {out_dir}")


# ==============================================================================
# FUNÇÕES AUXILIARES
# ==============================================================================
def load_rv_benchmark(filepath):
    """
    Carrega o benchmark de Volatilidade Realizada (RV) diário.

    Returns
    -------
    pd.Series
        Série com index=date (datetime) e valores de RV diária.
    """
    df = pd.read_parquet(filepath)
    rv = df['rv']
    rv.index = pd.to_datetime(rv.index)
    rv.index.name = 'date'
    print(f"  RV Benchmark carregado: {len(rv)} dias "
          f"({rv.index[0].date()} → {rv.index[-1].date()})")
    return rv


def aggregate_variance_to_daily(forecast_var, bar_index):
    """
    Agrega variância condicional intraday para o nível diário (soma por dia).

    A soma é a agregação correta: se os retornos intraday são condicionalmente
    independentes, a variância diária é a soma das variâncias por barra.
    Isso é análogo à construção do RV (soma de retornos ao quadrado).

    Parameters
    ----------
    forecast_var : array-like
        Variância condicional prevista por barra (intraday).
    bar_index : pd.DatetimeIndex
        Índice temporal das barras de teste.

    Returns
    -------
    pd.Series
        Variância prevista agregada por dia.
    """
    df = pd.DataFrame({
        'forecast_var': np.asarray(forecast_var),
        'date': bar_index.date
    })
    daily = df.groupby('date')['forecast_var'].sum()
    daily.index = pd.to_datetime(daily.index)
    daily.index.name = 'date'
    return daily


def train_test_split_temporal(df, train_ratio=0.80):
    """Split temporal simples: primeiros N% para treino, restante para teste."""
    n = len(df)
    split_idx = int(n * train_ratio)
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    print(f"  Split temporal: {len(train)} treino ({train_ratio*100:.0f}%) | "
          f"{len(test)} teste ({(1-train_ratio)*100:.0f}%)")
    print(f"  Treino: {train.index[0]} → {train.index[-1]}")
    print(f"  Teste:  {test.index[0]} → {test.index[-1]}")
    return train, test


def cpcv_model_selection(returns_train, cpcv, criterion='aic',
                          p_max=5, q_max=5, garch_m_max=2, garch_s_max=2):
    """
    Seleção de parâmetros ARMA-GARCH via CPCV (critério: AIC ou BIC).

    Se o backend Rust estiver disponível, executa TODOS os folds em paralelo
    em uma única chamada (massivamente mais rápido).
    """
    n_samples = len(returns_train)
    summary = cpcv.get_summary(n_samples)
    print(f"\n  CPCV configuração: {summary}")

    if RUST_AVAILABLE:
        return _cpcv_rust(returns_train, cpcv, criterion,
                          p_max, q_max, garch_m_max, garch_s_max)
    else:
        return _cpcv_python(returns_train, cpcv, criterion,
                            p_max, q_max, garch_m_max, garch_s_max)


def _expand_intervals_to_list(intervals):
    """
    Expande lista de intervalos [(start, end), ...] em lista plana de inteiros Python.
    Usa range() nativo — sem numpy, sem alocação de array int64.
    """
    result = []
    for s, e in intervals:
        result.extend(range(s, e))
    return result


def _cpcv_rust(returns_train, cpcv, criterion, p_max, q_max, gm_max, gs_max):
    """
    CPCV via Rust: streaming de intervalos + grid search em lotes.

    Otimização de memória: usa cpcv.split_intervals() que yield tuplas
    compactas (~160 bytes/fold) em vez de arrays numpy (~1 MB/fold).
    Os intervalos são expandidos para listas Python apenas no momento
    de enviar cada lote ao Rust, e liberados imediatamente após.

    Memória de pico: ~300 MB (1 batch de 200 folds expandidos)
    vs ~180 GB (142K folds como arrays numpy).
    """
    n_samples = len(returns_train)
    n_combos  = (p_max + 1) * (q_max + 1) * gm_max * gs_max
    total_fits = cpcv.n_splits * n_combos

    BATCH_SIZE = 10

    print(f"\n  CPCV Rust (Streaming Mode): {cpcv.n_splits:,} folds × {n_combos} "
          f"combinações = {total_fits:,} ajustes MLE")
    print(f"  (streaming por intervalos em lotes de {BATCH_SIZE} folds; "
          f"sem materialização de arrays numpy)")

    t_start = time.time()

    # Converter retornos para lista Python UMA VEZ e reutilizar em todos os batches
    returns_list = np.asarray(returns_train, dtype=np.float64).tolist()

    fold_results = []
    fold_infos = []

    # Buffers do lote atual (intervalos compactos — ~KBs por batch)
    batch_train_ivals = []
    batch_test_ivals  = []
    batch_infos       = []

    if _HAS_TQDM:
        pbar = _tqdm(total=cpcv.n_splits, desc="  CPCV Rust", unit="fold",
                     ncols=90, dynamic_ncols=True)

    for train_ivals, test_ivals, fold_info in cpcv.split_intervals(
        n_samples, progress=False
    ):
        batch_train_ivals.append(train_ivals)
        batch_test_ivals.append(test_ivals)
        batch_infos.append(fold_info)

        if len(batch_train_ivals) >= BATCH_SIZE:
            # Expandir intervalos → listas Python puras (sem numpy)
            train_lists = [_expand_intervals_to_list(iv) for iv in batch_train_ivals]
            test_lists  = [_expand_intervals_to_list(iv) for iv in batch_test_ivals]

            batch_res = cpcv_grid_search_rust(
                None, train_lists, test_lists,
                p_max, q_max, gm_max, gs_max, criterion,
                _returns_as_list=returns_list
            )
            fold_results.extend(batch_res)
            fold_infos.extend(batch_infos)

            if _HAS_TQDM:
                pbar.update(len(batch_train_ivals))

            # Liberar buffers do lote
            batch_train_ivals = []
            batch_test_ivals  = []
            batch_infos       = []

    # Lote residual
    if batch_train_ivals:
        train_lists = [_expand_intervals_to_list(iv) for iv in batch_train_ivals]
        test_lists  = [_expand_intervals_to_list(iv) for iv in batch_test_ivals]

        batch_res = cpcv_grid_search_rust(
            None, train_lists, test_lists,
            p_max, q_max, gm_max, gs_max, criterion,
            _returns_as_list=returns_list
        )
        fold_results.extend(batch_res)
        fold_infos.extend(batch_infos)

        if _HAS_TQDM:
            pbar.update(len(batch_train_ivals))

    if _HAS_TQDM:
        pbar.close()

    # Liberar lista de retornos (não mais necessária)
    del returns_list

    t_elapsed = time.time() - t_start
    rate = cpcv.n_splits / t_elapsed if t_elapsed > 0 else 0
    print(f"  CPCV Rust concluído em {t_elapsed:.1f}s  "
          f"({rate:.0f} folds/s  |  {total_fits/t_elapsed:.0f} fits/s)")

    # ── Agregar resultados ──────────────────────────────────────────────────
    rows = []
    for i, res in enumerate(
        _pbar(fold_results, desc="  Processando resultados",
              unit="fold", ncols=90, leave=False)
    ):
        if res.get('converged', False):
            rows.append({
                'fold'       : i + 1,
                'test_groups': fold_infos[i]['test_groups'],
                'arima_order': (res['arma_p'], 0, res['arma_q']),
                'garch_order': (res['garch_m'], res['garch_s']),
                'aic'        : res['aic'],
                'bic'        : res['bic'],
            })

    if not rows:
        raise RuntimeError("Nenhum fold do CPCV convergiu.")

    df_folds = pd.DataFrame(rows)
    print(f"\n  Folds convergidos: {len(rows):,} / {cpcv.n_splits:,}")

    # Encontrar parâmetros mais frequentes
    param_counts = df_folds.groupby(['arima_order', 'garch_order']).agg(
        count   =('fold', 'size'),
        mean_aic=('aic',  'mean'),
        std_aic =('aic',  'std'),
    ).reset_index()
    param_counts = param_counts.sort_values(['count', 'mean_aic'],
                                             ascending=[False, True])
    print(f"\n  Top-5 parâmetros por frequência ({criterion.upper()} médio):")
    print(param_counts.head(5).to_string(index=False))

    best_params = param_counts.iloc[0]
    best_arima  = best_params['arima_order']
    best_garch  = best_params['garch_order']

    print(f"\n  >>> Parâmetros selecionados: ARMA{best_arima}-GARCH{best_garch}")
    print(f"      Frequência: {best_params['count']:,}/{len(rows):,} folds  "
          f"({100*best_params['count']/len(rows):.1f}%)")

    return best_arima, best_garch, df_folds


def _cpcv_python(returns_train, cpcv, criterion,
                  p_max, q_max, gm_max, gs_max):
    """Fallback: CPCV em Python puro (lento). Seleção exclusivamente por AIC/BIC."""
    n_samples = len(returns_train)
    fold_results = []

    for fold_idx, (train_idx, test_idx, fold_info) in _pbar(
        enumerate(cpcv.split(n_samples, progress=False)),
        total=cpcv.n_splits,
        desc="  CPCV Python",
        unit="fold",
        ncols=90,
    ):
        print(f"  Fold {fold_idx + 1}/{cpcv.n_splits} "
              f"(test groups: {fold_info['test_groups']})")

        fold_returns_train = returns_train.iloc[train_idx]

        try:
            best_result = grid_search(
                fold_returns_train,
                p_max=p_max, q_max=q_max,
                garch_m_max=gm_max, garch_s_max=gs_max,
                criterion=criterion
            )

            fold_results.append({
                'fold': fold_idx + 1,
                'test_groups': fold_info['test_groups'],
                'arima_order': (best_result['best_arma_p'], 0,
                                best_result['best_arma_q']),
                'garch_order': (best_result['best_garch_m'],
                                best_result['best_garch_s']),
                'aic': best_result['best_aic'],
                'bic': best_result['best_bic'],
            })
        except Exception as e:
            print(f"      ERRO no fold {fold_idx + 1}: {e}")
            continue

    if not fold_results:
        raise RuntimeError("Nenhum fold do CPCV convergiu.")

    df_folds = pd.DataFrame(fold_results)
    print(f"\n  Resumo dos folds CPCV:")
    print(df_folds.to_string(index=False))

    param_counts = df_folds.groupby(['arima_order', 'garch_order']).agg(
        count=('fold', 'size'),
        mean_aic=('aic', 'mean'),
    ).reset_index()
    param_counts = param_counts.sort_values(['count', 'mean_aic'],
                                             ascending=[False, True])

    best_params = param_counts.iloc[0]
    best_arima = best_params['arima_order']
    best_garch = best_params['garch_order']

    print(f"\n  >>> Parâmetros selecionados: ARMA{best_arima}-GARCH{best_garch}")
    return best_arima, best_garch, df_folds


def run_pipeline(filepath, bar_type_name, rv_benchmark):
    """
    Executa o pipeline completo para um tipo de barra.

    Parameters
    ----------
    filepath : str
        Caminho do arquivo parquet com as barras.
    bar_type_name : str
        Nome descritivo do tipo de barra.
    rv_benchmark : pd.Series
        Volatilidade Realizada diária (benchmark consenso).
    """
    print(f"\n{'='*70}")
    print(f"PIPELINE: {bar_type_name}")
    print(f"{'='*70}")

    # 1. Carregar e calcular retornos logarítmicos
    print(f"\n[1/7] Carregando dados e calculando retornos logarítmicos...")
    df = pd.read_parquet(filepath)

    # Converter timestamp e definir índice temporal
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df.sort_index(inplace=True)
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df.dropna(subset=['log_return'], inplace=True)

    print(f"  Dados carregados: {len(df)} barras "
          f"({df.index[0]} → {df.index[-1]})")

    # 2. Split treino/teste
    print(f"\n[2/7] Dividindo treino/teste...")
    train, test = train_test_split_temporal(df, TRAIN_RATIO)

    returns_train = train['log_return']
    returns_test = test['log_return']

    # 3. CPCV para seleção de parâmetros
    print(f"\n[3/7] Seleção de parâmetros via CPCV...")
    cpcv = CombPurgedKFold(
        n_groups=CPCV_N_GROUPS,
        k_test=CPCV_K_TEST,
        embargo_bars=EMBARGO_BARS
    )

    best_arima, best_garch, cpcv_results = cpcv_model_selection(
        returns_train, cpcv,
        criterion=SELECTION_CRITERION,
        p_max=ARIMA_P_MAX, q_max=ARIMA_Q_MAX,
        garch_m_max=GARCH_P_MAX, garch_s_max=GARCH_Q_MAX
    )

    # Extrair ordens do ARMA (p, q) — ignorar d para o Rust
    arma_p = best_arima[0]
    arma_q = best_arima[2]
    garch_m = best_garch[0]
    garch_s = best_garch[1]

    # 4. Retreinar modelo final em TODO o treino
    print(f"\n[4/7] Retreinando modelo final ARMA({arma_p},{arma_q})-"
          f"GARCH({garch_m},{garch_s}) em todo o treino...")
    final_model = fit_model(returns_train, arma_p, arma_q, garch_m, garch_s)
    print(f"  AIC final: {final_model.get('aic', 'N/A'):.2f} | "
          f"BIC final: {final_model.get('bic', 'N/A'):.2f}")

    # 5. Previsão no teste
    print(f"\n[5/7] Gerando previsões da variância no teste ({len(returns_test)} obs)...")
    t_start = time.time()

    # Concatenar treino + teste e usar Rust para forecast com parâmetros fixos
    returns_full = pd.concat([returns_train, returns_test])
    n_train = len(returns_train)

    forecast_result = forecast_variance(
        returns_full, n_train, arma_p, arma_q, garch_m, garch_s
    )

    if forecast_result.get('converged', False):
        forecast_var = np.array(forecast_result['forecast_variance'])
    else:
        raise RuntimeError(f"Forecast falhou: {forecast_result.get('error', 'Unknown')}")

    t_elapsed = time.time() - t_start
    print(f"  Previsão concluída em {t_elapsed:.1f}s")

    # 5b. Agregar variância prevista de intraday → diário e alinhar com RV benchmark
    print(f"\n  Agregando variância prevista para nível diário...")
    daily_forecast = aggregate_variance_to_daily(forecast_var, test.index)
    print(f"  Dias com previsão: {len(daily_forecast)} "
          f"({daily_forecast.index[0].date()} → {daily_forecast.index[-1].date()})")

    # Alinhar com o benchmark (inner join — apenas dias presentes em ambos)
    aligned = pd.DataFrame({
        'forecast': daily_forecast,
        'rv': rv_benchmark
    }).dropna()
    print(f"  Dias alinhados com RV benchmark: {len(aligned)}")

    if len(aligned) == 0:
        raise RuntimeError(
            "Nenhum dia em comum entre as previsões e o RV benchmark. "
            "Verifique o período dos dados."
        )

    daily_forecast_aligned = aligned['forecast'].values
    rv_aligned = aligned['rv'].values

    # 6. Métricas (nível diário contra RV benchmark)
    print(f"\n[6/7] Calculando métricas de avaliação (vs RV Benchmark)...")
    metrics = compute_all_metrics(daily_forecast_aligned, rv_aligned,
                                  model_name=bar_type_name)

    print(f"  MSE:        {metrics['mse']:.2e}")
    print(f"  MAE:        {metrics['mae']:.2e}")
    print(f"  QLIKE:      {metrics['qlike']:.4f}")
    print(f"  R² (MZ):    {metrics['r_squared']:.4f}")
    print(f"  MZ α:       {metrics['mz_alpha']:.2e} (p={metrics['mz_alpha_pvalue']:.4f})")
    print(f"  MZ β:       {metrics['mz_beta']:.4f} (p={metrics['mz_beta_pvalue']:.4f})")

    # 7. Bootstrap CI das métricas
    print(f"\n[7/9] Calculando intervalos de confiança bootstrap (n_boot=2000)...")
    ci_result = bootstrap_metric_ci(daily_forecast_aligned, rv_aligned,
                                    n_boot=2000, alpha=0.05)
    for metric_name in ('mse', 'mae', 'qlike'):
        info = ci_result[metric_name]
        print(f"  {metric_name.upper():5s}: {info['point']:.4e}  "
              f"IC 95% [{info['lower']:.4e}, {info['upper']:.4e}]  "
              f"SE={info['se']:.4e}")

    # 8. Gráficos de previsão + IC + bootstrap
    print(f"\n[8/9] Gerando gráficos de previsão com IC e bootstrap...")
    plot_forecast_with_ci(
        daily_forecast_aligned, aligned.index, rv_aligned,
        bar_type_name, (arma_p, arma_q), (garch_m, garch_s),
        ci_result, alpha=0.05
    )


    # 9. Análise de resíduos da previsão: ê_t = σ̂²_t − RV_t
    print(f"\n[9/9] Análise de resíduos da previsão...")
    forecast_errors = daily_forecast_aligned - rv_aligned
    plot_residual_analysis(
        forecast_errors,
        daily_forecast_aligned,
        aligned.index,
        series_label='Resíduos de Previsão (σ̂²_t − RV_t)',
        bar_type_name=bar_type_name,
        filename_prefix='forecast_errors'
    )

    return {
        'bar_type': bar_type_name,
        'best_arima': best_arima,
        'best_garch': best_garch,
        'final_model': final_model,
        'forecast_var': daily_forecast_aligned,
        'realized_var': rv_aligned,
        'metrics': metrics,
        'ci_result': ci_result,
        'cpcv_results': cpcv_results,
        'test_index': aligned.index,
    }





def plot_cpcv_comparison(cpcv_dollar, cpcv_time):
    """
    Gera 2 gráficos INDEPENDENTES de comparação CPCV entre Dollar e Time bars:
      1. AIC por fold CPCV
      2. BIC por fold CPCV

    Todos os arquivos são salvos em comparacao_modelos/.
    """
    out_dir = os.path.join(RESULTS_DIR, 'comparacao_modelos')
    os.makedirs(out_dir, exist_ok=True)

    n_dollar = len(cpcv_dollar)
    n_time   = len(cpcv_time)

    specs = [
        ('aic', 'AIC', '01'),
        ('bic', 'BIC', '02'),
    ]

    for metric, label, idx in specs:
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.bar(np.arange(n_dollar) - 0.2, cpcv_dollar[metric],
               width=0.4, label='Dollar Bars', alpha=0.8, color='steelblue')
        ax.bar(np.arange(n_time) + 0.2, cpcv_time[metric],
               width=0.4, label='Time Bars', alpha=0.8, color='darkorange')
        ax.set_xlabel('Fold CPCV')
        ax.set_ylabel(label)
        ax.set_title(f'{label} por Fold CPCV — Dollar Bars vs Time Bars', fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        p = os.path.join(out_dir, f'cpcv_{idx}_{metric}_por_fold.png')
        plt.savefig(p, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    [cpcv {idx}/02] {label} por fold salvo: {p}")

    print(f"  Comparação CPCV concluída (2 gráficos) → {out_dir}")


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.random.seed(RANDOM_SEED)

    # Logger
    log_path = os.path.join(RESULTS_DIR, "pipeline_log.txt")
    sys.stdout = Logger(log_path)

    n_folds = math.comb(CPCV_N_GROUPS, CPCV_K_TEST)
    print("=" * 70)
    print("PIPELINE ARIMA-GARCH: Dollar Bars vs Time Bars (5 min)")
    print(f"Diretório de saída: {RESULTS_DIR}")
    print(f"Benchmark: Volatilidade Realizada (RV Consenso)")
    print(f"Backend: {'Rust (arma_garch_cv)' if RUST_AVAILABLE else 'Python (arch)'}")
    print(f"CPCV: N={CPCV_N_GROUPS}, K={CPCV_K_TEST}  →  C({CPCV_N_GROUPS},{CPCV_K_TEST}) = {n_folds:,} folds")
    print(f"Grid: ARMA(0..{ARIMA_P_MAX}, 0..{ARIMA_Q_MAX}) × GARCH(1..{GARCH_P_MAX}, 1..{GARCH_Q_MAX})")
    n_combos = (ARIMA_P_MAX + 1) * (ARIMA_Q_MAX + 1) * GARCH_P_MAX * GARCH_Q_MAX
    print(f"Total de ajustes MLE por pipeline: {n_folds:,} × {n_combos} = {n_folds * n_combos:,}")
    print(f"AVISO: volume computacional elevado — backend Rust obrigatório.")
    print(f"Início: {pd.Timestamp.now()}")
    print("=" * 70)

    # Carregar RV benchmark uma única vez
    print(f"\n[0] Carregando RV Benchmark...")
    rv_benchmark = load_rv_benchmark(FILE_RV_BENCHMARK)

    t_total_start = time.time()

    # ==========================================
    # EXECUTAR PIPELINE PARA CADA TIPO DE BARRA
    # ==========================================
    results_dollar = run_pipeline(FILE_DOLLAR_5M, "Dollar_Bars_5m", rv_benchmark)
    results_time   = run_pipeline(FILE_TIME_5M,   "Time_Bars_5m",   rv_benchmark)

    # ==========================================
    # INTERSEÇÃO DE DATAS: garante mesmo período para comparação
    # ==========================================
    print(f"\n{'='*70}")
    print("ALINHAMENTO DE DATAS: Dollar Bars vs Time Bars")
    print(f"{'='*70}")

    idx_dollar = results_dollar['test_index']
    idx_time   = results_time['test_index']
    common_dates = idx_dollar.intersection(idx_time)

    print(f"  Dias no teste Dollar Bars : {len(idx_dollar)}"
          f"  ({idx_dollar[0].date()} → {idx_dollar[-1].date()})")
    print(f"  Dias no teste Time Bars   : {len(idx_time)}"
          f"  ({idx_time[0].date()} → {idx_time[-1].date()})")
    print(f"  Dias em COMUM (interseção): {len(common_dates)}"
          f"  ({common_dates[0].date()} → {common_dates[-1].date()})")

    if len(common_dates) == 0:
        raise RuntimeError(
            "Nenhum dia em comum entre os testes de Dollar e Time bars. "
            "Verifique os períodos dos dados de entrada."
        )

    if len(common_dates) < len(idx_dollar) or len(common_dates) < len(idx_time):
        print(f"  [AVISO] Períodos de teste diferiram — usando apenas os "
              f"{len(common_dates)} dias comuns para a comparação final.")

    # Filtrar Dollar bars para datas comuns
    mask_dollar = pd.Series(idx_dollar).isin(common_dates).values
    fcast_dollar_common = results_dollar['forecast_var'][mask_dollar]
    rv_dollar_common    = results_dollar['realized_var'][mask_dollar]

    # Filtrar Time bars para datas comuns
    mask_time = pd.Series(idx_time).isin(common_dates).values
    fcast_time_common = results_time['forecast_var'][mask_time]
    rv_time_common    = results_time['realized_var'][mask_time]

    # Recalcular métricas sobre o período comum
    from metricas import compute_all_metrics
    metrics_dollar_common = compute_all_metrics(fcast_dollar_common, rv_dollar_common,
                                                model_name="Dollar_Bars_5m_common")
    metrics_time_common   = compute_all_metrics(fcast_time_common,   rv_time_common,
                                                model_name="Time_Bars_5m_common")

    # Recalcular IC bootstrap sobre o período comum
    print(f"\n  Recalculando IC bootstrap sobre período comum (n_boot=2000)...")
    ci_dollar_common = bootstrap_metric_ci(fcast_dollar_common, rv_dollar_common,
                                           n_boot=2000, alpha=0.05)
    ci_time_common   = bootstrap_metric_ci(fcast_time_common,   rv_time_common,
                                           n_boot=2000, alpha=0.05)

    # ==========================================
    # COMPARAÇÃO FINAL (período comum)
    # ==========================================
    print(f"\n{'='*70}")
    print("COMPARAÇÃO FINAL: Dollar Bars vs Time Bars (período comum)")
    print(f"Benchmark: Volatilidade Realizada (RV Consenso)")
    print(f"{'='*70}")

    comp_table, dm_result = comparison_table(
        metrics_dollar_common,
        metrics_time_common,
        fcast_dollar_common,
        fcast_time_common,
        rv_dollar_common,
        rv_time_common,
    )

    print(f"\n--- Tabela Comparativa de Métricas (vs RV Benchmark) — {len(common_dates)} dias comuns ---")
    print(comp_table.to_string())

    # IC bootstrap resumidos (período comum)
    print(f"\n--- Intervalos de Confiança Bootstrap 95% (n_boot=2000, período comum) ---")
    for bar_label, ci_res in [("Dollar Bars", ci_dollar_common), ("Time Bars", ci_time_common)]:
        print(f"  {bar_label}:")
        for m in ('mse', 'mae', 'qlike'):
            info = ci_res[m]
            print(f"    {m.upper():5s} = {info['point']:.4e}  "
                  f"[{info['lower']:.4e}, {info['upper']:.4e}]  SE={info['se']:.4e}")

    print(f"\n--- Teste de Diebold-Mariano (Dollar vs Time) — período comum ---")
    print(f"  DM Statistic: {dm_result['dm_stat']:.4f}")
    print(f"  p-value:      {dm_result['p_value']:.4f}")
    print(f"  Conclusão:    {dm_result['conclusion']}")

    print(f"\n--- Modelos Selecionados ---")
    print(f"  Dollar Bars: ARMA{results_dollar['best_arima']}-"
          f"GARCH{results_dollar['best_garch']}")
    print(f"  Time Bars:   ARMA{results_time['best_arima']}-"
          f"GARCH{results_time['best_garch']}")

    # Gráficos comparativos
    plot_cpcv_comparison(results_dollar['cpcv_results'],
                         results_time['cpcv_results'])

    # Salvar resultados
    csv_path = os.path.join(RESULTS_DIR, "comparacao_metricas.csv")
    comp_table.to_csv(csv_path)
    print(f"\n  Tabela salva: {csv_path}")

    # Salvar IC bootstrap em CSV (período comum)
    ci_rows = []
    for bar_label, ci_res in [("Dollar_Bars_5m", ci_dollar_common),
                               ("Time_Bars_5m",   ci_time_common)]:
        for m in ('mse', 'mae', 'qlike'):
            info = ci_res[m]
            ci_rows.append({
                'modelo': bar_label,
                'metrica': m.upper(),
                'pontual': info['point'],
                'ic_lower_95': info['lower'],
                'ic_upper_95': info['upper'],
                'se_bootstrap': info['se'],
                'n_dias_comuns': len(common_dates),
            })
    ci_df = pd.DataFrame(ci_rows)
    ci_csv = os.path.join(RESULTS_DIR, "ic_bootstrap_metricas.csv")
    ci_df.to_csv(ci_csv, index=False)
    print(f"  IC Bootstrap (período comum) salvo: {ci_csv}")

    for name, res in [("dollar", results_dollar), ("time", results_time)]:
        cpcv_csv = os.path.join(RESULTS_DIR, f"cpcv_folds_{name}.csv")
        res['cpcv_results'].to_csv(cpcv_csv, index=False)
        print(f"  CPCV folds salvo: {cpcv_csv}")

    t_total = time.time() - t_total_start
    print(f"\n{'='*70}")
    print(f"Pipeline concluído em {t_total/60:.1f} minutos")
    print(f"Resultados salvos em: {RESULTS_DIR}")
    print(f"{'='*70}")

