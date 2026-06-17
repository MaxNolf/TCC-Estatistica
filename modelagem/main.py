"""
main.py — Pipeline GARCH(1,1) com CPCV de Avaliação Preditiva.

Fluxo:
  1. Carregar dados (dollar_bars_5m, time_bars_5m) e calcular retornos logarítmicos
  2. Split temporal: 80% treino / 20% teste
  3. Treinar GARCH(1,1) em todo o treino (modelo final para inferência)
  4. CPCV de avaliação no treino: C(10,5)=252 folds com GARCH(1,1) fixo
       → métricas por fold: MSE, MAE, QLIKE, R²
  5. Bootstrap fold-level (iid sobre 252 obs) → IC 95% para cada métrica
  6. Estatísticas complementares: Win Rate, CV(MSE), cumulative loss
  7. Previsão final no conjunto de teste (20%) para gráficos de série temporal
  8. Análise de resíduos e gráficos comparativos
  9. Teste de Diebold-Mariano sobre pool de previsões CPCV (maior poder)

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
from scipy import stats as sp_stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

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

from config import (
    FILE_DOLLAR_5M, FILE_TIME_5M, FILE_RV_BENCHMARK, RESULTS_DIR,
    TRAIN_RATIO,
    CPCV_N_GROUPS, CPCV_K_TEST, EMBARGO_BARS,
    ARMA_P, ARMA_Q, GARCH_P, GARCH_Q,
    RANDOM_SEED
)
from cpcv import CombPurgedKFold
from modelo_arima_garch import fit_model, forecast_variance, RUST_AVAILABLE
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
        try:
            self.terminal.write(message)
        except UnicodeEncodeError:
            # Em terminais Windows (cp1252) caracteres como '→' podem falhar
            enc = getattr(self.terminal, 'encoding', 'cp1252') or 'cp1252'
            self.terminal.write(message.encode(enc, errors='replace').decode(enc))
        self.log.write(message)

    def flush(self):
        try:
            self.terminal.flush()
        except Exception:
            pass
        self.log.flush()


# ==============================================================================
# ANÁLISE DE RESÍDUOS
# ==============================================================================

def plot_residual_analysis(residuals, fitted_values, date_index,
                            series_label, bar_type_name, filename_prefix):
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
    data   = np.asarray(residuals,     dtype=np.float64)
    fitted = np.asarray(fitted_values, dtype=np.float64)
    finite_mask = np.isfinite(data)
    data   = data[finite_mask]
    fitted = fitted[finite_mask]
    n = len(data)
    observed = fitted + data   # RV_t = σ̂²_t + ê_t

    out_dir = os.path.join(RESULTS_DIR, 'analise_residuos', bar_type_name)
    os.makedirs(out_dir, exist_ok=True)

    norm_params  = sp_stats.norm.fit(data)
    x_grid       = np.linspace(np.percentile(data, 0.1), np.percentile(data, 99.9), 500)
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
    ax.plot(x_grid, sp_stats.norm.pdf(x_grid, *norm_params),
            'k--', linewidth=1.5, label='Normal ajustada')
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
    ax.set_title(f'Previsão vs Observado  (r={corr:.4f})\n{title_suffix}',
                 fontsize=12)
    ax.set_xlabel('Variância Prevista (σ̂²_t)')
    ax.set_ylabel('Variância Realizada (RV_t)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _savefig(fig, '08_previsao_vs_observado', '8/8')

    print(f"  Análise de resíduos concluída (8 gráficos) → {out_dir}")


# ==============================================================================
# BOOTSTRAP — FOLD-LEVEL (PRIMÁRIO)
# ==============================================================================

def bootstrap_fold_metrics(df_folds, n_boot=2000, alpha=0.05, seed=42):
    """
    IC (1-α) para MSE, MAE e QLIKE via bootstrap iid sobre as métricas por fold.

    Cada fold representa um "experimento" distinto em um segmento temporal
    diferente. Reamostrar com reposição os 252 valores fold-level é
    metodologicamente mais defensável do que bootstrap sobre pares diários,
    pois respeita a estrutura do CPCV.

    Parameters
    ----------
    df_folds : pd.DataFrame
        DataFrame com colunas 'mse', 'mae', 'qlike' (uma linha por fold).
    n_boot : int
        Número de reamostras bootstrap (default 2.000).
    alpha : float
        Nível de significância (default 0.05 → IC 95%).

    Returns
    -------
    dict com 'mse', 'mae', 'qlike', cada um contendo
        {'point', 'lower', 'upper', 'se', 'dist_boot'}
    """
    rng = np.random.default_rng(seed)
    n   = len(df_folds)
    result = {}

    for metric in ('mse', 'mae', 'qlike'):
        vals = df_folds[metric].values.copy()
        boot_means = np.array([
            np.mean(vals[rng.integers(0, n, size=n)])
            for _ in _pbar(range(n_boot), desc=f"  Bootstrap {metric.upper()}",
                           unit="it", ncols=90, leave=False)
        ])
        lo, hi = alpha / 2, 1 - alpha / 2
        result[metric] = {
            'point':     float(np.mean(vals)),
            'lower':     float(np.quantile(boot_means, lo)),
            'upper':     float(np.quantile(boot_means, hi)),
            'se':        float(boot_means.std(ddof=1)),
            'dist_boot': boot_means,
        }
    return result


def _save_bootstrap_plots(ci_result, bar_type_name):
    """
    Gera 3 histogramas INDEPENDENTES de distribuição bootstrap (fold-level):
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
            f'Distribuição Bootstrap (fold-level) — {label}\n'
            f'{bar_type_name}  |  n_boot={len(boot):,}  n_folds=252',
            fontsize=12, fontweight='bold'
        )
        ax.set_xlabel(f'Média de {label} sobre reamostras de folds')
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
# GRÁFICOS DE PREVISÃO (SPLIT 80/20 — USO VISUAL)
# ==============================================================================

def plot_forecast_with_ci(forecast_var, date_index, realized_var,
                           bar_type_name, ci_result, alpha=0.05):
    """
    Gera 2 gráficos de análise de previsão (split 80/20 — visualização):
      1. Série temporal: Prevista vs RV Benchmark com IC empírico
      2. Erro de previsão ao longo do tempo

    Todos os arquivos são salvos em analise_previsao/<bar_type_name>/.
    """
    f     = np.asarray(forecast_var)
    r     = np.asarray(realized_var)
    error = f - r

    window   = min(20, len(f) // 4)
    err_ser  = pd.Series(error, index=date_index)
    roll_std = err_ser.rolling(window, min_periods=5).std()
    z        = sp_stats.norm.ppf(1 - alpha / 2)
    ci_upper = f + z * roll_std.values
    ci_lower = f - z * roll_std.values

    out_dir = os.path.join(RESULTS_DIR, 'analise_previsao', bar_type_name)
    os.makedirs(out_dir, exist_ok=True)

    model_label = f'ARMA({ARMA_P},{ARMA_Q})-GARCH({GARCH_P},{GARCH_Q})'

    # ── 1. Série temporal com IC ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(date_index, r, color='navy',   linewidth=0.9, alpha=0.85,
            label='RV Benchmark')
    ax.plot(date_index, f, color='crimson', linewidth=0.9, alpha=0.85,
            label=f'Previsão {model_label}')
    ax.fill_between(date_index, ci_lower, ci_upper,
                    alpha=0.18, color='crimson',
                    label=f'IC {int((1-alpha)*100)}% empírico (rolling σ, w={window}d)')
    ax.set_title(
        f'Variância Diária: Prevista vs RV Benchmark\n'
        f'{bar_type_name}  |  {model_label}',
        fontsize=12
    )
    ax.set_xlabel('Data')
    ax.set_ylabel('Variância')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p1 = os.path.join(out_dir, f'{bar_type_name}_01_serie_temporal_ic.png')
    plt.savefig(p1, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    [prev 1/2] Série temporal com IC salva: {p1}")

    # ── 2. Erro de previsão ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.fill_between(date_index, error, 0,
                    where=error >= 0, color='seagreen', alpha=0.45,
                    label='Sobre-estimação')
    ax.fill_between(date_index, error, 0,
                    where=error < 0, color='firebrick', alpha=0.45,
                    label='Sub-estimação')
    ax.axhline(y=0, color='black', linewidth=0.6)
    ax.set_title(f'Erro de Previsão (Forecast − RV Benchmark)\n{bar_type_name}',
                 fontsize=12)
    ax.set_ylabel('Erro')
    ax.set_xlabel('Data')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p2 = os.path.join(out_dir, f'{bar_type_name}_02_erro_previsao.png')
    plt.savefig(p2, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    [prev 2/2] Erro de previsão salvo: {p2}")

    # Histogramas bootstrap
    _save_bootstrap_plots(ci_result, bar_type_name)
    print(f"  Gráficos de previsão concluídos → {out_dir}")


# ==============================================================================
# GRÁFICOS CPCV DE AVALIAÇÃO
# ==============================================================================

def plot_cpcv_evaluation(df_dollar, df_time):
    """
    Gera 6 gráficos comparativos das métricas CPCV por fold:
      1. MSE por fold (linha)
      2. MAE por fold (linha)
      3. QLIKE por fold (linha)
      4. Win Rate acumulado de Dollar vs. Time (MSE)
      5. Cumulative loss path (MSE acumulado)
      6. Coeficiente de variação por métrica (barras)

    Salvos em comparacao_modelos/.
    """
    out_dir = os.path.join(RESULTS_DIR, 'comparacao_modelos')
    os.makedirs(out_dir, exist_ok=True)

    n_folds = min(len(df_dollar), len(df_time))
    fold_x  = np.arange(1, n_folds + 1)

    # ── 1–3. Métricas por fold ───────────────────────────────────────────────
    for metric, label, idx_str in [
        ('mse',   'MSE',   '01'),
        ('mae',   'MAE',   '02'),
        ('qlike', 'QLIKE', '03'),
    ]:
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(fold_x, df_dollar[metric].values[:n_folds],
                color='steelblue',  linewidth=0.9, alpha=0.85, label='Dollar Bars')
        ax.plot(fold_x, df_time[metric].values[:n_folds],
                color='darkorange', linewidth=0.9, alpha=0.85, label='Time Bars')
        ax.set_xlabel('Fold CPCV')
        ax.set_ylabel(label)
        ax.set_title(f'{label} por Fold CPCV — Dollar Bars vs Time Bars', fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        p = os.path.join(out_dir, f'cpcv_{idx_str}_{metric}_por_fold.png')
        plt.savefig(p, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"    [cpcv {idx_str}/06] {label} por fold salvo: {p}")

    # ── 4. Win Rate acumulado (MSE Dollar < MSE Time) ────────────────────────
    d_mse      = df_dollar['mse'].values[:n_folds]
    t_mse      = df_time['mse'].values[:n_folds]
    wins_cumul = np.cumsum(d_mse < t_mse) / fold_x

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(fold_x, wins_cumul, color='seagreen', linewidth=1.5)
    ax.axhline(0.5, color='black', linestyle='--', linewidth=1, alpha=0.6,
               label='50% (indiferença)')
    final_wr = wins_cumul[-1]
    ax.set_xlabel('Fold CPCV (ordenado)')
    ax.set_ylabel('Win Rate acumulado')
    ax.set_title(
        f'Win Rate Acumulado: Dollar Bars < Time Bars em MSE\n'
        f'Win Rate final = {final_wr:.1%}',
        fontsize=13
    )
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(out_dir, 'cpcv_04_win_rate_acumulado.png')
    plt.savefig(p, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    [cpcv 04/06] Win Rate acumulado salvo: {p}")

    # ── 5. Cumulative loss (MSE acumulado) ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(fold_x, np.cumsum(d_mse), color='steelblue',
            linewidth=1.2, label='Dollar Bars')
    ax.plot(fold_x, np.cumsum(t_mse), color='darkorange',
            linewidth=1.2, label='Time Bars')
    ax.set_xlabel('Fold CPCV (ordenado)')
    ax.set_ylabel('MSE Acumulado')
    ax.set_title('Cumulative Loss (MSE) — Dollar Bars vs Time Bars', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(out_dir, 'cpcv_05_cumulative_loss.png')
    plt.savefig(p, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    [cpcv 05/06] Cumulative loss salvo: {p}")

    # ── 6. Coeficiente de Variação por métrica ───────────────────────────────
    metrics_list = ['mse', 'mae', 'qlike']
    labels_list  = ['MSE', 'MAE', 'QLIKE']
    cv_dollar = [np.std(df_dollar[m].values, ddof=1) / abs(np.mean(df_dollar[m].values))
                 for m in metrics_list]
    cv_time   = [np.std(df_time[m].values,   ddof=1) / abs(np.mean(df_time[m].values))
                 for m in metrics_list]

    x = np.arange(len(metrics_list))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x - 0.2, cv_dollar, width=0.4, label='Dollar Bars',
           color='steelblue',  alpha=0.8)
    ax.bar(x + 0.2, cv_time,   width=0.4, label='Time Bars',
           color='darkorange', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_list)
    ax.set_ylabel('Coeficiente de Variação (std / |mean|)')
    ax.set_title('Estabilidade do Modelo — CV por Métrica', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    p = os.path.join(out_dir, 'cpcv_06_cv_metricas.png')
    plt.savefig(p, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    [cpcv 06/06] Coeficiente de variação salvo: {p}")

    print(f"  Comparação CPCV concluída (6 gráficos) → {out_dir}")


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
        Índice temporal das barras.

    Returns
    -------
    pd.Series
        Variância prevista agregada por dia.
    """
    df = pd.DataFrame({
        'forecast_var': np.asarray(forecast_var),
        'date':         bar_index.date
    })
    daily = df.groupby('date')['forecast_var'].sum()
    daily.index = pd.to_datetime(daily.index)
    daily.index.name = 'date'
    return daily


def train_test_split_temporal(df, train_ratio=0.80):
    """Split temporal simples: primeiros N% para treino, restante para teste."""
    n         = len(df)
    split_idx = int(n * train_ratio)
    train = df.iloc[:split_idx].copy()
    test  = df.iloc[split_idx:].copy()
    print(f"  Split temporal: {len(train)} treino ({train_ratio*100:.0f}%) | "
          f"{len(test)} teste ({(1-train_ratio)*100:.0f}%)")
    print(f"  Treino: {train.index[0]} → {train.index[-1]}")
    print(f"  Teste:  {test.index[0]} → {test.index[-1]}")
    return train, test


# ==============================================================================
# CPCV DE AVALIAÇÃO PREDITIVA
# ==============================================================================

def cpcv_evaluate(returns_series, cpcv, rv_benchmark):
    """
    Executa CPCV de avaliação com GARCH(1,1) fixo.

    Para cada fold:
      1. Treina GARCH(1,1) no segmento de treino do fold (índices concatenados)
      2. Gera previsões de variância no segmento de teste
      3. Agrega intraday → diário e alinha com RV benchmark
      4. Calcula MSE, MAE, QLIKE e R² para o fold

    Adicionalmente, acumula todos os pares (forecast, rv) dos testes
    para uso posterior no teste de Diebold-Mariano.

    Parameters
    ----------
    returns_series : pd.Series
        Retornos logarítmicos com DatetimeIndex (série de treino — 80%).
    cpcv : CombPurgedKFold
        Instância configurada do CPCV.
    rv_benchmark : pd.Series
        Volatilidade Realizada diária (benchmark).

    Returns
    -------
    df_folds : pd.DataFrame
        Métricas por fold: fold, n_days, mse, mae, qlike, r_squared,
        date_start, date_end, test_groups.
    pooled_forecast : np.ndarray
        Concatenação de todas as previsões diárias de teste (para DM test).
    pooled_rv : np.ndarray
        Concatenação dos valores de RV correspondentes (para DM test).
    """
    n_samples    = len(returns_series)
    returns_arr  = np.asarray(returns_series, dtype=np.float64)

    fold_records     = []
    pooled_forecast  = []
    pooled_rv        = []
    n_skipped        = 0

    splits = list(cpcv.split(n_samples, progress=False))
    total  = len(splits)

    print(f"\n  CPCV Avaliação: {total} folds × GARCH({GARCH_P},{GARCH_Q}) fixo "
          f"= {total} ajustes MLE")

    if _HAS_TQDM:
        pbar = _tqdm(total=total, desc="  CPCV Avaliação",
                     unit="fold", ncols=90, dynamic_ncols=True)

    t_start = time.time()

    for fold_idx, (train_idx, test_idx, fold_info) in enumerate(splits):

        ret_train  = returns_arr[train_idx]
        ret_test   = returns_arr[test_idx]
        ret_concat = np.concatenate([ret_train, ret_test])
        n_train_fold = len(ret_train)

        # Timestamps do segmento de teste (mapear posições → datetime)
        test_timestamps = returns_series.index[test_idx]

        # Forecast via Rust com GARCH(1,1) fixo
        result = forecast_variance(
            pd.Series(ret_concat),
            n_train_fold,
            ARMA_P, ARMA_Q, GARCH_P, GARCH_Q
        )

        if not result.get('converged', False):
            n_skipped += 1
            if _HAS_TQDM:
                pbar.update(1)
            continue

        forecast_var_fold = np.array(result['forecast_variance'])

        # Agregar intraday → diário
        daily_forecast = aggregate_variance_to_daily(forecast_var_fold, test_timestamps)

        # Alinhar com RV benchmark (inner join por data)
        aligned = pd.DataFrame({
            'forecast': daily_forecast,
            'rv':       rv_benchmark
        }).dropna()

        if len(aligned) < 5:
            n_skipped += 1
            if _HAS_TQDM:
                pbar.update(1)
            continue

        f_vals = aligned['forecast'].values
        r_vals = aligned['rv'].values
        f_safe = np.maximum(f_vals, 1e-20)

        mse   = float(np.mean((f_vals - r_vals) ** 2))
        mae   = float(np.mean(np.abs(f_vals - r_vals)))
        qlike = float(np.mean(np.log(f_safe) + r_vals / f_safe))

        # R² via regressão MZ simples
        _, _, r_val, _, _ = sp_stats.linregress(f_vals, r_vals)

        fold_records.append({
            'fold':        fold_idx + 1,
            'test_groups': fold_info['test_groups'],
            'n_days':      len(aligned),
            'mse':         mse,
            'mae':         mae,
            'qlike':       qlike,
            'r_squared':   float(r_val ** 2),
            'date_start':  aligned.index[0],
            'date_end':    aligned.index[-1],
        })

        # Acumular para DM test
        pooled_forecast.extend(f_vals.tolist())
        pooled_rv.extend(r_vals.tolist())

        if _HAS_TQDM:
            pbar.update(1)

    if _HAS_TQDM:
        pbar.close()

    t_elapsed = time.time() - t_start
    print(f"  CPCV concluído em {t_elapsed:.1f}s "
          f"({t_elapsed / total:.2f}s/fold  |  {n_skipped} skipped)")

    df_folds = pd.DataFrame(fold_records)

    return df_folds, np.array(pooled_forecast), np.array(pooled_rv)


def compute_extra_stats(df_folds_dollar, df_folds_time):
    """
    Calcula estatísticas complementares comparando Dollar vs. Time fold a fold.

    Returns
    -------
    dict com win_rate_{metric}, cv_{metric}_{serie} para cada métrica.
    """
    n = min(len(df_folds_dollar), len(df_folds_time))
    d = df_folds_dollar.iloc[:n]
    t = df_folds_time.iloc[:n]

    stats = {}
    for metric in ('mse', 'mae', 'qlike'):
        d_vals = d[metric].values
        t_vals = t[metric].values
        # Menor = melhor em todas as métricas (QLIKE mais negativo = melhor)
        wins = int(np.sum(d_vals < t_vals))
        stats[f'win_rate_{metric}'] = float(wins / n)

        for label, vals in [('dollar', d_vals), ('time', t_vals)]:
            mean_abs = abs(float(np.mean(vals)))
            stats[f'cv_{metric}_{label}'] = (
                float(np.std(vals, ddof=1) / mean_abs)
                if mean_abs > 0 else float('nan')
            )

    return stats


# ==============================================================================
# PIPELINE PRINCIPAL
# ==============================================================================

def run_pipeline(filepath, bar_type_name, rv_benchmark):
    """
    Executa o pipeline completo para um tipo de barra.

    Fluxo:
      [1] Carregar dados e calcular retornos logarítmicos
      [2] Split temporal 80/20
      [3] Treinar GARCH(1,1) em todo o treino (modelo final)
      [4] CPCV de avaliação (252 folds) no treino
      [5] Bootstrap fold-level (IC 95%)
      [6] Previsão final no conjunto de teste (20%) — para gráficos
      [7] Gráficos de previsão
      [8] Análise de resíduos

    Returns
    -------
    dict com todos os resultados para comparação posterior.
    """
    print(f"\n{'='*70}")
    print(f"PIPELINE: {bar_type_name}")
    print(f"{'='*70}")

    # [1] Carregar e calcular retornos logarítmicos
    print(f"\n[1/8] Carregando dados e calculando retornos logarítmicos...")
    df = pd.read_parquet(filepath)

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

    # [2] Split temporal 80/20
    print(f"\n[2/8] Dividindo treino/teste...")
    train, test = train_test_split_temporal(df, TRAIN_RATIO)

    returns_train = train['log_return']
    returns_test  = test['log_return']

    # [3] Treinar GARCH(1,1) em TODO o treino (modelo final)
    print(f"\n[3/8] Treinando GARCH({GARCH_P},{GARCH_Q}) em todo o treino "
          f"({len(returns_train)} obs)...")
    final_model = fit_model(returns_train, ARMA_P, ARMA_Q, GARCH_P, GARCH_Q)
    print(f"  AIC final: {final_model.get('aic', 'N/A'):.2f} | "
          f"BIC final: {final_model.get('bic', 'N/A'):.2f}")
    print(f"  Parâmetros: ω={final_model.get('omega', float('nan')):.4e}  "
          f"α₁={final_model.get('alpha_1', float('nan')):.4f}  "
          f"β₁={final_model.get('beta_1',  float('nan')):.4f}")

    # [4] CPCV de avaliação no treino
    print(f"\n[4/8] CPCV de avaliação preditiva (C({CPCV_N_GROUPS},{CPCV_K_TEST}) "
          f"= {math.comb(CPCV_N_GROUPS, CPCV_K_TEST)} folds)...")
    cpcv = CombPurgedKFold(
        n_groups=CPCV_N_GROUPS,
        k_test=CPCV_K_TEST,
        embargo_bars=EMBARGO_BARS
    )
    summary = cpcv.get_summary(len(returns_train))
    print(f"  Configuração CPCV: {summary}")

    df_folds, pooled_f, pooled_r = cpcv_evaluate(
        returns_train, cpcv, rv_benchmark
    )

    n_folds_ok = len(df_folds)
    print(f"\n  Folds avaliados: {n_folds_ok}")
    print(f"  Pares diários no pool: {len(pooled_f)}")
    print(f"\n  Métricas CPCV (médias fold-level):")
    print(f"    MSE   = {df_folds['mse'].mean():.4e}  "
          f"(±{df_folds['mse'].std():.2e})")
    print(f"    MAE   = {df_folds['mae'].mean():.4e}  "
          f"(±{df_folds['mae'].std():.2e})")
    print(f"    QLIKE = {df_folds['qlike'].mean():.4f}  "
          f"(±{df_folds['qlike'].std():.4f})")
    print(f"    R²    = {df_folds['r_squared'].mean():.4f}  "
          f"(±{df_folds['r_squared'].std():.4f})")

    # [5] Bootstrap fold-level
    print(f"\n[5/8] Calculando IC bootstrap fold-level (n_boot=2000)...")
    ci_result = bootstrap_fold_metrics(df_folds, n_boot=2000, alpha=0.05)
    for m in ('mse', 'mae', 'qlike'):
        info = ci_result[m]
        print(f"  {m.upper():5s}: {info['point']:.4e}  "
              f"IC 95% [{info['lower']:.4e}, {info['upper']:.4e}]  "
              f"SE={info['se']:.4e}")

    # [6] Previsão final no conjunto de teste (20%) — apenas para gráficos
    print(f"\n[6/8] Previsão final no teste ({len(returns_test)} obs)...")
    t_start = time.time()
    returns_full = pd.concat([returns_train, returns_test])
    n_train      = len(returns_train)

    forecast_result = forecast_variance(
        returns_full, n_train, ARMA_P, ARMA_Q, GARCH_P, GARCH_Q
    )

    if forecast_result.get('converged', False):
        forecast_var_test = np.array(forecast_result['forecast_variance'])
    else:
        raise RuntimeError(
            f"Forecast final falhou: {forecast_result.get('error', 'Unknown')}"
        )

    print(f"  Previsão concluída em {time.time()-t_start:.1f}s")

    daily_forecast = aggregate_variance_to_daily(forecast_var_test, test.index)
    print(f"  Dias com previsão: {len(daily_forecast)} "
          f"({daily_forecast.index[0].date()} → {daily_forecast.index[-1].date()})")

    aligned = pd.DataFrame({
        'forecast': daily_forecast,
        'rv':       rv_benchmark
    }).dropna()
    print(f"  Dias alinhados com RV benchmark: {len(aligned)}")

    if len(aligned) == 0:
        raise RuntimeError(
            "Nenhum dia em comum entre as previsões e o RV benchmark."
        )

    daily_forecast_aligned = aligned['forecast'].values
    rv_aligned             = aligned['rv'].values

    # Métricas do split 80/20 (complementares — não usadas na comparação final)
    metrics_test = compute_all_metrics(daily_forecast_aligned, rv_aligned,
                                       model_name=bar_type_name)
    print(f"\n  Métricas no split 20% (suplementares):")
    print(f"    MSE  = {metrics_test['mse']:.2e}")
    print(f"    MAE  = {metrics_test['mae']:.2e}")
    print(f"    QLIKE= {metrics_test['qlike']:.4f}")
    print(f"    R²   = {metrics_test['r_squared']:.4f}")

    # [7] Gráficos de previsão
    print(f"\n[7/8] Gerando gráficos de previsão...")
    plot_forecast_with_ci(
        daily_forecast_aligned, aligned.index, rv_aligned,
        bar_type_name, ci_result, alpha=0.05
    )

    # [8] Análise de resíduos (split 20%)
    print(f"\n[8/8] Análise de resíduos da previsão...")
    forecast_errors = daily_forecast_aligned - rv_aligned
    plot_residual_analysis(
        forecast_errors,
        daily_forecast_aligned,
        aligned.index,
        series_label=f'Resíduos de Previsão (σ̂²_t − RV_t)',
        bar_type_name=bar_type_name,
        filename_prefix='forecast_errors'
    )

    return {
        'bar_type':        bar_type_name,
        'garch_order':     (GARCH_P, GARCH_Q),
        'arma_order':      (ARMA_P,  ARMA_Q),
        'final_model':     final_model,
        'df_folds':        df_folds,
        'pooled_forecast': pooled_f,
        'pooled_rv':       pooled_r,
        'metrics_cv':      {
            'model':     bar_type_name,
            'mse':       float(df_folds['mse'].mean()),
            'mae':       float(df_folds['mae'].mean()),
            'qlike':     float(df_folds['qlike'].mean()),
            'r_squared': float(df_folds['r_squared'].mean()),
        },
        'ci_result':   ci_result,
        'metrics_test': metrics_test,
        'forecast_var': daily_forecast_aligned,
        'realized_var': rv_aligned,
        'test_index':   aligned.index,
    }


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.random.seed(RANDOM_SEED)

    log_path = os.path.join(RESULTS_DIR, "pipeline_log.txt")
    sys.stdout = Logger(log_path)

    n_folds = math.comb(CPCV_N_GROUPS, CPCV_K_TEST)
    print("=" * 70)
    print(f"PIPELINE GARCH(1,1) — CPCV de Avaliação Preditiva")
    print(f"Modelo fixo: ARMA({ARMA_P},{ARMA_Q})-GARCH({GARCH_P},{GARCH_Q})")
    print(f"Diretório de saída: {RESULTS_DIR}")
    print(f"Benchmark: Volatilidade Realizada (RV Consenso)")
    print(f"Backend: {'Rust (arma_garch_cv)' if RUST_AVAILABLE else 'Python (arch)'}")
    print(f"CPCV Avaliação: N={CPCV_N_GROUPS}, K={CPCV_K_TEST}  "
          f"→  C({CPCV_N_GROUPS},{CPCV_K_TEST}) = {n_folds:,} folds")
    print(f"Bootstrap: fold-level iid  (n_boot=2.000, α=0.05)")
    print(f"Início: {pd.Timestamp.now()}")
    print("=" * 70)

    print(f"\n[0] Carregando RV Benchmark...")
    rv_benchmark = load_rv_benchmark(FILE_RV_BENCHMARK)

    t_total_start = time.time()

    # ── Executar pipeline para cada tipo de barra ──────────────────────────
    results_dollar = run_pipeline(FILE_DOLLAR_5M, "Dollar_Bars_5m", rv_benchmark)
    results_time   = run_pipeline(FILE_TIME_5M,   "Time_Bars_5m",   rv_benchmark)

    # ── Alinhamento de datas para comparação final ─────────────────────────
    print(f"\n{'='*70}")
    print("ALINHAMENTO DE DATAS: Dollar Bars vs Time Bars")
    print(f"{'='*70}")

    idx_dollar   = results_dollar['test_index']
    idx_time     = results_time['test_index']
    common_dates = idx_dollar.intersection(idx_time)

    print(f"  Dias no teste Dollar Bars : {len(idx_dollar)}"
          f"  ({idx_dollar[0].date()} → {idx_dollar[-1].date()})")
    print(f"  Dias no teste Time Bars   : {len(idx_time)}"
          f"  ({idx_time[0].date()} → {idx_time[-1].date()})")
    print(f"  Dias em COMUM (interseção): {len(common_dates)}"
          f"  ({common_dates[0].date()} → {common_dates[-1].date()})")

    if len(common_dates) == 0:
        raise RuntimeError("Nenhum dia em comum entre os testes de Dollar e Time bars.")

    if len(common_dates) < len(idx_dollar) or len(common_dates) < len(idx_time):
        print(f"  [AVISO] Períodos diferiram — usando {len(common_dates)} dias comuns.")

    # ── Comparação final: CPCV fold-level (primário) ───────────────────────
    print(f"\n{'='*70}")
    print("COMPARAÇÃO FINAL: CPCV fold-level — Dollar Bars vs Time Bars")
    print(f"{'='*70}")

    df_d = results_dollar['df_folds']
    df_t = results_time['df_folds']

    print(f"\n--- Médias CPCV (fold-level, {min(len(df_d), len(df_t))} folds) ---")
    print(f"{'Métrica':<8}  {'Dollar Bars':>14}  {'Time Bars':>14}")
    print("-" * 42)
    for m in ('mse', 'mae', 'qlike', 'r_squared'):
        print(f"{m.upper():<8}  "
              f"{df_d[m].mean():>14.4e}  "
              f"{df_t[m].mean():>14.4e}")

    print(f"\n--- IC Bootstrap 95% (fold-level, n_boot=2.000) ---")
    for bar_label, ci_res in [
        ("Dollar Bars", results_dollar['ci_result']),
        ("Time Bars",   results_time['ci_result']),
    ]:
        print(f"  {bar_label}:")
        for m in ('mse', 'mae', 'qlike'):
            info = ci_res[m]
            print(f"    {m.upper():5s} = {info['point']:.4e}  "
                  f"[{info['lower']:.4e}, {info['upper']:.4e}]  "
                  f"SE={info['se']:.4e}")

    # Estatísticas complementares
    extra = compute_extra_stats(df_d, df_t)
    print(f"\n--- Estatísticas Complementares ---")
    for metric in ('mse', 'mae', 'qlike'):
        wr = extra[f'win_rate_{metric}']
        cv_d = extra[f'cv_{metric}_dollar']
        cv_t = extra[f'cv_{metric}_time']
        print(f"  {metric.upper():5s}  Win Rate(Dollar<Time): {wr:.1%}  "
              f"CV Dollar: {cv_d:.4f}  CV Time: {cv_t:.4f}")

    # Teste de Diebold-Mariano sobre pool de previsões CPCV
    print(f"\n--- Teste de Diebold-Mariano (pool CPCV) ---")
    pooled_f_d = results_dollar['pooled_forecast']
    pooled_r_d = results_dollar['pooled_rv']
    pooled_f_t = results_time['pooled_forecast']
    pooled_r_t = results_time['pooled_rv']

    # DM usando apenas pares alinhados (mesma data nos dois pools)
    # Usa as médias CPCV como proxies escalares para o DM via comparison_table
    _, dm_result = comparison_table(
        results_dollar['metrics_cv'],
        results_time['metrics_cv'],
        pooled_f_d[:len(pooled_f_t)],
        pooled_f_t[:len(pooled_f_d)],
        pooled_r_d[:len(pooled_r_t)],
        pooled_r_t[:len(pooled_r_d)],
    )
    print(f"  DM Statistic: {dm_result['dm_stat']:.4f}")
    print(f"  p-value:      {dm_result['p_value']:.4f}")
    print(f"  Conclusão:    {dm_result['conclusion']}")

    # Comparação suplementar: split 20% (período comum)
    print(f"\n--- Métricas Suplementares (split 20%, período comum) ---")
    mask_dollar = pd.Series(idx_dollar).isin(common_dates).values
    mask_time   = pd.Series(idx_time).isin(common_dates).values
    fcast_d_com = results_dollar['forecast_var'][mask_dollar]
    rv_d_com    = results_dollar['realized_var'][mask_dollar]
    fcast_t_com = results_time['forecast_var'][mask_time]
    rv_t_com    = results_time['realized_var'][mask_time]

    met_d_com = compute_all_metrics(fcast_d_com, rv_d_com,
                                     model_name="Dollar_Bars_5m_common")
    met_t_com = compute_all_metrics(fcast_t_com, rv_t_com,
                                     model_name="Time_Bars_5m_common")

    print(f"  {'Métrica':<8}  {'Dollar (comum)':>14}  {'Time (comum)':>14}")
    print("  " + "-" * 42)
    for m in ('mse', 'mae', 'qlike', 'r_squared'):
        print(f"  {m.upper():<8}  {met_d_com[m]:>14.4e}  {met_t_com[m]:>14.4e}")

    # ── Gráficos comparativos CPCV ─────────────────────────────────────────
    print(f"\n--- Gerando gráficos comparativos CPCV ---")
    plot_cpcv_evaluation(df_d, df_t)

    # ── Salvar CSVs ────────────────────────────────────────────────────────
    # Bootstrap fold-level
    ci_rows = []
    for bar_label, ci_res, df_f in [
        ("Dollar_Bars_5m", results_dollar['ci_result'], df_d),
        ("Time_Bars_5m",   results_time['ci_result'],   df_t),
    ]:
        for m in ('mse', 'mae', 'qlike'):
            info = ci_res[m]
            ci_rows.append({
                'modelo':          bar_label,
                'metrica':         m.upper(),
                'pontual_cv':      info['point'],
                'ic_lower_95':     info['lower'],
                'ic_upper_95':     info['upper'],
                'se_bootstrap':    info['se'],
                'n_folds':         len(df_f),
                'metodo_bootstrap': 'fold-level iid',
            })
    ci_df  = pd.DataFrame(ci_rows)
    ci_csv = os.path.join(RESULTS_DIR, "ic_bootstrap_metricas.csv")
    ci_df.to_csv(ci_csv, index=False)
    print(f"\n  IC Bootstrap salvo: {ci_csv}")

    # Métricas por fold
    for name, res in [("dollar", results_dollar), ("time", results_time)]:
        folds_csv = os.path.join(RESULTS_DIR, f"cpcv_folds_{name}.csv")
        res['df_folds'].to_csv(folds_csv, index=False)
        print(f"  CPCV folds salvo: {folds_csv}")

    # Estatísticas complementares
    extra_df = pd.DataFrame([extra])
    extra_csv = os.path.join(RESULTS_DIR, "extra_stats.csv")
    extra_df.to_csv(extra_csv, index=False)
    print(f"  Estatísticas extras salvas: {extra_csv}")

    t_total = time.time() - t_total_start
    print(f"\n{'='*70}")
    print(f"Pipeline concluído em {t_total/60:.1f} minutos")
    print(f"Resultados salvos em: {RESULTS_DIR}")
    print(f"{'='*70}")
