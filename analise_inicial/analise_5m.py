import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import warnings

# Suprimindo alguns avisos para manter a saída limpa
warnings.filterwarnings("ignore")

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

def load_and_preprocess(filepath):
    print(f"Carregando: {filepath}")
    df = pd.read_parquet(filepath)
    
    # Tratamento do timestamp
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
        
    df.sort_index(inplace=True)
    
    # Identificar a coluna de preço (focando em 'close')
    if 'close' in df.columns:
        price_col = 'close'
    else:
        print("Aviso: coluna 'close' não encontrada. Usando a primeira coluna numérica.")
        price_col = df.select_dtypes(include=[np.number]).columns[0]
        
    # Calcular retornos logarítmicos
    df['log_return'] = np.log(df[price_col] / df[price_col].shift(1))
    
    # Remover NaNs gerados pelo shift
    df.dropna(subset=['log_return'], inplace=True)
    
    return df

def analyze_series(df, series_name, out_dir):
    returns = df['log_return']
    returns_sq = returns ** 2
    
    print(f"\n" + "="*50)
    print(f"ANÁLISE PARA: {series_name}")
    print("="*50)
    
    # 4 Momentos
    mean = returns.mean()
    var = returns.var()
    skew = returns.skew()
    kurt = returns.kurtosis()
    
    print("\n--- Estatísticas Descritivas (1º ao 4º Momento) ---")
    print(f"Média (1º momento):       {mean:.8f}")
    print(f"Variância (2º momento):   {var:.8f}")
    print(f"Assimetria (3º momento):  {skew:.4f}")
    print(f"Curtose (4º momento):     {kurt:.4f} (Excesso de curtose Fisher)")
    
    # Testes de Normalidade
    print("\n--- Testes de Normalidade ---")
    jb_stat, jb_p = stats.jarque_bera(returns)
    print(f"Jarque-Bera:             Estatística = {jb_stat:.4f}, p-valor = {jb_p:.4g}")
    
    ks_stat, ks_p = stats.kstest(returns, 'norm', args=(mean, np.sqrt(var)))
    print(f"Kolmogorov-Smirnov:      Estatística = {ks_stat:.4f}, p-valor = {ks_p:.4g}")
    
    # Testes de Estacionariedade
    print("\n--- Testes de Estacionariedade ---")
    adf_result = adfuller(returns)
    print(f"ADF:                     Estatística = {adf_result[0]:.4f}, p-valor = {adf_result[1]:.4g}")
    
    kpss_result = kpss(returns, regression='c', nlags="auto")
    print(f"KPSS:                    Estatística = {kpss_result[0]:.4f}, p-valor = {kpss_result[1]:.4g}")
    
    # Testes de Heteroscedasticidade e Autocorrelação
    print("\n--- Testes de Autocorrelação e Heteroscedasticidade ---")
    lb_ret = acorr_ljungbox(returns, lags=[10], return_df=True)
    print(f"Ljung-Box (Retornos, lag=10):  Estatística = {lb_ret['lb_stat'].iloc[0]:.4f}, p-valor = {lb_ret['lb_pvalue'].iloc[0]:.4g}")
    
    lb_ret2 = acorr_ljungbox(returns_sq, lags=[10], return_df=True)
    print(f"Ljung-Box (Retornos^2, lag=10): Estatística = {lb_ret2['lb_stat'].iloc[0]:.4f}, p-valor = {lb_ret2['lb_pvalue'].iloc[0]:.4g}")
    
    arch_stat, arch_p, _, _ = het_arch(returns, nlags=10)
    print(f"Engle's ARCH Test (lag=10):    Estatística = {arch_stat:.4f}, p-valor = {arch_p:.4g}")
    
    # ==========================
    # PLOTS
    # ==========================
    
    # 1. Normalidade: Histograma + Curva Normal e QQ-Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.histplot(returns, bins=100, stat='density', alpha=0.6, color='blue', ax=axes[0])
    xmin, xmax = axes[0].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mean, np.sqrt(var))
    axes[0].plot(x, p, 'k', linewidth=2, label='Normal Curve')
    axes[0].set_title(f'Histograma com Curva Normal ({series_name})')
    axes[0].legend()
    
    stats.probplot(returns, dist="norm", plot=axes[1])
    axes[1].set_title(f'QQ-Plot ({series_name})')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{series_name}_normalidade.png'), dpi=300)
    plt.close()
    
    # 2. ACF e PACF dos Retornos
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_acf(returns, lags=40, ax=axes[0], alpha=0.05, title=f'ACF - Retornos ({series_name})')
    plot_pacf(returns, lags=40, ax=axes[1], alpha=0.05, title=f'PACF - Retornos ({series_name})')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{series_name}_acf_pacf_retornos.png'), dpi=300)
    plt.close()
    
    # 3. ACF e PACF dos Retornos ao Quadrado
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_acf(returns_sq, lags=40, ax=axes[0], alpha=0.05, title=f'ACF - Retornos^2 ({series_name})')
    plot_pacf(returns_sq, lags=40, ax=axes[1], alpha=0.05, title=f'PACF - Retornos^2 ({series_name})')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{series_name}_acf_pacf_retornos2.png'), dpi=300)
    plt.close()

if __name__ == "__main__":
    base_dir = r"c:\Codigos\TCC"
    data_dir = os.path.join(base_dir, "dados_baixados")
    out_dir = os.path.join(base_dir, "analise_inicial", "graficos")
    os.makedirs(out_dir, exist_ok=True)
    
    file_dollar = os.path.join(data_dir, "BTCUSDT_dollar_bars_5m.parquet")
    file_time = os.path.join(data_dir, "BTCUSDT_time_bars_5m.parquet")
    
    df_dollar = load_and_preprocess(file_dollar)
    df_time = load_and_preprocess(file_time)
    
    # Redirecionando a saída padrão para salvar os resultados também num arquivo
    import sys
    log_path = os.path.join(base_dir, "analise_inicial", "resultados_estatisticos.txt")
    
    class Logger(object):
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "w", encoding='utf-8')

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            self.terminal.flush()
            self.log.flush()

    sys.stdout = Logger(log_path)
    
    print("Iniciando análises...")
    analyze_series(df_dollar, "Dollar_Bars_5m", out_dir)
    analyze_series(df_time, "Time_Bars_5m", out_dir)
    
    print(f"\nConcluído! Gráficos e resultados salvos na pasta: {os.path.join(base_dir, 'analise_inicial')}")
