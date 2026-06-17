"""
yang_zhang.py — Estimador de Variância Realizada Yang-Zhang em janela rolante.

Implementa o estimador Yang-Zhang (2000) adaptado para barras intraday.
O estimador combina três componentes:
  1. Variância overnight (close_{t-1} → open_t)
  2. Variância open-to-close (open_t → close_t)
  3. Estimador Rogers-Satchell (usa OHLC completo)

Referência: Yang, D., & Zhang, Q. (2000). "Drift-Independent Volatility Estimation
Based on High, Low, Open, and Close Prices." Journal of Business, 73(3), 477-492.
"""

import numpy as np
import pandas as pd


def yang_zhang_variance(df, window=288, k=0.34):
    """
    Calcula a variância realizada Yang-Zhang em janela rolante.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com colunas 'open', 'high', 'low', 'close'.
        O índice deve ser temporal e estar ordenado.
    window : int
        Tamanho da janela rolante em número de barras.
    k : float
        Peso de ponderação entre open-to-close e Rogers-Satchell.
        O valor ótimo (que minimiza a variância do estimador) é 0.34.

    Returns
    -------
    pd.Series
        Série temporal da variância Yang-Zhang (por barra, não anualizada).
    """
    required_cols = ['open', 'high', 'low', 'close']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Coluna '{col}' não encontrada no DataFrame. "
                             f"Colunas disponíveis: {df.columns.tolist()}")

    # 1. Componente Overnight: log(open_t / close_{t-1})
    log_overnight = np.log(df['open'] / df['close'].shift(1))

    # 2. Componente Open-to-Close: log(close_t / open_t)
    log_open_close = np.log(df['close'] / df['open'])

    # 3. Componente Rogers-Satchell:
    #    RS_t = log(H_t/C_t) * log(H_t/O_t) + log(L_t/C_t) * log(L_t/O_t)
    log_hc = np.log(df['high'] / df['close'])
    log_ho = np.log(df['high'] / df['open'])
    log_lc = np.log(df['low'] / df['close'])
    log_lo = np.log(df['low'] / df['open'])
    rs = log_hc * log_ho + log_lc * log_lo

    # Variâncias rolantes de cada componente
    var_overnight = log_overnight.rolling(window=window, min_periods=window).var()
    var_open_close = log_open_close.rolling(window=window, min_periods=window).var()
    var_rs = rs.rolling(window=window, min_periods=window).mean()

    # Fórmula Yang-Zhang:
    # σ²_YZ = σ²_overnight + k * σ²_open-to-close + (1 - k) * σ²_RS
    yz_variance = var_overnight + k * var_open_close + (1 - k) * var_rs

    yz_variance.name = 'yz_variance'
    return yz_variance


def compute_yz_for_bars(df, window=288, k=0.34):
    """
    Wrapper que carrega, valida e calcula o Yang-Zhang para um DataFrame de barras.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com OHLC e timestamp.
    window : int
        Janela rolante.
    k : float
        Peso do estimador.

    Returns
    -------
    pd.DataFrame
        DataFrame original acrescido das colunas 'log_return' e 'yz_variance'.
    """
    df = df.copy()

    # Tratar timestamp se necessário
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df.sort_index(inplace=True)

    # Calcular retornos logarítmicos
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))

    # Calcular variância Yang-Zhang
    df['yz_variance'] = yang_zhang_variance(df, window=window, k=k)

    # Remover NaNs iniciais (gerados pelo shift e pela janela rolante)
    df.dropna(subset=['log_return', 'yz_variance'], inplace=True)

    return df
