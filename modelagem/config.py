"""
config.py — Configuração centralizada do pipeline GARCH(1,1) com CPCV de Avaliação.

O modelo é fixado em ARMA(0,0)-GARCH(1,1) para ambas as séries.
O CPCV é utilizado para avaliação preditiva out-of-sample, não para seleção
de parâmetros.
"""

import os

# ==============================================================================
# CAMINHOS
# ==============================================================================
BASE_DIR = r"c:\Codigos\TCC"
DATA_DIR = os.path.join(BASE_DIR, "dados_baixados")
MODEL_DIR = os.path.join(BASE_DIR, "modelagem")
RESULTS_DIR = os.path.join(MODEL_DIR, "resultados_s2")

FILE_DOLLAR_5M = os.path.join(DATA_DIR, "BTCUSDT_dollar_bars_5m.parquet")
FILE_TIME_5M = os.path.join(DATA_DIR, "BTCUSDT_time_bars_5m.parquet")

# Benchmark de Volatilidade Realizada (RV consenso diário)
FILE_RV_BENCHMARK = os.path.join(
    BASE_DIR, "Volatilidade_Benchmark_Completo", "rv_benchmark_consenso.parquet"
)


# ==============================================================================
# SPLIT TREINO / TESTE
# ==============================================================================
TRAIN_RATIO = 0.80     # 80% treino, 20% teste

# ==============================================================================
# CPCV — Combinatorial Purged Cross-Validation (para AVALIAÇÃO)
# ==============================================================================
CPCV_N_GROUPS = 10     # Número de grupos na divisão temporal
CPCV_K_TEST = 5        # Número de grupos usados como teste por combinação
EMBARGO_BARS = 240     # Período de embargo (24h) para evitar leakage

# ==============================================================================
# MODELO FIXO: ARMA(0,0)-GARCH(1,1)
# Justificativa: o CPCV de seleção (resultados_s1) elegeu GARCH(1,1) para
# Time Bars (100% dos folds) e GARCH(1,2) para Dollar Bars (92.5% dos folds).
# O GARCH(1,1) é mais parcimonioso e amplamente adotado na literatura;
# sua ordem (p=1, q=1) captura persistência e impacto de choques sem
# sobreajuste. Os retornos log são aproximadamente ruído branco (ADF),
# justificando ARMA(0,0).
# ==============================================================================
ARMA_P  = 0   # sem componente autorregressivo
ARMA_Q  = 0   # sem componente de média móvel
GARCH_P = 1   # lag de persistência da variância condicional
GARCH_Q = 1   # lag de impacto de choques (efeito ARCH)

# ==============================================================================
# RANDOM SEED
# ==============================================================================
RANDOM_SEED = 42

