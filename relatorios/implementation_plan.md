# Modelagem ARIMA-GARCH: Dollar Bars vs Time Bars (5 min)

Pipeline completo de modelagem e comparação de previsão de variância entre barras de dólar e barras de tempo, utilizando validação CPCV e proxy de variância realizada Yang-Zhang.

## User Review Required

> [!IMPORTANT]
> **Dependência `arch`**: A biblioteca `arch` (para modelos GARCH) **não está instalada** no ambiente. Precisarei instalá-la via `pip install arch`. Confirma?

> [!IMPORTANT]
> **Janela do Yang-Zhang**: Como os dados são barras de 5 minutos e não barras diárias, o conceito de "overnight" (close-to-open) no Yang-Zhang clássico se traduz em "close da barra anterior → open da barra atual". A janela rolante padrão que proponho é **288 barras** (equivalente a 24h de dados em barras de 5min). Concorda com essa janela ou prefere outro valor?

> [!IMPORTANT]
> **Split Treino/Teste**: Proponho um split temporal de **80% treino / 20% teste** (aproximadamente Jan/2024–Jul/2025 para treino e Jul/2025–Dez/2025 para teste). A CPCV será aplicada **apenas dentro do treino** para selecionar os melhores parâmetros ARIMA(p,d,q)-GARCH(P,Q). O teste final permanece fixo e intocado. Concorda?

## Open Questions

> [!NOTE]
> **Embargo no CPCV**: Para modelos GARCH que possuem memória na volatilidade condicional, é necessário definir um período de "embargo" entre treino e teste nos folds do CPCV para evitar leakage. Proponho um embargo de **288 barras** (24h), compatível com a janela do Yang-Zhang. Isso é conservador o suficiente?

## Arquitetura do Pipeline

O pipeline será dividido em **4 módulos** independentes dentro de `c:\Codigos\TCC\modelagem\`:

```
modelagem/
├── config.py              # Constantes e hiperparâmetros centralizados
├── yang_zhang.py           # Cálculo da proxy de variância realizada
├── cpcv.py                 # Implementação do CPCV (Combinatorial Purged Cross-Validation)
├── modelo_arima_garch.py   # Fitting ARIMA-GARCH + seleção AIC/BIC
├── metricas.py             # Métricas de avaliação de previsão de variância
├── main.py                 # Orquestrador principal do pipeline
└── resultados/             # Pasta para outputs (gráficos, tabelas, logs)
```

---

## Proposed Changes

### 1. Configuração Central

#### [NEW] [config.py](file:///c:/Codigos/TCC/modelagem/config.py)

Arquivo de configuração centralizado com todos os hiperparâmetros do pipeline:

| Parâmetro | Valor | Justificativa |
|---|---|---|
| `YZ_WINDOW` | 288 | 24h em barras de 5min |
| `TRAIN_RATIO` | 0.80 | 80% treino, 20% teste |
| `CPCV_N_GROUPS` | 6 | Número de grupos na divisão CPCV |
| `CPCV_K_TEST` | 2 | Número de grupos usados como teste em cada combinação |
| `EMBARGO_BARS` | 288 | 24h de embargo entre folds |
| `ARIMA_P_MAX` | 5 | Máximo de lags AR |
| `ARIMA_D_MAX` | 1 | Máximo de diferenciação |
| `ARIMA_Q_MAX` | 5 | Máximo de lags MA |
| `GARCH_P_MAX` | 2 | Máximo de lags GARCH |
| `GARCH_Q_MAX` | 2 | Máximo de lags ARCH |

---

### 2. Proxy de Variância Realizada (Yang-Zhang)

#### [NEW] [yang_zhang.py](file:///c:/Codigos/TCC/modelagem/yang_zhang.py)

Implementação do estimador Yang-Zhang adaptado para barras intraday:

**Fórmula:**
$$\sigma^2_{YZ} = \sigma^2_{overnight} + k \cdot \sigma^2_{open-to-close} + (1-k) \cdot \sigma^2_{RS}$$

Onde:
- $\sigma^2_{overnight}$: variância de `log(open_t / close_{t-1})` em janela rolante
- $\sigma^2_{open-to-close}$: variância de `log(close_t / open_t)` em janela rolante
- $\sigma^2_{RS}$: média rolante do estimador Rogers-Satchell: `log(H/C)*log(H/O) + log(L/C)*log(L/O)`
- $k = 0.34$ (valor ótimo da literatura para minimizar variância do estimador)

**Saída**: Série temporal de variância realizada Yang-Zhang para cada tipo de barra.

---

### 3. CPCV (Combinatorial Purged Cross-Validation)

#### [NEW] [cpcv.py](file:///c:/Codigos/TCC/modelagem/cpcv.py)

Implementação manual do CPCV seguindo Marcos López de Prado (Advances in Financial Machine Learning):

**Fluxo**:
1. Dividir os dados de treino em $N=6$ grupos ordenados temporalmente
2. Para cada combinação $\binom{N}{K} = \binom{6}{2} = 15$ combinações:
   - $K=2$ grupos são usados como teste
   - Os demais $N-K=4$ grupos são usados como treino
   - **Purging**: Remover do treino as observações cuja janela de cálculo (e.g., lags do ARIMA/GARCH) se sobrepõe ao início do teste
   - **Embargo**: Excluir um buffer de 288 barras após cada bloco de teste para eliminar leakage por autocorrelação serial

**Para cada fold CPCV:**
- Ajustar ARIMA(p,d,q)-GARCH(P,Q) com grid search sobre os parâmetros
- Registrar AIC e BIC de cada combinação de parâmetros
- Registrar métricas de previsão de variância (MSE, QLIKE) contra a proxy Yang-Zhang no fold de teste

**Seleção final de parâmetros:**
- Selecionar a combinação (p,d,q,P,Q) com **menor AIC/BIC médio** e **menor desvio-padrão** entre os 15 folds (busca de estabilidade, não apenas performance pontual)

---

### 4. Modelo ARIMA-GARCH

#### [NEW] [modelo_arima_garch.py](file:///c:/Codigos/TCC/modelagem/modelo_arima_garch.py)

Usando a biblioteca `arch`:

1. **ARIMA para a média condicional**: Filtrar os retornos com `statsmodels.tsa.arima.model.ARIMA` para obter os resíduos
2. **GARCH para a variância condicional**: Ajustar `arch.arch_model` nos resíduos do ARIMA
3. **Previsão out-of-sample**: Gerar previsões 1-step-ahead da variância condicional no período de teste
4. **Information Criteria**: Extrair AIC/BIC do modelo conjunto para ranking

---

### 5. Métricas de Comparação

#### [NEW] [metricas.py](file:///c:/Codigos/TCC/modelagem/metricas.py)

Métricas para avaliar o poder preditivo da variância, seguindo Patton (2011) para robustez:

| Métrica | Fórmula | Propósito |
|---|---|---|
| **MSE** | $\frac{1}{T}\sum(\hat{\sigma}^2_t - RV_t)^2$ | Erro quadrático padrão |
| **MAE** | $\frac{1}{T}\sum\|\hat{\sigma}^2_t - RV_t\|$ | Erro absoluto |
| **QLIKE** | $\frac{1}{T}\sum\left(\log(\hat{\sigma}^2_t) + \frac{RV_t}{\hat{\sigma}^2_t}\right)$ | Robusta a ruído na proxy (Patton, 2011) |
| **R²** | Regressão Mincer-Zarnowitz | Eficiência do forecast |
| **MZ α,β** | $RV_t = \alpha + \beta\hat{\sigma}^2_t + \varepsilon_t$ | Teste de viés ($H_0: \alpha=0, \beta=1$) |
| **Diebold-Mariano** | Teste de igualdade de poder preditivo | Comparação estatística entre Dollar vs Time bars |

O **Diebold-Mariano test** é particularmente crucial aqui: ele testa formalmente se a diferença na acurácia preditiva entre os modelos (dollar bars vs time bars) é estatisticamente significante.

---

### 6. Orquestrador Principal

#### [NEW] [main.py](file:///c:/Codigos/TCC/modelagem/main.py)

Fluxo de execução:

```
1. Carregar dados (dollar_bars_5m, time_bars_5m)
2. Calcular retornos logarítmicos
3. Calcular proxy Yang-Zhang para ambos
4. Split temporal: 80% treino / 20% teste
5. CPCV no treino:
   - Grid search ARIMA(p,d,q) × GARCH(P,Q)
   - Selecionar melhor modelo por AIC/BIC médio
6. Retreinar modelo final com melhores parâmetros em TODO o treino
7. Previsão 1-step-ahead no teste
8. Calcular métricas (MSE, MAE, QLIKE, MZ, Diebold-Mariano)
9. Gerar tabelas comparativas e gráficos
10. Salvar resultados em modelagem/resultados/
```

---

## Verification Plan

### Execução
- Executar `main.py` e verificar se o pipeline completo roda sem erros para ambos os tipos de barras.
- Verificar se os gráficos de variância prevista vs realizada são coerentes.

### Validação Estatística
- Confirmar que o Mincer-Zarnowitz retorna coeficientes sensatos (β próximo de 1).
- Verificar que o Diebold-Mariano tem direção consistente com as métricas MSE/QLIKE.
- Assegurar que a CPCV gera 15 folds e que os índices de purging/embargo estão corretos (sem sobreposição treino-teste).
