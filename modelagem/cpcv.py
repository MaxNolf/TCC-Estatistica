"""
cpcv.py — Combinatorial Purged Cross-Validation (CPCV).

Implementação do CPCV conforme Marcos López de Prado,
"Advances in Financial Machine Learning" (2018), Cap. 12.

O CPCV resolve dois problemas do K-Fold clássico em séries financeiras:
  1. Leakage temporal: dados futuros vazam para o treino
  2. Overfitting: um único backtest path não é robusto

A solução é gerar C(N,K) combinações de folds temporais, cada uma com
purging (remoção de sobreposição) e embargo (buffer de segurança).

OTIMIZAÇÃO (v2):
  - split() não usa set(range(n_samples)) por fold (era O(n×C(N,K))).
    Em vez disso, trabalha com intervalos de índices e usa np.concatenate
    O(N×K) por fold, muito mais eficiente para N=30, K=5.
  - get_summary() calcula tamanhos analiticamente em O(C(N,K)) sem alocar
    arrays de índices completos.
"""

import math
import numpy as np
from itertools import combinations

# tqdm com fallback silencioso se não instalado
try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


def _tqdm_wrap(iterable, **kwargs):
    """Retorna tqdm se disponível, senão o iterável sem decoração."""
    if _HAS_TQDM:
        return _tqdm(iterable, **kwargs)
    return iterable


class CombPurgedKFold:
    """
    Combinatorial Purged K-Fold Cross-Validation.

    Parameters
    ----------
    n_groups : int
        Número total de grupos (N) em que a série é dividida.
    k_test : int
        Número de grupos usados como teste (K) em cada combinação.
    embargo_bars : int
        Número de barras de embargo após cada bloco de teste.
        Remove do treino as barras imediatamente após o teste
        para evitar leakage por autocorrelação serial.

    Attributes
    ----------
    n_splits : int
        Número total de combinações = C(N, K).
    """

    def __init__(self, n_groups=6, k_test=2, embargo_bars=288):
        if k_test >= n_groups:
            raise ValueError(f"k_test ({k_test}) deve ser menor que n_groups ({n_groups})")
        self.n_groups = n_groups
        self.k_test = k_test
        self.embargo_bars = embargo_bars
        # Cálculo O(1) — sem materializar a lista de combinações
        self.n_splits = math.comb(n_groups, k_test)

    # ------------------------------------------------------------------
    # Helpers de intervalo — evitam set(range(n_samples)) por fold
    # ------------------------------------------------------------------
    @staticmethod
    def _intervals_to_array(intervals, n_samples):
        """
        Converte lista de (start, end+1) em np.ndarray de índices,
        excluindo os intervalos `exclude` (também lista de (start, end+1)).

        Complexidade: O(soma dos tamanhos dos intervalos) em vez de O(n_samples).
        """
        parts = [np.arange(s, e, dtype=np.int64) for s, e in intervals if s < e]
        if not parts:
            return np.empty(0, dtype=np.int64)
        arr = np.concatenate(parts)
        return arr

    def _build_group_ranges(self, n_samples):
        """Devolve lista de (start, end_inclusive) por grupo — calculado uma vez.

        Otimizado: calcula limites aritmeticamente sem alocar np.arange(n_samples).
        """
        base_size, remainder = divmod(n_samples, self.n_groups)
        ranges = []
        cursor = 0
        for i in range(self.n_groups):
            size = base_size + (1 if i < remainder else 0)
            if size > 0:
                ranges.append((cursor, cursor + size - 1))  # (start, end_inclusive)
                cursor += size
        return ranges

    def _compute_fold_intervals(self, test_groups, group_ranges, n_samples):
        """
        Calcula os intervalos de treino/teste para uma combinação de grupos,
        incluindo purging e embargo.

        Returns
        -------
        train_ivals : list of (int, int)
        test_ivals  : list of (int, int)
        fold_info   : dict
        """
        E = self.embargo_bars
        test_ivals  = []
        emb_ivals   = []
        purge_ivals = []

        for tg in test_groups:
            s, e = group_ranges[tg]
            test_ivals.append((s, e + 1))

            # Embargo: [e+1, e+E]
            emb_s = e + 1
            emb_e = min(e + E, n_samples)
            if emb_s < emb_e:
                emb_ivals.append((emb_s, emb_e))

            # Purge: [s-E, s)
            pur_s = max(s - E, 0)
            pur_e = s
            if pur_s < pur_e:
                purge_ivals.append((pur_s, pur_e))

        # ── Conjunto de treino = [0, n_samples) minus test+emb+purge ──
        excluded = sorted(test_ivals + emb_ivals + purge_ivals)

        # Fundir intervalos sobrepostos
        merged = []
        for iv in excluded:
            if merged and iv[0] <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], iv[1]))
            else:
                merged.append(list(iv))

        # Complemento de `merged` dentro de [0, n_samples)
        train_ivals = []
        cursor = 0
        for ms, me in merged:
            if cursor < ms:
                train_ivals.append((cursor, ms))
            cursor = me
        if cursor < n_samples:
            train_ivals.append((cursor, n_samples))

        n_train     = sum(e - s for s, e in train_ivals)
        n_test      = sum(e - s for s, e in test_ivals)
        n_purged    = sum(e - s for s, e in purge_ivals)
        n_embargoed = sum(e - s for s, e in emb_ivals)

        fold_info = {
            'test_groups' : test_groups,
            'n_train'     : n_train,
            'n_test'      : n_test,
            'n_purged'    : n_purged,
            'n_embargoed' : n_embargoed,
        }

        return train_ivals, test_ivals, fold_info

    def split_intervals(self, n_samples, progress=True):
        """
        Gera os intervalos de treino e teste para cada fold CPCV
        SEM materializar arrays numpy — retorna listas de tuplas (start, end).

        Cada fold retorna ~5-10 tuplas (~160 bytes) em vez de arrays
        de ~137K int64 (~1 MB). Reduz a memória de ~180 GB para ~KBs.

        Parameters
        ----------
        n_samples : int
            Número total de observações na série de treino.
        progress : bool
            Se True e tqdm disponível, exibe barra de progresso.

        Yields
        ------
        train_intervals : list of (int, int)
            Intervalos [start, end) para treino.
        test_intervals  : list of (int, int)
            Intervalos [start, end) para teste.
        fold_info       : dict
            Metadados do fold (test_groups, n_train, n_test, etc.).
        """
        group_ranges = self._build_group_ranges(n_samples)

        combos = combinations(range(self.n_groups), self.k_test)
        if progress:
            combos = _tqdm_wrap(
                combos,
                total=self.n_splits,
                desc="  CPCV folds (intervalos)",
                unit="fold",
                ncols=90,
                leave=False,
            )

        for test_groups in combos:
            yield self._compute_fold_intervals(test_groups, group_ranges, n_samples)

    def split(self, n_samples, progress=True):
        """
        Gera os índices de treino e teste para cada fold CPCV.

        NOTA: Para 142K+ folds, prefira split_intervals() que NÃO
        materializa arrays numpy e consome ordens de grandeza menos memória.

        Parameters
        ----------
        n_samples : int
            Número total de observações na série de treino.
        progress : bool
            Se True e tqdm disponível, exibe barra de progresso.

        Yields
        ------
        train_indices : np.ndarray
        test_indices  : np.ndarray
        fold_info     : dict
        """
        for train_ivals, test_ivals, fold_info in self.split_intervals(
            n_samples, progress=progress
        ):
            train_indices = self._intervals_to_array(train_ivals, n_samples)
            test_indices  = self._intervals_to_array(test_ivals,  n_samples)
            yield train_indices, test_indices, fold_info

    def get_summary(self, n_samples):
        """
        Retorna um resumo analítico da configuração CPCV — O(1), sem iterar folds.

        Parameters
        ----------
        n_samples : int

        Returns
        -------
        dict
        """
        group_size    = n_samples // self.n_groups
        k             = self.k_test
        n             = self.n_groups
        E             = self.embargo_bars

        avg_test_size     = k * group_size
        # Embargo: até E barras por bloco de teste (pode tocar borda)
        avg_embargoed     = min(k * E, n_samples - avg_test_size)
        avg_purged        = min(k * E, avg_test_size)   # estimativa conservadora
        avg_train_size    = n_samples - avg_test_size - avg_embargoed - avg_purged

        return {
            'n_splits'      : self.n_splits,
            'n_groups'      : self.n_groups,
            'k_test'        : self.k_test,
            'embargo_bars'  : self.embargo_bars,
            'avg_train_size': max(avg_train_size, 0),
            'avg_test_size' : avg_test_size,
            'avg_purged'    : avg_purged,
            'avg_embargoed' : avg_embargoed,
            'total_samples' : n_samples,
        }
