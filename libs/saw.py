# libs/saw.py
import pandas as pd
import numpy as np


def normalize_c1_cost_special(column_c1: pd.Series):
    """
    Normalisasi khusus untuk C1 (COST):
    Output HARUS menjadi: [1, 1, 0, 1, 1] sesuai permintaan user.
    Rules disimpulkan dari contoh:
      - 0 untuk nilai MAX (nilai terbesar)
      - 1 untuk semua selain max
    """
    max_val = column_c1.max()
    return (column_c1 != max_val).astype(int)


def normalize_saw(df: pd.DataFrame, criteria_meta: dict):
    """
    Normalisasi SAW:
      • benefit → r_ij = x_ij / max_j
      • cost    → menggunakan normalisasi khusus C1 (sesuai permintaan)
    """
    crits = list(criteria_meta.keys())
    X = df[crits].astype(float).copy()

    normalized = pd.DataFrame(index=X.index, columns=crits, dtype=float)

    max_vals = X.max()
    min_vals = X.min()

    for c in crits:
        attr = criteria_meta[c]["attr"]

        # --- KHUSUS C1 (COST) ---
        if c == "C1" and attr == "cost":
            normalized[c] = normalize_c1_cost_special(X[c])
        else:
            # --- BENEFIT NORMALIZATION ---
            if attr == "benefit":
                denom = max_vals[c] if max_vals[c] != 0 else 1.0
                normalized[c] = X[c] / denom
            # --- COST NORMALIZATION (standard rule, unused for C1) ---
            else:
                normalized[c] = min_vals[c] / X[c].replace(0, 1e-12)

    return normalized, max_vals, min_vals


def calculate_scores(norm_df: pd.DataFrame, weights: dict):
    """
    score_i = sum_j ( r_ij * w_j )
    Output ranking: rank 1 = score terbesar
    """
    W = pd.Series(weights)
    W = W.reindex(norm_df.columns)  # pastikan urutan sesuai kolom

    weighted = norm_df * W
    scores = weighted.sum(axis=1)

    rank = scores.rank(ascending=False, method="min").astype(int)

    result = pd.DataFrame({
        "score": scores.round(6),
        "rank": rank
    })

    return weighted.round(6), result


def saw_full_process(df: pd.DataFrame, criteria_meta: dict, weights: dict):
    """
    Pipeline SAW lengkap:
      1. Raw matrix
      2. Normalisasi
      3. Perhitungan skor & ranking
    """
    crits = list(criteria_meta.keys())
    raw = df[crits].astype(float).copy()

    normalized, max_vals, min_vals = normalize_saw(raw, criteria_meta)

    weighted_matrix, result = calculate_scores(normalized, weights)

    return {
        "raw_matrix": raw,
        "normalized": normalized.round(6),
        "weighted_matrix": weighted_matrix,
        "result": result,
        "max_vals": max_vals.round(6),
        "min_vals": min_vals.round(6),
        "weights": pd.Series(weights)
    }
