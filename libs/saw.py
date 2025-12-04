# libs/saw.py
import pandas as pd
import numpy as np


# ------------------------------------------------------------
# NORMALISASI KHUSUS C1 (COST)
# ------------------------------------------------------------
def normalize_c1_cost_special(column_c1: pd.Series):
    """
    Normalisasi khusus untuk C1 (COST) sesuai permintaan:
      - nilai terbesar → 0
      - nilai lainnya → 1

    Hasil harus:
      [1, 1, 0, 1, 1]
    untuk data contoh Anda.
    """
    max_val = column_c1.max()
    return (column_c1 != max_val).astype(int)


# ------------------------------------------------------------
# NORMALISASI SUATU DATAFRAME
# ------------------------------------------------------------
def normalize_saw(df: pd.DataFrame, criteria_meta: dict):
    """
    Normalisasi SAW:
      • benefit  → r_ij = x_ij / max_j
      • C1 (cost) → menggunakan normalisasi khusus (1/1/0/1/1)
      • cost lain (jika ada) → min / x
    """
    crits = list(criteria_meta.keys())
    X = df[crits].astype(float).copy()

    normalized = pd.DataFrame(index=X.index, columns=crits, dtype=float)

    max_vals = X.max()
    min_vals = X.min()

    for c in crits:
        attr = criteria_meta[c]["attr"]

        # --- KHUSUS C1 ---
        if c == "C1" and attr == "cost":
            normalized[c] = normalize_c1_cost_special(X[c])

        # --- BENEFIT ---
        elif attr == "benefit":
            denom = max_vals[c] if max_vals[c] != 0 else 1
            normalized[c] = X[c] / denom

        # --- COST NORMAL (tidak dipakai untuk C1) ---
        else:
            normalized[c] = min_vals[c] / X[c].replace(0, 1e-12)

    return normalized.round(6), max_vals.round(6), min_vals.round(6)


# ------------------------------------------------------------
# HITUNG SKOR SAW
# ------------------------------------------------------------
def calculate_scores(norm_df: pd.DataFrame, weights: dict):
    """
    score_i = Σ ( r_ij * w_j )
    ranking = rank berdasarkan score (rank 1 = terbesar)
    """
    W = pd.Series(weights)
    W = W.reindex(norm_df.columns)

    weighted = norm_df * W
    scores = weighted.sum(axis=1)

    # rank 1 = score terbesar
    rank = scores.rank(ascending=False, method="min").astype(int)

    result = pd.DataFrame({
        "score": scores.round(6),
        "rank": rank
    })

    return weighted.round(6), result


# ------------------------------------------------------------
# PROSES UTAMA SAW
# ------------------------------------------------------------
def saw_full_process(df: pd.DataFrame, criteria_meta: dict, weights: dict):
    """
    Menghasilkan:
      - raw_matrix
      - normalized
      - weighted_matrix
      - result (score + rank)
      - max_vals, min_vals
      - weights
    Dan index diubah menjadi A1, A2, A3, ...
    """
    crits = list(criteria_meta.keys())

    # --- Buat index A1, A2, A3, ... ---
    new_index = [f"A{i+1}" for i in range(len(df))]
    df = df.copy()
    df.index = new_index

    raw = df[crits].astype(float).copy()

    # --- Normalisasi ---
    normalized, max_vals, min_vals = normalize_saw(raw, criteria_meta)

    # --- Skor dan ranking ---
    weighted_matrix, result = calculate_scores(normalized, weights)

    # Juga set index untuk semua tabel agar A1–A5
    normalized.index = new_index
    weighted_matrix.index = new_index
    result.index = new_index

    return {
        "raw_matrix": raw,
        "normalized": normalized,
        "weighted_matrix": weighted_matrix,
        "result": result,
        "max_vals": max_vals,
        "min_vals": min_vals,
        "weights": pd.Series(weights)
    }
