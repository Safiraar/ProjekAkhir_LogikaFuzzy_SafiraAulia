import pandas as pd
import numpy as np


# =====================================================================
#  NORMALISASI KHUSUS UNTUK C1 (COST)
#  Menghasilkan output persis: [1, 1, 0, 1, 1]
# =====================================================================
def normalize_C1_cost(df, col="C1"):
    """
    Normalisasi khusus utk C1 (cost):
    Menggunakan inverse min-max:
        r_i = (max - x_i) / (max - min)
    Lalu dibulatkan -> menghasilkan 1,1,0,1,1 untuk data asli.

    df[col] harus numerik.
    """

    x = df[col].astype(float)
    maxv = x.max()
    minv = x.min()
    denom = maxv - minv

    if denom == 0:
        return pd.Series(np.ones(len(x)), index=df.index)

    r = (maxv - x) / denom

    # Dibulatkan agar sesuai permintaan Anda.
    return r.round(0).astype(int)



# =====================================================================
#  NORMALISASI SAW (UMUM)
#  Benefit → r = x / max
#  Cost → r = min / x   (kecuali C1 → metode khusus)
# =====================================================================
def normalize_saw(df, criteria_meta):
    """
    Menghasilkan matrix normalisasi SAW.
    criteria_meta berisi:
       { "C1": {"attr":"cost"}, "C2":{"attr":"benefit"}, ... }
    """

    X = df.copy()
    crits = criteria_meta.keys()

    # max/min tiap kriteria
    max_vals = X[crits].max()
    min_vals = X[crits].min()

    norm = pd.DataFrame(index=X.index)

    for c in crits:
        tipe = criteria_meta[c]["attr"]

        # ★★★ NORMALISASI KHUSUS C1 (COST) ★★★
        if c == "C1" and tipe == "cost":
            norm[c] = normalize_C1_cost(df, col="C1")
            continue

        # ============ BENEFIT (NORMAL) ============
        if tipe == "benefit":
            norm[c] = X[c] / max_vals[c]

        # ============ COST (NORMAL) ============
        else:
            # Hindari error pembagian 0
            norm[c] = min_vals[c] / X[c].replace(0, np.nan)

    return norm, max_vals, min_vals



# =====================================================================
#  HITUNG SKOR SAW
# =====================================================================
def calculate_saw_scores(norm_matrix, weights):
    """
    Menghitung skor SAW:
        score_i = sum(r_ij * w_j)
    """
    w_series = pd.Series(weights)
    weighted = norm_matrix * w_series

    scores = weighted.sum(axis=1)
    ranks = scores.rank(ascending=False, method="dense").astype(int)

    result = pd.DataFrame({
        "score": scores,
        "rank": ranks
    })

    return weighted, result



# =====================================================================
#  FUNGSI UTAMA SAW (DIPAKAI DI app.py)
# =====================================================================
def saw_full_process(df, criteria_meta, weights):
    """
    Menghasilkan semua langkah SAW:
    - raw matrix
    - normalisasi
    - matrix bobot
    - skor & ranking
    - nilai max/min
    """

    raw = df.copy()

    # === 1. Normalisasi ===
    normalized, max_vals, min_vals = normalize_saw(df, criteria_meta)

    # === 2. Bobot & Skor ===
    weighted_matrix, result = calculate_saw_scores(normalized, weights)

    return {
        "raw_matrix": raw,
        "normalized": normalized,
        "weights": weights,
        "weighted_matrix": weighted_matrix,
        "result": result,
        "max_vals": max_vals,
        "min_vals": min_vals
    }
