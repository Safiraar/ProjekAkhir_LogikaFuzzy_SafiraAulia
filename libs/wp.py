# libs/wp.py
import pandas as pd
import numpy as np


def wp_full_process(df: pd.DataFrame, criteria_meta: dict, weights: dict):
    """
    Weighted Product (WP) steps:
      1. Cost → bobot menjadi negatif (exponent)
      2. Hitung S_i = ∏ (x_ij ^ w_j)
      3. Hitung V_i = S_i / Σ S_i
    Revisi:
      • Index diubah menjadi A1–A5 dst.
      • Pada result: hanya tampilkan V dan rank (S disembunyikan).
    """

    crits = list(criteria_meta.keys())

    # ------------------------------------------------------------
    # 1. Ubah index menjadi A1–A5
    # ------------------------------------------------------------
    new_index = [f"A{i+1}" for i in range(len(df))]
    df = df.copy()
    df.index = new_index

    # Raw matrix
    X = df[crits].astype(float).copy()

    # Bobot
    W = pd.Series(weights)

    # ------------------------------------------------------------
    # 2. Exponents (cost → negatif)
    # ------------------------------------------------------------
    exp = W.copy()
    for c in crits:
        if criteria_meta[c]['attr'] == 'cost':
            exp[c] = -abs(W[c])  # cost → pangkat negatif

    # ------------------------------------------------------------
    # 3. Hitung S_i
    # ------------------------------------------------------------
    # Tambahkan epsilon agar tidak terjadi log(0)
    eps = 1e-12
    S = X.clip(lower=eps).pow(exp).prod(axis=1)

    # ------------------------------------------------------------
    # 4. Hitung V_i
    # ------------------------------------------------------------
    V = S / S.sum()

    # ------------------------------------------------------------
    # 5. Hasil akhir (HANYA V dan rank)
    # ------------------------------------------------------------
    result = pd.DataFrame({
        "V": V.round(6),
        "rank": V.rank(ascending=False, method="min").astype(int)
    })
    result.index = new_index

    return {
        "raw_matrix": X,
        "exponents": exp,
        "S": S.round(12),       # tetap dikembalikan tapi tidak ditampilkan di result
        "V": V.round(6),
        "result": result,       # hanya V & rank
        "weights": W
    }
