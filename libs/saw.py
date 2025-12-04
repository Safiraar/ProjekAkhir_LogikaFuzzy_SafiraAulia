# libs/saw.py
import pandas as pd

def normalize_saw(df: pd.DataFrame, criteria_meta: dict):
    """
    Normalisasi SAW:
    - benefit: r_ij = x_ij / max_j
    - cost   : r_ij = min_j / x_ij
    """
    crits = list(criteria_meta.keys())

    # Matriks original
    X = df[crits].astype(float).copy()

    # Tempat hasil normalisasi
    norm = pd.DataFrame(index=df.index, columns=crits, dtype=float)

    # Nilai max/min per kriteria
    max_vals = X.max()
    min_vals = X.min()

    # Loop setiap kriteria
    for c in crits:
        tipe = criteria_meta[c]["attr"]  # "benefit" atau "cost"

        if tipe == "benefit":
            # x_ij / max_j
            denom = max_vals[c] if max_vals[c] != 0 else 1.0
            norm[c] = X[c] / denom

        else:  # COST
            # min_j / x_ij
            denom = X[c].replace(0, 1e-9)  # hindari pembagi 0
            norm[c] = min_vals[c] / denom

    return norm.round(6), X.round(6), max_vals.round(6), min_vals.round(6)


def weighted_saw(norm_df: pd.DataFrame, weights: dict):
    """
    Bobot * normalisasi, lalu scoring.
    """
    W = pd.Series(weights)
    weighted = norm_df * W

    score = weighted.sum(axis=1)
    rank = score.rank(ascending=False, method="min").astype(int)

    result = pd.DataFrame({
        "score": score.round(6),
        "rank": rank
    })

    return weighted.round(6), result


def saw_full_process(df, criteria_meta, weights):
    """
    Full pipeline SAW:
    1. raw_matrix
    2. normalized
    3. weighted_matrix
    4. result (score + rank)
    5. max_vals, min_vals
    """
    norm, rawX, max_vals, min_vals = normalize_saw(df, criteria_meta)
    weighted, result = weighted_saw(norm, weights)

    return {
        "raw_matrix": rawX,
        "normalized": norm,
        "weighted_matrix": weighted,
        "result": result,
        "max_vals": max_vals,
        "min_vals": min_vals,
        "weights": pd.Series(weights)
    }
