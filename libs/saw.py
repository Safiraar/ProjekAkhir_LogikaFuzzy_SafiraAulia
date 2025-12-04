# libs/saw.py
import pandas as pd

def normalize_saw(df: pd.DataFrame, criteria_meta: dict):
    """
    Normalisasi SAW:
      - benefit: r_ij = x_ij / max_j
      - cost   : r_ij = min_j / x_ij

    Inputs:
      df: DataFrame yang berisi kolom kriteria (mis. "C1","C2",...)
      criteria_meta: dict, mis. {"C1": {"attr":"cost"}, "C2":{"attr":"benefit"}, ...}

    Returns:
      normalized: DataFrame r_ij
      max_vals: Series max tiap kriteria
      min_vals: Series min tiap kriteria
    """
    crits = list(criteria_meta.keys())
    X = df[crits].astype(float).copy()

    max_vals = X.max()
    min_vals = X.min()

    norm = pd.DataFrame(index=X.index, columns=crits, dtype=float)

    for c in crits:
        attr = criteria_meta[c].get("attr", criteria_meta[c].get("type", "benefit"))
        if attr == "benefit":
            denom = max_vals[c] if max_vals[c] != 0 else 1.0
            norm[c] = X[c] / denom
        else:  # cost
            # r_ij = min_j / x_ij (hindari pembagi 0)
            norm[c] = min_vals[c] / X[c].replace(0, 1e-12)

    # round for neatness
    return norm.round(6), max_vals.round(6), min_vals.round(6)


def calculate_scores(norm_df: pd.DataFrame, weights: dict):
    """
    Menghitung skor SAW:
      score_i = sum_j ( r_ij * w_j )

    Returns:
      weighted_matrix: DataFrame r_ij * w_j
      result: DataFrame berisi 'score' dan 'rank' (rank 1 = terbesar)
    """
    W = pd.Series(weights)
    # pastikan kolom urut sesuai W index
    W = W.reindex(norm_df.columns)
    weighted = norm_df * W

    score = weighted.sum(axis=1)
    # ranking: highest score -> rank 1
    rank = score.rank(ascending=False, method="min").astype(int)

    result = pd.DataFrame({
        "score": score.round(6),
        "rank": rank
    })

    return weighted.round(6), result


def saw_full_process(df: pd.DataFrame, criteria_meta: dict, weights: dict):
    """
    Full SAW pipeline:
      - raw_matrix (input)
      - normalized (r_ij)
      - weighted_matrix (r_ij * w_j)
      - result (score & rank)
      - max_vals, min_vals
      - weights (Series)

    Returns dict with these keys so app.py dapat menampilkan tiap langkah.
    """
    # raw matrix (only criteria columns expected)
    crits = list(criteria_meta.keys())
    raw = df[crits].astype(float).copy()

    # 1) Normalisasi
    normalized, max_vals, min_vals = normalize_saw(raw, criteria_meta)

    # 2) Bobot & skor
    weighted_matrix, result = calculate_scores(normalized, weights)

    return {
        "raw_matrix": raw,
        "normalized": normalized,
        "weighted_matrix": weighted_matrix,
        "result": result,
        "max_vals": max_vals,
        "min_vals": min_vals,
        "weights": pd.Series(weights)
    }
