# libs/saw.py
import pandas as pd
import numpy as np

def normalize_saw(df: pd.DataFrame, criteria_meta: dict):
    """
    Normalize matrix for SAW.
    For benefit: r_ij = x_ij / max_j
    For cost:    r_ij = min_j / x_ij
    Returns DataFrame of normalized values and intermediate matrices.
    """
    crits = [c for c in criteria_meta.keys()]
    X = df[crits].astype(float).copy()
    norm = pd.DataFrame(index=df.index, columns=crits, dtype=float)
    max_vals = X.max()
    min_vals = X.min()
    for c in crits:
        if criteria_meta[c]['attr'] == 'benefit':
            # guard divide by zero
            maxv = max_vals[c] if max_vals[c] != 0 else 1.0
            norm[c] = X[c] / maxv
        else:  # cost
            minv = min_vals[c] if min_vals[c] != 0 else 1.0
            # r_ij = min / x_ij
            norm[c] = minv / X[c]
    return norm.round(6), X.round(6), max_vals.round(6), min_vals.round(6)

def weight_and_score_saw(norm_df: pd.DataFrame, weights: dict):
    """
    Multiply normalized by weights and sum to get final score.
    weights: dict with keys matching norm_df.columns and values summing to 1 (or not)
    Returns weighted matrix, scores series.
    """
    # ensure weights aligned
    W = pd.Series(weights)
    weighted = norm_df * W
    scores = weighted.sum(axis=1)
    result = pd.DataFrame({
        "score": scores
    })
    result['rank'] = result['score'].rank(ascending=False, method='min').astype(int)
    return weighted.round(6), result.round(6)

# Helper to produce step-by-step dict
def saw_full_process(df, criteria_meta, weights):
    norm, rawX, max_vals, min_vals = normalize_saw(df, criteria_meta)
    weighted, result = weight_and_score_saw(norm, weights)
    return {
        "raw_matrix": rawX,
        "normalized": norm,
        "weighted_matrix": weighted,
        "result": result,
        "max_vals": max_vals,
        "min_vals": min_vals,
        "weights": pd.Series(weights)
    }
