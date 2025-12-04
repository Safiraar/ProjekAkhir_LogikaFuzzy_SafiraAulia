# libs/wp.py
import pandas as pd
import numpy as np

def wp_full_process(df: pd.DataFrame, criteria_meta: dict, weights: dict):
    """
    Weighted Product steps:
    1) For cost criteria, convert weight to negative (power)
    2) Compute S_i = product_j (x_ij ^ w_j)
    3) Compute V_i = S_i / sum(S_i)
    Return dict with intermediate matrices and result.
    """
    crits = [c for c in criteria_meta.keys()]
    X = df[crits].astype(float).copy()
    # weights as pandas series
    W = pd.Series(weights)
    # For cost, weight becomes negative exponent
    exp = W.copy()
    for c in crits:
        if criteria_meta[c]['attr'] == 'cost':
            exp[c] = -abs(W[c])
    # compute S_i
    # to avoid nan/inf if x=0, add tiny epsilon
    eps = 1e-9
    S = X.clip(lower=eps).pow(exp).prod(axis=1)
    V = S / S.sum()
    result = pd.DataFrame({"S": S, "V": V})
    result['rank'] = result['V'].rank(ascending=False, method='min').astype(int)
    return {
        "raw_matrix": X,
        "exponents": exp,
        "S": S.round(12),
        "V": V.round(6),
        "result": result.round(6),
        "weights": W
    }
