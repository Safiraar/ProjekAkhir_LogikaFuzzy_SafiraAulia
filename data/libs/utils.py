# libs/utils.py
import pandas as pd

# Default criteria metadata (nama, atribut, bobot). Bobot default sesuai PDF.
DEFAULT_CRITERIA = {
    "C1": {"name": "Harga", "attr": "cost", "weight": 0.30},
    "C2": {"name": "Website", "attr": "benefit", "weight": 0.25},
    "C3": {"name": "Storage", "attr": "benefit", "weight": 0.15},
    "C4": {"name": "Pengunjung", "attr": "benefit", "weight": 0.20},
    "C5": {"name": "Domain", "attr": "benefit", "weight": 0.10},
}

def load_data(path="data/alternatives.csv"):
    df = pd.read_csv(path)
    # Ensure kode is string
    df['kode'] = df['kode'].astype(str)
    return df

def validate_weights(weights: dict):
    # Normalize if not summing to 1
    s = sum(weights.values())
    if s == 0:
        raise ValueError("Total bobot tidak boleh 0")
    return {k: v/s for k,v in weights.items()}

def get_criteria_list():
    # returns list of criteria keys in order
    return ["C1","C2","C3","C4","C5"]
