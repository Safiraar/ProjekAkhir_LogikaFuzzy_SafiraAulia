import streamlit as st
import pandas as pd
from libs import utils

st.title("Home")

df_init, criteria_meta, default_weights = utils.load_data(), utils.DEFAULT_CRITERIA, {k: utils.DEFAULT_CRITERIA[k]["weight"] for k in utils.DEFAULT_CRITERIA}

st.subheader("Kriteria & Bobot (default)")

crit_table = pd.DataFrame.from_dict({
    k: {
        "Nama": criteria_meta[k]["name"],
        "Atribut": criteria_meta[k]["attr"],
        "Bobot": criteria_meta[k]["weight"]
    }
} for k in criteria_meta)

st.table(crit_table)

st.subheader("Data Alternatif (nilai awal)")
st.dataframe(df_init)
