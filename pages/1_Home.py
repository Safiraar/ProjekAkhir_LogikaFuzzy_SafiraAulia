import streamlit as st
import pandas as pd
from libs import utils

st.title("Home")

# load data & criteria
df_init = utils.load_data()
criteria_meta = utils.DEFAULT_CRITERIA
default_weights = {k: criteria_meta[k]["weight"] for k in criteria_meta}

st.subheader("Kriteria & Bobot (default)")

crit_table = pd.DataFrame.from_dict({
    k: {
        "Nama": criteria_meta[k]["name"],
        "Atribut": criteria_meta[k]["attr"],
        "Bobot": criteria_meta[k]["weight"]
    } for k in criteria_meta
}).T  # transpose agar kolom menjadi Nama/Atribut/Bobot dan index = kode kriteria
                           

st.table(crit_table)

st.subheader("Data Alternatif (nilai awal)")
st.dataframe(df_init)
