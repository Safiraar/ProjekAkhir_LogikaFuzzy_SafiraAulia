import streamlit as st
import pandas as pd
from libs import utils
from libs import saw as saw_mod
from libs import wp as wp_mod

st.title("Perhitungan SAW & WP")

# load data & criteria
df_init = utils.load_data()
criteria_meta = utils.DEFAULT_CRITERIA
default_weights = {k: criteria_meta[k]["weight"] for k in criteria_meta}

# session state editable data
if "df" not in st.session_state:
    st.session_state.df = df_init.copy()

if "weights" not in st.session_state:
    st.session_state.weights = default_weights.copy()

st.subheader("Edit Tabel Alternatif")

edited = st.experimental_data_editor(st.session_state.df, num_rows="dynamic")

if st.button("Simpan Tabel"):
    st.session_state.df = edited.copy()
    st.success("Tabel diperbarui!")

st.subheader("Edit Bobot Kriteria")

cols = st.columns(len(criteria_meta))
new_weights = {}

# pastikan iterasi sesuai keys urutan
keys = list(criteria_meta.keys())
for i, c in enumerate(keys):
    with cols[i]:
        new_weights[c] = st.number_input(
            f"{c} - {criteria_meta[c]['name']}",
            value=float(st.session_state.weights.get(c, criteria_meta[c]['weight'])),
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            key=f"w_{c}"
        )

if st.button("Simpan Bobot"):
    st.session_state.weights = utils.validate_weights(new_weights)
    st.success("Bobot diperbarui!")

if st.button("Hitung SAW & WP"):
    df = st.session_state.df.copy()
    weights = st.session_state.weights

    saw = saw_mod.saw_full_process(df, criteria_meta, weights)
    wp = wp_mod.wp_full_process(df, criteria_meta, weights)

    st.session_state.last_results = {"saw": saw, "wp": wp}

    st.success("Perhitungan selesai! Buka halaman Pembanding.")
