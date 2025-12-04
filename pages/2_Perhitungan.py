import streamlit as st
import pandas as pd
from libs import utils
from libs import saw as saw_mod
from libs import wp as wp_mod

st.title("Perhitungan SAW & WP")

df_init, criteria_meta, default_weights = utils.load_data(), utils.DEFAULT_CRITERIA, {k: utils.DEFAULT_CRITERIA[k]["weight"] for k in utils.DEFAULT_CRITERIA}

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

for i, c in enumerate(criteria_meta):
    with cols[i]:
        new_weights[c] = st.number_input(
            c,
            value=float(st.session_state.weights[c]),
            min_value=0.0,
            max_value=1.0,
            step=0.01
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
