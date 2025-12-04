import streamlit as st
import pandas as pd

st.title("Pembanding SAW vs WP")

if "last_results" not in st.session_state:
    st.warning("Belum ada hasil. Silakan hitung dulu di halaman Perhitungan.")
else:
    saw_proc = st.session_state.last_results["saw"]
    wp_proc = st.session_state.last_results["wp"]

    # pastikan index kompatibel (menggunakan index numeric yang sama)
    saw_r = saw_proc["result"][["score", "rank"]].rename(columns={"score": "score_saw", "rank": "rank_saw"})
    wp_r = wp_proc["result"][["V", "rank"]].rename(columns={"V": "score_wp", "rank": "rank_wp"})

    combined = pd.concat([saw_r, wp_r], axis=1)

    st.subheader("Perbandingan Ranking")
    st.dataframe(combined)

    # Ambil index nilai terkecil (rank 1)
    top_saw = combined["rank_saw"].idxmin()
    top_wp = combined["rank_wp"].idxmin()

    if top_saw == top_wp:
        st.success(f"Kedua metode sepakat: {top_saw}")
    else:
        st.error(f"Metode berbeda: SAW = {top_saw}, WP = {top_wp}")
