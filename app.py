# app.py
import streamlit as st
import pandas as pd
from libs import utils
from libs import saw as saw_mod
from libs import wp as wp_mod

st.set_page_config(page_title="Sistem Pendukung Keputusan dalam Layanan Hosting Website dengan Logika Fuzzy (Metode SAW & Metode WP)", layout="wide")

# --- load initial data ---
@st.cache_data
def load_initial():
    df = utils.load_data()
    criteria_meta = utils.DEFAULT_CRITERIA
    # default weights
    weights = {k: criteria_meta[k]['weight'] for k in criteria_meta}
    return df, criteria_meta, weights

df_init, criteria_meta, default_weights = load_initial()

# --- Sidebar navigation ---
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["Home", "Perhitungan", "Pembanding"])

# --- Home page ---
if page == "Home":
    st.title("Sistem Pendukung Keputusan dalam Layanan Hosting Website dengan Logika Fuzzy (Metode SAW & Metode WP")
    st.text("Dibuat oleh Safira Aulia Rahma (4611422125)")
    st.markdown("""
    **Deskripsi**  
    Aplikasi ini menghitung peringkat alternatif berdasarkan metode **Simple Additive Weighting (SAW)**  
    dan **Weighted Product (WP)**. Data awal (alternatif, nilai kriteria, bobot) dimuat dari file sumber (sama persis dengan penghitungan manual Anda).  
    """)
    st.info("Sumber data awal digunakan sesuai dengan penghitungan manual di excel.")
    # show criteria table
    st.subheader("Kriteria & Bobot (default)")
    crit_table = pd.DataFrame.from_dict({k: {"Nama": criteria_meta[k]['name'],
                                             "Atribut": criteria_meta[k]['attr'],
                                             "Bobot (default)": criteria_meta[k]['weight']} for k in criteria_meta},
                                        orient="index")
    crit_table.index.name = "Kode Kriteria"
    st.table(crit_table)

    st.subheader("Data Alternatif (nilai awal)")
    st.dataframe(df_init)
    st.caption("Alternatif & nilai awal (C1..C5) sesuai tabel manual. :contentReference[oaicite:1]{index=1}")

# --- Perhitungan page ---
elif page == "Perhitungan":
    st.title("Perhitungan SAW & WP")
    st.markdown("Anda bisa mengedit alternatif (baris) dan bobot kriteria di sini. Setelah selesai, klik **Hitung** untuk melihat perhitungan lengkap.")
    # copy initial data to session state for editability
    if "df" not in st.session_state:
        st.session_state.df = df_init.copy()
    if "weights" not in st.session_state:
        st.session_state.weights = default_weights.copy()

    # show editable data editor
    st.subheader("Tabel Alternatif (editable)")
    edited = st.data_editor(st.session_state.df, num_rows="dynamic")
    if st.button("Simpan perubahan tabel"):
        st.session_state.df = edited.copy()
        st.success("Perubahan disimpan ke session.")

    st.subheader("Atur Bobot Kriteria")
    cols = st.columns(len(criteria_meta))
    new_weights = {}
    for i, c in enumerate(criteria_meta.keys()):
        with cols[i]:
            w = st.number_input(label=f"{c} ({criteria_meta[c]['name']})",
                                 min_value=0.0, max_value=1.0,
                                 value=float(st.session_state.weights.get(c, criteria_meta[c]['weight'])),
                                 step=0.01, key=f"w_{c}")
            new_weights[c] = float(w)
    if st.button("Simpan bobot"):
        st.session_state.weights = utils.validate_weights(new_weights)
        st.success(f"Bobot tersimpan (ternormalisasi). Total bobot = {sum(st.session_state.weights.values()):.4f}")

    st.divider()
    st.write("**Preview bobot yg digunakan:**")
    st.write(pd.Series(st.session_state.weights).rename("bobot"))

    # perform calculations when requested
    if st.button("Hitung SAW & WP"):
        df = st.session_state.df.copy().reset_index(drop=True)
        weights = utils.validate_weights(st.session_state.weights)
        # SAW
        saw_proc = saw_mod.saw_full_process(df, criteria_meta, weights)
        wp_proc = wp_mod.wp_full_process(df, criteria_meta, weights)

        # Show SAW details
        st.subheader("Hasil SAW — Langkah demi langkah")
        with st.expander("1. Matriks Awal (X)"):
            st.dataframe(saw_proc['raw_matrix'])
        with st.expander("2. Normalisasi (r_ij)"):
            st.write("Untuk benefit: r_ij = x_ij / max_j ; untuk cost: r_ij = min_j / x_ij")
            st.dataframe(saw_proc['normalized'])
            st.caption(f"max per kriteria: {saw_proc['max_vals'].to_dict()} | min per kriteria: {saw_proc['min_vals'].to_dict()}")
        with st.expander("3. Perkalian dengan bobot (r_ij * w_j)"):
            st.write("Bobot yang dipakai:")
            st.write(saw_proc['weights'])
            st.dataframe(saw_proc['weighted_matrix'])
        with st.expander("4. Skor akhir & perankingan"):
            st.dataframe(saw_proc['result'].sort_values("score", ascending=False))

        # Show WP details
        st.subheader("Hasil WP — Langkah demi langkah")
        with st.expander("1. Matriks Awal (X)"):
            st.dataframe(wp_proc['raw_matrix'])
        with st.expander("2. Eksponen (w_j atau -w_j bila cost)"):
            st.write("Eksponen (untuk cost jadi negatif):")
            st.dataframe(wp_proc['exponents'].to_frame('exponent'))
        with st.expander("3. Hitung S_i = product(x_ij ^ exponent_j)"):
            st.dataframe(wp_proc['S'].to_frame("S"))
        with st.expander("4. Hitung V_i = S_i / sum(S_i) dan perankingan"):
            st.dataframe(wp_proc['result'].sort_values("V", ascending=False))

        # store results in session for comparison page
        st.session_state.last_results = {"saw": saw_proc, "wp": wp_proc}
        st.success("Perhitungan selesai. Hasil disimpan untuk halaman Pembanding.")

# --- Pembanding page ---
elif page == "Pembanding":
    st.title("Pembanding: SAW vs WP")
    if "last_results" not in st.session_state:
        st.warning("Belum ada perhitungan. Pergi ke halaman Perhitungan dan klik 'Hitung SAW & WP' terlebih dahulu.")
    else:
        saw_proc = st.session_state.last_results['saw']
        wp_proc = st.session_state.last_results['wp']

        st.subheader("Tabel Perbandingan Rangking")
        saw_ranks = saw_proc['result'][['score','rank']].rename(columns={"score":"score_saw","rank":"rank_saw"})
        wp_ranks = wp_proc['result'][['V','rank']].rename(columns={"V":"score_wp","rank":"rank_wp"})
        combined = pd.concat([saw_ranks, wp_ranks], axis=1)
        # index otomatis A1-A5 dari SAW
        combined.index = saw_proc['result'].index
        st.dataframe(combined)  # tidak usah sort, Anda ingin urutan tetap

        st.subheader("Analisis kecocokan")
        # check if top alternatives are same
        top_saw = combined['rank_saw'].idxmin()
        top_wp = combined['rank_wp'].idxmin()
        st.write(f"Top SAW: **{top_saw}**  — Top WP: **{top_wp}**")
        if top_saw == top_wp:
            st.success(f"Kedua metode setuju pada alternatif **{top_saw}**.")
        else:
            st.error("Metode menghasilkan alternatif terbaik yang **berbeda**.")
            st.write("Periksa detail di halaman Perhitungan untuk melihat langkah yang menyebabkan perbedaan.")
        # Also list any alternatives where ranking differs
        diffs = combined[combined['rank_saw'] != combined['rank_wp']]
        if not diffs.empty:
            st.subheader("Alternatif dengan perbedaan ranking")
            st.dataframe(diffs)
        else:
            st.write("Semua alternatif memiliki ranking yang sama pada kedua metode.")

st.sidebar.markdown("---")
st.sidebar.caption("Aplikasi by: Safira- Logika Fuzzy")
