import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")
st.title("Crystal of Atlan - Enhance Tracker & Kalkulator (20% Success, 12x)")

p = 0.2
max_tries = 12
runs = 50000

if "records" not in st.session_state:
    st.session_state.records = []

col1, col2 = st.columns([1,2])
with col1:
    st.header("Input Live dari Player")
    attempt_type = st.selectbox("Jenis percobaan", ["tumbal", "main"], key="atype")
    result = st.radio("Hasil", ["success", "fail"], index=1, key="res")
    if st.button("Rekam percobaan"):
        st.session_state.records.append({"type": attempt_type, "result": result})
    if st.button("Reset rekaman"):
        st.session_state.records = []

    st.markdown("---")
    st.header("Import dari Log / CSV")
    uploaded_file = st.file_uploader("Upload file log/CSV (format: type,result per baris)")
    if uploaded_file is not None:
        try:
            df_in = pd.read_csv(uploaded_file, header=None)
            df_in.columns = ["type", "result"]
            for _, r in df_in.iterrows():
                st.session_state.records.append({"type": str(r["type"]).strip(), "result": str(r["result"]).strip()})
            st.success(f"{len(df_in)} baris ditambahkan dari file.")
        except Exception:
            st.error("Gagal membaca file. Pastikan format CSV sederhana tanpa header: type,result per baris.")

    if st.button("Download rekaman (CSV)" ):
        if st.session_state.records:
            out_df = pd.DataFrame(st.session_state.records)
            csv = out_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", data=csv, file_name="enhance_records.csv", mime="text/csv")
        else:
            st.info("Tidak ada rekaman untuk di-download.")

with col2:
    st.header("Ringkasan & Statistik Otomatis")
    if not st.session_state.records:
        st.info("Belum ada rekaman. Rekam percobaan pakai tombol di kiri atau upload log.")
    else:
        df = pd.DataFrame(st.session_state.records)
        df.index = np.arange(1, len(df) + 1)
        st.subheader("Daftar percobaan (terbaru di bawah)")
        st.dataframe(df.tail(200))

        # compute basic stats
        total_attempts = len(df)
        main_attempts = df[df["type"] == "main"].shape[0]
        main_successes = df[(df["type"] == "main") & (df["result"] == "success")].shape[0]
        overall_successes = df[df["result"] == "success"].shape[0]

        cols_stat = st.columns(4)
        cols_stat[0].metric("Total percobaan", total_attempts)
        cols_stat[1].metric("Main attempts", main_attempts)
        cols_stat[2].metric("Main sukses", main_successes)
        cols_stat[3].metric("Total sukses", overall_successes)

        # Compute empirical probability of success on a main attempt
        emp_main_rate = (main_successes / main_attempts) if main_attempts > 0 else np.nan
        st.write(f"Empirical success rate on MAIN attempts: {emp_main_rate:.3f}" if not np.isnan(emp_main_rate) else "Belum ada main attempt")

        # Compute probability of at least one main success within 12 total attempts using current records as starting prefix
        # We'll treat the recorded sequence as prefix and simulate continuation to estimate probability of achieving main success within 12 total attempts
        def prob_finish_from_prefix(prefix_records, T_choice=None, sims=20000):
            prefix_len = len(prefix_records)
            # if already reached target within prefix
            for i, rec in enumerate(prefix_records, start=1):
                if rec["type"] == "main" and rec["result"] == "success" and i <= max_tries:
                    return 1.0
            if prefix_len >= max_tries:
                return 1.0 if any((rec["type"] == "main" and rec["result"] == "success") for rec in prefix_records[:max_tries]) else 0.0
            rng = np.random.default_rng(42)
            success_count = 0
            for _ in range(sims):
                ta = prefix_len
                # check if prefix already has main success
                if any((rec["type"] == "main" and rec["result"] == "success") for rec in prefix_records[:max_tries]):
                    success_count += 1
                    continue
                # simulate remaining attempts assuming player will follow same strategy T_choice (if provided), otherwise assume they will do main every attempt
                while ta < max_tries:
                    ta += 1
                    if rng.random() < p:
                        # success could be on tumbal or main; only count if main
                        # assume non-specified attempts are MAIN for this simulation
                        success_count += 1
                        break
            return success_count / sims

        prob_est = prob_finish_from_prefix(st.session_state.records)
        st.write(f"Estimasi peluang mencapai minimal 1 main-success dalam total ≤ {max_tries} percobaan (menggunakan asumsi main selanjutnya): {prob_est:.3f}")

        st.subheader("Perbandingan dengan Teori (simulasi)")
        # Show simulated curves for strategies T=0..3
        def simulate_strategy_fixed(T, runs, p, max_tries):
            rng = np.random.default_rng()
            total_attempts = np.zeros(runs, dtype=int)
            for i in range(runs):
                ta = 0
                success = False
                while ta < max_tries:
                    if T > 0:
                        for _ in range(T):
                            ta += 1
                            if ta >= max_tries:
                                break
                            if rng.random() < p:
                                success = True
                                break
                    if success or ta >= max_tries:
                        break
                    ta += 1
                    if rng.random() < p:
                        success = True
                        break
                total_attempts[i] = ta if success else max_tries
            return total_attempts

        chart_data = {"N": np.arange(1, max_tries + 1), "Analitik": 1 - (1 - p) ** np.arange(1, max_tries + 1)}
        for T in [0,1,2,3]:
            sim = simulate_strategy_fixed(T, 20000, p, max_tries)
            emp_cdf = np.array([(sim <= k).mean() for k in chart_data["N"]])
            chart_data[f"T={T}"] = emp_cdf
        chart_df = pd.DataFrame(chart_data).set_index("N")
        st.line_chart(chart_df)

        st.subheader("Heatmap terakhir (probabilitas sukses ≤ N)")
        heatmap_df = chart_df.copy()
        fig, ax = plt.subplots(figsize=(8,3))
        sns.heatmap(heatmap_df.T, annot=True, fmt=".2f", cbar=True, ax=ax)
        st.pyplot(fig)

st.sidebar.header("Pengaturan")
st.sidebar.write("Peluang tiap attempt = 20% (tetap)")

