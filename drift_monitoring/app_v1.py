import streamlit as st
import pandas as pd
import numpy as np
from prediction_model.config import config
from drift_monitoring.drift_detector import AudioDriftDetector
from prediction_model.processing.preprocessing import CLUSTER_ARCHETYPES

st.set_page_config(page_title="Spotify MLOps - Drift Monitor", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
    .main { background-color: #0b0f19; color: #f8fafc; }
    .stMetric { background: rgba(30, 41, 59, 0.7); border-radius: 10px; padding: 12px; border: 1px solid rgba(255,255,255,0.08); }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ Spotify MLOps: Data Drift & Feature Quality Monitor")
st.caption("Continuous Monitoring & Statistical Divergence Analysis (Kolmogorov-Smirnov & Wasserstein Distance)")

detector = AudioDriftDetector()
baseline_df = detector.load_baseline()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Baseline Samples", f"{len(baseline_df):,}")
with col2:
    st.metric("Monitored Features", f"{len(config.AUDIO_FEATURES)}")
with col3:
    st.metric("Significance Threshold (α)", f"{config.DRIFT_P_VALUE_THRESHOLD}")
with col4:
    st.metric("Model Version", config.MODEL_VERSION)

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📊 Live Drift Test", "🎯 Test Simulated Drift", "🔍 Feature Distribution Comparison"])

with tab1:
    st.subheader("Test Current Production Batch for Distribution Drift")
    uploaded_file = st.file_uploader("Upload incoming inference CSV batch", type=["csv"])
    
    if uploaded_file is not None:
        curr_df = pd.read_csv(uploaded_file)
        results = detector.detect_drift(curr_df)
        
        status_color = "green" if results["status"] == "HEALTHY" else "orange" if results["status"] == "DRIFT_WARNING" else "red"
        st.markdown(f"### Status: :{status_color}[{results['status']}]")
        st.write(results["status_description"])
        
        c_a, c_b = st.columns(2)
        c_a.metric("Drift Share", f"{results['drift_share_percent']}%")
        c_b.metric("Drifted Features", f"{results['drifted_features_count']} / {results['total_features_tested']}")
        
        st.dataframe(pd.DataFrame(results["feature_drifts"]).T)

with tab2:
    st.subheader("Simulate Production Data Shifts")
    drift_scenario = st.selectbox("Select Scenario", [
        ("retro_acoustic", "Retro 1930s Acoustic Shift (Loudness drop, High Acousticness)"),
        ("high_bpm_edm", "Modern High-BPM EDM Wave (High Danceability & Energy, High Tempo)")
    ], format_func=lambda x: x[1])
    
    if st.button("Run Drift Simulation"):
        sim_df = detector.generate_synthetic_drift_batch(drift_type=drift_scenario[0], n_samples=300)
        sim_results = detector.detect_drift(sim_df)
        
        st.warning(f"Detected Status: {sim_results['status']} ({sim_results['drift_share_percent']}% features drifted)")
        st.write(sim_results["status_description"])
        
        drift_table = []
        for feat, d in sim_results["feature_drifts"].items():
            drift_table.append({
                "Feature": feat,
                "Drifted?": "🚨 YES" if d["has_drifted"] else "✅ NO",
                "KS Statistic": d["ks_statistic"],
                "p-value": d["p_value"],
                "Baseline Mean": d["baseline_mean"],
                "Batch Mean": d["current_mean"]
            })
        st.dataframe(pd.DataFrame(drift_table))

with tab3:
    st.subheader("Baseline Feature Statistics")
    st.dataframe(baseline_df[config.AUDIO_FEATURES].describe().T)
