import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# ------------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="CAN Bus Intrusion Detection System (IDS)",
    page_icon="🛡️",
    layout="wide"
)

import os

# ------------------------------------------------------------------------------
# 2. LOAD TRAINED ARTIFACTS
# ------------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    try:
        # Use absolute path relative to app.py
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "oc_svm_model.joblib")
        prep_path = os.path.join(base_dir, "can_preprocessor.joblib")
        
        model = joblib.load(model_path)
        preprocessor = joblib.load(prep_path)
        return model, preprocessor, True
    except Exception as e:
        st.sidebar.error(f"❌ Real Error: {e}")
        return None, None, False

model, preprocessor, is_loaded = load_artifacts()

# ------------------------------------------------------------------------------
# 3. SIDEBAR: METADATA
# ------------------------------------------------------------------------------
st.sidebar.title("🛡️ Automotive IDS")
st.sidebar.markdown("**Protocol:** Classical CAN (ISO 11898-1)")
st.sidebar.markdown("**Model:** One-Class SVM (RBF Kernel)")
st.sidebar.markdown("**Approach:** Unsupervised Novelty Detection")
st.sidebar.divider()

if is_loaded:
    st.sidebar.success("✅ Model Artifacts Loaded")
    st.sidebar.markdown(f"**Kernel:** `{model.kernel.upper()}`")
    st.sidebar.markdown(f"**Margin ($\\nu$):** `{model.nu}`")
    st.sidebar.markdown(f"**Support Vectors:** `{len(model.support_):,}`")
else:
    st.sidebar.warning("⚠️ Model Artifacts Missing")
    st.sidebar.info("Ensure `oc_svm_model.joblib` and `can_preprocessor.joblib` are uploaded to your repository.")

st.sidebar.divider()
st.sidebar.caption("University of Mindanao | CCE105 Milestone 2")

# ------------------------------------------------------------------------------
# 4. MAIN INTERFACE
# ------------------------------------------------------------------------------
st.title("🚗 In-Vehicle CAN Bus Intrusion Detection System")
st.markdown("""
This application detects out-of-distribution message payloads on in-vehicle CAN networks. 
Arbitration IDs (`CAN_ID`) and timestamps are excluded to prevent shortcut learning, evaluating messages 
strictly across **Data Length Code (`DLC`)** and payload bytes (`Payload_Byte_0` through `Payload_Byte_7`).
""")

tab1, tab2, tab3 = st.tabs([
    "🔍 Real-Time Frame Inspector", 
    "📂 Batch CSV Log Analysis", 
    "📊 Model Performance Receipts"
])

# ------------------------------------------------------------------------------
# TAB 1: REAL-TIME FRAME INSPECTOR
# ------------------------------------------------------------------------------
with tab1:
    st.subheader("Manual CAN Frame Payload Inspection")
    
    # 1. Initialize session state defaults if not already set
    if "dlc_val" not in st.session_state:
        st.session_state["dlc_val"] = 8
    
    default_hex = ["FE", "5B", "00", "00", "00", "3C", "00", "00"]
    for i in range(8):
        if f"byte_{i}" not in st.session_state:
            st.session_state[f"byte_{i}"] = default_hex[i]

    # 2. Callback functions to directly set session state
    def load_normal_preset():
        st.session_state["dlc_val"] = 8
        normal_bytes = ["05", "21", "68", "09", "21", "21", "00", "6F"]
        for idx, b in enumerate(normal_bytes):
            st.session_state[f"byte_{idx}"] = b

    def load_dos_preset():
        st.session_state["dlc_val"] = 8
        for idx in range(8):
            st.session_state[f"byte_{idx}"] = "00"

    # 3. Preset Buttons connected via on_click
    col_preset1, col_preset2 = st.columns(2)
    with col_preset1:
        st.button("🟢 Load Preset: Legitimate Engine Telemetry", on_click=load_normal_preset)
    with col_preset2:
        st.button("🔴 Load Preset: Injected DoS Attack Frame (00 00 ... 00)", on_click=load_dos_preset)

    # 4. Slider tied to session state
    dlc = st.slider("Data Length Code (DLC):", min_value=1, max_value=8, key="dlc_val")

    st.markdown("**Payload Data Field [Bytes 0 to 7] (Base-16 Hexadecimal):**")
    cols = st.columns(8)
    byte_inputs = []

    # 5. Text inputs bound directly to session state
    for i in range(8):
        with cols[i]:
            disabled = i >= dlc
            val = st.text_input(f"Byte {i}", max_chars=2, disabled=disabled, key=f"byte_{i}")
            
            if disabled:
                byte_inputs.append(-1.0)
            else:
                try:
                    byte_inputs.append(float(int(val.strip(), 16)))
                except ValueError:
                    byte_inputs.append(-1.0)

# ------------------------------------------------------------------------------
# TAB 2: BATCH CSV LOG ANALYSIS
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("Batch CAN Network Trace Evaluation")
    st.markdown("Upload a CSV file with columns `DLC` and `Payload_Byte_0` through `Payload_Byte_7`.")
    
    uploaded_file = st.file_uploader("Upload CAN Bus Trace (.csv)", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        required_cols = [
            "DLC", "Payload_Byte_0", "Payload_Byte_1", "Payload_Byte_2", 
            "Payload_Byte_3", "Payload_Byte_4", "Payload_Byte_5", 
            "Payload_Byte_6", "Payload_Byte_7"
        ]
        
        if all(col in batch_df.columns for col in required_cols):
            if st.button("🚀 Run Batch Intrusion Detection"):
                if not is_loaded:
                    st.error("Model artifacts missing.")
                else:
                    with st.spinner("Processing network stream..."):
                        X_eval = batch_df[required_cols].astype(float)
                        X_prepared = preprocessor.transform(X_eval)
                        preds = model.predict(X_prepared)
                        scores = -model.decision_function(X_prepared)

                        batch_df["IDS_Verdict"] = np.where(preds == -1, "Attack (1)", "Normal (0)")
                        batch_df["Anomaly_Score"] = scores.round(4)

                        attack_count = (preds == -1).sum()
                        normal_count = (preds == 1).sum()

                        b1, b2, b3 = st.columns(3)
                        with b1:
                            st.metric("Total Processed Frames", f"{len(batch_df):,}")
                        with b2:
                            st.metric("Normal Frames", f"{normal_count:,}")
                        with b3:
                            st.metric("Attacks Intercepted", f"{attack_count:,}", delta=f"{(attack_count/len(batch_df)*100):.1f}% Detection Rate")

                        st.line_chart(batch_df["Anomaly_Score"], height=250)
                        st.dataframe(batch_df[["IDS_Verdict", "Anomaly_Score"] + required_cols].head(100))
        else:
            st.error(f"Missing required columns. Expected: `{required_cols}`")

# ------------------------------------------------------------------------------
# TAB 3: PERFORMANCE RECEIPTS
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("Milestone 2 Validation Metrics (Held-Out Test Set)")
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Accuracy", "97.46%")
    with m2:
        st.metric("Attack Recall", "100.0%", help="10,000 / 10,000 DoS frames detected (0 False Negatives)")
    with m3:
        st.metric("Precision", "95.16%")
    with m4:
        st.metric("ROC-AUC", "0.9778")
    with m5:
        st.metric("False Positive Rate", "5.09%")

    st.markdown("""
    ---
    ### Model Architectural Highlights:
    * **Zero Signature Dependency:** Trains exclusively on legitimate telemetry to catch zero-day DoS flooding.
    * **Shortcut-Resistant:** `CAN_ID` and `Timestamp` are dropped so the model evaluates payload structure rather than ID artifacts.
    * **Deterministic Latency:** Evaluates non-linear RBF kernel distance without deep recursive branching.
    """)