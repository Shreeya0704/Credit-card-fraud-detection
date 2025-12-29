import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# 1. Page Configuration
st.set_page_config(
    page_title="Bank GenAI Fraud Auditor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Hardcoded Scenario Data ---
SCENARIO_1_TEXT = "POS Transaction: WALMART GROCERY #4582 - $45.20 (Card Verified)"
scenario_1_features = [0.0] * 30
scenario_1_features[29] = 45.20

SCENARIO_2_TEXT = "E-COMMERCE TRANSFER: OFFSHORE HOLDINGS LLC - $50,000.00 (Device Unknown)"
SCENARIO_2_FEATURES = [
    406.0, -2.312, 1.951, -1.609, 3.997, -0.522, -1.426, -2.537, 1.391, -2.770, 
    -2.772, 3.202, -2.899, -0.595, -4.289, 0.389, -1.140, -2.830, -0.016, 0.416, 
    0.126, 0.517, -0.035, -0.465, 0.320, 0.044, 0.177, 0.261, -0.143, 50000.0
]


# 2. Sidebar for Scenario Selection
with st.sidebar:
    st.title("👨‍💻 Case Simulation")
    st.write("Select a transaction scenario to audit.")
    
    scenario = st.radio(
        "Scenarios",
        ("✅ Scenario 1: Standard Transaction (Walmart)", "🚨 Scenario 2: High-Risk Anomaly (Offshore)"),
        label_visibility="collapsed"
    )

# 3. Data Logic based on Scenario
if scenario == "✅ Scenario 1: Standard Transaction (Walmart)":
    transaction_text = SCENARIO_1_TEXT
    features = scenario_1_features
else:
    transaction_text = SCENARIO_2_TEXT
    features = SCENARIO_2_FEATURES


# 4. Main UI Layout
st.header("🛡️ Bank GenAI Fraud Auditor", divider='rainbow')

st.subheader("Transaction Details")
st.info(transaction_text)

if st.button("Run Compliance Check", type="primary", width='stretch'):
    with st.spinner("Contacting Fraud Detection API..."):
        try:
            api_url = "http://127.0.0.1:8000/predict"
            payload = {"transaction_details": transaction_text, "data": features}
            
            response = requests.post(api_url, json=payload, timeout=20)
            response.raise_for_status()
            
            results = response.json()

            # 5. Results Display
            st.subheader("Analysis Results", divider='blue')
            col1, col2 = st.columns((2, 3))

            # Column 1: The Verdict & Gauge
            with col1:
                prediction = results.get('prediction')
                probability = results.get('probability', 0)
                risk_score = probability * 100

                # UI Bug Fix: Use st.success/st.error instead of st.metric
                if prediction == 0:
                    st.success("✅ NOT FRAUD - Transaction Approved")
                else:
                    st.error("🚨 FRAUD DETECTED - Account Frozen")

                # Gauge Chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=risk_score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Fraud Probability", 'font': {'size': 20}},
                    gauge={
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "darkred" if risk_score > 50 else "darkgreen"},
                        'steps': [
                            {'range': [0, 50], 'color': 'lightgreen'},
                            {'range': [50, 100], 'color': 'lightcoral'}
                        ],
                    }
                ))
                fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig, width='stretch')

            # Column 2: GenAI Analyst Report
            with col2:
                st.markdown("##### 📝 Suspicious Activity Report (SAR)")
                explanation = results.get("explanation", "No GenAI explanation required. The transaction is within normal parameters.")
                st.text_area("Generated Report", explanation, height=250, disabled=True)
                
        except requests.exceptions.RequestException as e:
            st.error(f"API Error: Could not connect to the backend service. Ensure the FastAPI server is running. Details: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: grey;'>"
    "System Status: <span style='color: green;'>Online</span> | "
    "Privacy Module: <span style='color: blue;'>Active (Microsoft Presidio)</span>"
    "</div>",
    unsafe_allow_html=True
)