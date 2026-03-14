import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & ACADEMIC STYLING
# -------------------------------------------------------------------------
st.set_page_config(page_title="Bioprocess Optimization Portal", layout="wide")

# Plotly academic template (white background, clear gridlines)
layout_template = go.layout.Template(
    layout=go.Layout(plot_bgcolor="white", paper_bgcolor="white", 
                     xaxis=dict(showgrid=True, gridcolor='lightgrey', linecolor='black'),
                     yaxis=dict(showgrid=True, gridcolor='lightgrey', linecolor='black'))
)

# -------------------------------------------------------------------------
# 2. MODEL INITIALIZATION (TRAINED ON SYNTHETIC BASELINE FOR DEMO)
# -------------------------------------------------------------------------
@st.cache_resource
def train_models():
    # Replace this block with actual CCD training data ingestion later
    np.random.seed(42)
    n_samples = 100
    X_train = np.random.uniform(low=[5, 0.5, 1, 6], high=[15, 2, 5, 8], size=(n_samples, 4))
    
    # Synthetic biological logic (PHA prefers high C/N, BS prefers moderate, CDW prefers both high)
    Y_pha = 180 - 1.5 * (X_train[:, 0] - 12)**2 - 20 * (X_train[:, 1] - 1.0)**2 + np.random.normal(0, 5, n_samples)
    Y_bs = 80 - 2.0 * (X_train[:, 0] - 8)**2 - 15 * (X_train[:, 3] - 7.2)**2 + np.random.normal(0, 5, n_samples)
    Y_cdw = 0.5 + 0.1 * X_train[:, 0] + 0.5 * X_train[:, 1] + np.random.normal(0, 0.1, n_samples)
    
    Y_train = np.column_stack((np.clip(Y_pha, 0, None), np.clip(Y_bs, 0, None), np.clip(Y_cdw, 0, None)))
    
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_train)
    Y_scaled = scaler_Y.fit_transform(Y_train)
    
    # Deep Model (ANN)
    ann = MLPRegressor(hidden_layer_sizes=(15, 15), activation='tanh', solver='lbfgs', max_iter=5000)
    ann.fit(X_scaled, Y_scaled)
    
    # Light Model (Gaussian Process with Matern kernel for spatial correlation)
    kernel = Matern(nu=2.5)
    gp_pha = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=0.1)
    gp_bs = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=0.1)
    gp_cdw = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=0.1)
    
    gp_pha.fit(X_scaled, Y_scaled[:, 0])
    gp_bs.fit(X_scaled, Y_scaled[:, 1])
    gp_cdw.fit(X_scaled, Y_scaled[:, 2])
    
    return scaler_X, scaler_Y, ann, gp_pha, gp_bs, gp_cdw

scaler_X, scaler_Y, ann, gp_pha, gp_bs, gp_cdw = train_models()

# -------------------------------------------------------------------------
# 3. PORTAL UI & INTERACTIVITY
# -------------------------------------------------------------------------
st.title("Dual PHA/Biosurfactant Prediction & Optimization Portal")
st.markdown("Compare Deep (ANN) and Light (GP) predictions, extrapolate yields, and validate against wet-lab data.")

# --- Consensus Optimal Formulation ---
st.header("1. Recommended Consensus Formulation")
st.info("**Optimal Parameters:** Glucose: 10.07 g/L | NH₄Cl: 1.15 g/L | PO₄: 5.00 g/L | pH: 7.39")
col1, col2, col3 = st.columns(3)
col1.metric("Predicted PHA Yield", "171.06 mg")
col2.metric("Predicted BS Yield", "70.29 mg")
col3.metric("Predicted Bacteria Weight (CDW)", "1.95 g")

# --- Manual Parameter Tweaking ---
st.header("2. Extrapolate Yields (Parameter Tweaking)")
st.sidebar.header("Culture Media Parameters")
gluc = st.sidebar.slider("Glucose (g/L)", 5.0, 20.0, 10.0, 0.5)
nh4cl = st.sidebar.slider("NH₄Cl (g/L)", 0.1, 3.0, 1.0, 0.1)
po4 = st.sidebar.slider("PO₄ Buffer (g/L)", 1.0, 10.0, 3.5, 0.5)
ph = st.sidebar.slider("pH", 5.0, 9.0, 7.0, 0.1)

# Predict based on sliders
X_input = np.array([[gluc, nh4cl, po4, ph]])
X_input_scaled = scaler_X.transform(X_input)

# Deep Prediction
ann_pred_scaled = ann.predict(X_input_scaled)
ann_pred = scaler_Y.inverse_transform(ann_pred_scaled)[0]

# Light Prediction
gp_pha_pred, gp_pha_std = gp_pha.predict(X_input_scaled, return_std=True)
gp_bs_pred, gp_bs_std = gp_bs.predict(X_input_scaled, return_std=True)
gp_cdw_pred, gp_cdw_std = gp_cdw.predict(X_input_scaled, return_std=True)
gp_pred_scaled = np.array([[gp_pha_pred[0], gp_bs_pred[0], gp_cdw_pred[0]]])
gp_pred = scaler_Y.inverse_transform(gp_pred_scaled)[0]

# Display Side-by-Side
c1, c2 = st.columns(2)
with c1:
    st.subheader("Deep Model (ANN) Prediction")
    st.write(f"**PHA:** {ann_pred[0]:.2f} mg")
    st.write(f"**Biosurfactant:** {ann_pred[1]:.2f} mg")
    st.write(f"**Bacteria Weight (CDW):** {ann_pred[2]:.2f} g")

with c2:
    st.subheader("Light Model (Gaussian Process) Prediction")
    st.write(f"**PHA:** {gp_pred[0]:.2f} mg")
    st.write(f"**Biosurfactant:** {gp_pred[1]:.2f} mg")
    st.write(f"**Bacteria Weight (CDW):** {gp_pred[2]:.2f} g")

# --- Wet-Lab Validation & Charting ---
st.header("3. Validation: Upload Wet-Lab Results")
st.markdown("Upload a CSV with columns: `Glucose`, `NH4Cl`, `PO4`, `pH`, `Actual_PHA`, `Actual_BS`, `Actual_CDW`.")

uploaded_file = st.file_uploader("Upload CSV Data", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    required_cols = ['Glucose', 'NH4Cl', 'PO4', 'pH', 'Actual_PHA', 'Actual_BS', 'Actual_CDW']
    
    if all(col in df.columns for col in required_cols):
        # Generate Predictions for Uploaded Data
        X_test = df[['Glucose', 'NH4Cl', 'PO4', 'pH']].values
        X_test_scaled = scaler_X.transform(X_test)
        
        ann_preds_scaled = ann.predict(X_test_scaled)
        ann_preds = scaler_Y.inverse_transform(ann_preds_scaled)
        
        df['Pred_PHA_Deep'] = ann_preds[:, 0]
        df['Pred_BS_Deep'] = ann_preds[:, 1]
        df['Pred_CDW_Deep'] = ann_preds[:, 2]
        
        st.success("Data processed successfully. Generating academic charts...")
        
        # Parity Plot Generation Function
        def plot_parity(df, actual_col, pred_col, title, unit):
            fig = go.Figure(layout=layout_template)
            # Perfect prediction reference line
            min_val = min(df[actual_col].min(), df[pred_col].min()) * 0.9
            max_val = max(df[actual_col].max(), df[pred_col].max()) * 1.1
            fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                     mode='lines', name='Perfect Prediction', line=dict(color='black', dash='dash')))
            # Actual data points
            fig.add_trace(go.Scatter(x=df[actual_col], y=df[pred_col],
                                     mode='markers', name='Data Points', marker=dict(size=10, color='blue', opacity=0.7)))
            fig.update_layout(title=title, xaxis_title=f"Actual Experimental Yield ({unit})", yaxis_title=f"Model Predicted Yield ({unit})")
            return fig

        t1, t2, t3 = st.tabs(["PHA Validation", "Biosurfactant Validation", "CDW Validation"])
        
        with t1:
            st.plotly_chart(plot_parity(df, 'Actual_PHA', 'Pred_PHA_Deep', "Parity Plot: Deep Model vs Actual PHA Yield", "mg"), use_container_width=True)
        with t2:
            st.plotly_chart(plot_parity(df, 'Actual_BS', 'Pred_BS_Deep', "Parity Plot: Deep Model vs Actual Biosurfactant", "mg"), use_container_width=True)
        with t3:
            st.plotly_chart(plot_parity(df, 'Actual_CDW', 'Pred_CDW_Deep', "Parity Plot: Deep Model vs Actual CDW", "g"), use_container_width=True)
            
        st.dataframe(df.style.format("{:.2f}"))
        
    else:
        st.error(f"Missing required columns. Please ensure CSV has: {required_cols}")