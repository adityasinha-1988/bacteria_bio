import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.graph_objects as go
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -------------------------------------------------------------------------
st.set_page_config(page_title="Bioprocess Optimization Portal", layout="wide")

layout_template = go.layout.Template(
    layout=go.Layout(plot_bgcolor="white", paper_bgcolor="white", 
                     xaxis=dict(showgrid=True, gridcolor='lightgrey', linecolor='black'),
                     yaxis=dict(showgrid=True, gridcolor='lightgrey', linecolor='black'))
)

# -------------------------------------------------------------------------
# 2. MODEL INITIALIZATION (SYNTHETIC BASELINE)
# -------------------------------------------------------------------------
@st.cache_resource
def train_models():
    np.random.seed(42)
    n_samples = 100
    X_train = np.random.uniform(low=[5, 0.5, 1, 6], high=[15, 2, 5, 8], size=(n_samples, 4))
    
    Y_pha = 180 - 1.5 * (X_train[:, 0] - 12)**2 - 20 * (X_train[:, 1] - 1.0)**2 + np.random.normal(0, 5, n_samples)
    Y_bs = 80 - 2.0 * (X_train[:, 0] - 8)**2 - 15 * (X_train[:, 3] - 7.2)**2 + np.random.normal(0, 5, n_samples)
    Y_cdw = 0.5 + 0.1 * X_train[:, 0] + 0.5 * X_train[:, 1] + np.random.normal(0, 0.1, n_samples)
    
    Y_train = np.column_stack((np.clip(Y_pha, 0, None), np.clip(Y_bs, 0, None), np.clip(Y_cdw, 0, None)))
    
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_train)
    Y_scaled = scaler_Y.fit_transform(Y_train)
    
    ann = MLPRegressor(hidden_layer_sizes=(15, 15), activation='tanh', solver='lbfgs', max_iter=5000)
    ann.fit(X_scaled, Y_scaled)
    
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

# --- Wet-Lab Validation & Data Upload ---
st.header("3. Validation: Upload Wet-Lab Results")

# Generate Excel Template in Memory
template_df = pd.DataFrame({
    'Experiment_Number': [1, 2],
    'Date': ['2026-03-14', '2026-03-15'],
    'Time': ['08:00', '10:30'],
    'Glucose': [10.0, 12.5], 
    'NH4Cl': [1.0, 1.5], 
    'PO4': [3.5, 4.0], 
    'pH': [7.0, 7.2],
    'Actual_PHA': [158.0, 165.0], 
    'Actual_BS': [75.0, 68.0], 
    'Actual_CDW': [1.98, 1.45]
})

buffer = io.BytesIO()
with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
    template_df.to_excel(writer, index=False, sheet_name='CCD_Data')
excel_data = buffer.getvalue()

st.markdown("Download the Excel template, fill it with your experimental results, and upload it below.")
st.download_button(
    label="📥 Download Excel Template", 
    data=excel_data, 
    file_name="bioprocess_ccd_template.xlsx", 
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# Upload and Process Excel Data
uploaded_file = st.file_uploader("Upload Populated Excel Data", type=["xlsx"])

if uploaded_file is not None:
    # Read the Excel file using openpyxl
    df = pd.read_excel(uploaded_file, engine='openpyxl')
    
    # Required columns for prediction
    required_cols = ['Glucose', 'NH4Cl', 'PO4', 'pH', 'Actual_PHA', 'Actual_BS', 'Actual_CDW']
    
    if all(col in df.columns for col in required_cols):
        X_test = df[['Glucose', 'NH4Cl', 'PO4', 'pH']].values
        X_test_scaled = scaler_X.transform(X_test)
        
        # Deep Predictions
        ann_preds_scaled = ann.predict(X_test_scaled)
        ann_preds = scaler_Y.inverse_transform(ann_preds_scaled)
        
        df['Pred_PHA_Deep'] = ann_preds[:, 0]
        df['Pred_BS_Deep'] = ann_preds[:, 1]
        df['Pred_CDW_Deep'] = ann_preds[:, 2]
        
        # Chart Generation
        def plot_parity(df, actual_col, pred_col, title, unit):
            fig = go.Figure(layout=layout_template)
            min_val = min(df[actual_col].min(), df[pred_col].min()) * 0.9
            max_val = max(df[actual_col].max(), df[pred_col].max()) * 1.1
            fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                     mode='lines', name='Perfect Prediction', line=dict(color='black', dash='dash')))
            
            # Hover text to include tracking info if available
            hover_text = df.apply(lambda row: f"Exp: {row.get('Experiment_Number', 'N/A')} | Date: {row.get('Date', 'N/A')}", axis=1)
            
            fig.add_trace(go.Scatter(x=df[actual_col], y=df[pred_col],
                                     mode='markers', name='Data Points', text=hover_text,
                                     marker=dict(size=12, color='rgba(255, 99, 71, 0.8)', line=dict(width=2, color='darkred'))))
            fig.update_layout(title=title, xaxis_title=f"Actual Yield ({unit})", yaxis_title=f"Predicted Yield ({unit})")
            return fig

        t1, t2, t3 = st.tabs(["PHA Analysis", "Biosurfactant Analysis", "Biomass (CDW) Analysis"])
        
        with t1:
            st.plotly_chart(plot_parity(df, 'Actual_PHA', 'Pred_PHA_Deep', "Parity Plot: Deep Model vs Actual PHA", "mg"), use_container_width=True)
        with t2:
            st.plotly_chart(plot_parity(df, 'Actual_BS', 'Pred_BS_Deep', "Parity Plot: Deep Model vs Actual Biosurfactant", "mg"), use_container_width=True)
        with t3:
            st.plotly_chart(plot_parity(df, 'Actual_CDW', 'Pred_CDW_Deep', "Parity Plot: Deep Model vs Actual CDW", "g"), use_container_width=True)
            
        st.dataframe(df.style.format(precision=2))
        
    else:
        st.error(f"Missing required mathematical columns. Please ensure the Excel file contains: {required_cols}")
