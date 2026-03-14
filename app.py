import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# -------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# -------------------------------------------------------------------------
st.set_page_config(page_title="Bioprocess Optimization Portal", layout="wide")

layout_template = go.layout.Template(
    layout=go.Layout(plot_bgcolor="white", paper_bgcolor="white", 
                     xaxis=dict(showgrid=True, gridcolor='lightgrey', linecolor='black'),
                     yaxis=dict(showgrid=True, gridcolor='lightgrey', linecolor='black'))
)

# -------------------------------------------------------------------------
# 2. MODEL INITIALIZATION (CACHE BUSTED & ENSEMBLE)
# -------------------------------------------------------------------------
@st.cache_resource
def build_bioprocess_models_v3():
    np.random.seed(42)
    n_samples = 100
    X_train = np.random.uniform(low=[5, 0.5, 1, 6], high=[15, 2, 5, 8], size=(n_samples, 4))
    
    # Synthetic biological logic
    Y_pha = 180 - 1.5 * (X_train[:, 0] - 12)**2 - 20 * (X_train[:, 1] - 1.0)**2 + np.random.normal(0, 5, n_samples)
    Y_bs = 80 - 2.0 * (X_train[:, 0] - 8)**2 - 15 * (X_train[:, 3] - 7.2)**2 + np.random.normal(0, 5, n_samples)
    Y_cdw = 0.5 + 0.1 * X_train[:, 0] + 0.5 * X_train[:, 1] + np.random.normal(0, 0.1, n_samples)
    
    Y_train = np.column_stack((np.clip(Y_pha, 0, None), np.clip(Y_bs, 0, None), np.clip(Y_cdw, 0, None)))
    
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_train)
    Y_scaled = scaler_Y.fit_transform(Y_train)
    
    # Optimized Bagged Ensemble ANN
    base_ann = MLPRegressor(hidden_layer_sizes=(20, 20), activation='tanh', solver='lbfgs', max_iter=10000, alpha=0.05)
    bagged_ann = BaggingRegressor(estimator=base_ann, n_estimators=15, max_samples=0.85, n_jobs=-1, random_state=42)
    ann = MultiOutputRegressor(bagged_ann)
    ann.fit(X_scaled, Y_scaled)
    
    # Gaussian Process
    kernel = Matern(nu=2.5)
    gp_pha = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=0.1)
    gp_bs = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=0.1)
    gp_cdw = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=0.1)
    
    gp_pha.fit(X_scaled, Y_scaled[:, 0])
    gp_bs.fit(X_scaled, Y_scaled[:, 1])
    gp_cdw.fit(X_scaled, Y_scaled[:, 2])
    
    return scaler_X, scaler_Y, ann, gp_pha, gp_bs, gp_cdw

scaler_X, scaler_Y, ann, gp_pha, gp_bs, gp_cdw = build_bioprocess_models_v3()

# -------------------------------------------------------------------------
# 3. PORTAL UI: OPTIMAL FORMULATION
# -------------------------------------------------------------------------
st.title("Dual PHA/Biosurfactant Prediction & Optimization Portal")

st.header("1. Recommended Consensus Formulation")
st.info("**Optimal Parameters:** Glucose: 10.07 g/L | NH₄Cl: 1.15 g/L | PO₄: 5.00 g/L | pH: 7.39")
col1, col2, col3 = st.columns(3)
col1.metric("Predicted PHA Yield", "171.06 mg")
col2.metric("Predicted BS Yield", "70.29 mg")
col3.metric("Predicted Bacteria Weight (CDW)", "1.95 g")

st.divider()

# -------------------------------------------------------------------------
# 4. PORTAL UI: PARAMETER EXTRAPOLATION
# -------------------------------------------------------------------------
st.header("2. Extrapolate Yields (Parameter Tweaking)")
c_slider, c_chart = st.columns([1, 2])

with c_slider:
    st.subheader("Culture Media Parameters")
    gluc = st.slider("Glucose (g/L)", 5.0, 20.0, 10.0, 0.5)
    nh4cl = st.slider("NH₄Cl (g/L)", 0.1, 3.0, 1.0, 0.1)
    po4 = st.slider("PO₄ Buffer (g/L)", 1.0, 10.0, 3.5, 0.5)
    ph = st.slider("pH", 5.0, 9.0, 7.0, 0.1)

X_input = np.array([[gluc, nh4cl, po4, ph]])
X_input_scaled = scaler_X.transform(X_input)
ann_pred = scaler_Y.inverse_transform(ann.predict(X_input_scaled))[0]

gp_pred_scaled = np.array([[gp_pha.predict(X_input_scaled)[0], gp_bs.predict(X_input_scaled)[0], gp_cdw.predict(X_input_scaled)[0]]])
gp_pred = scaler_Y.inverse_transform(gp_pred_scaled)[0]

with c_chart:
    fig_live = go.Figure(data=[
        go.Bar(name='Deep Model (Ensemble ANN)', x=['PHA (mg)', 'Biosurfactant (mg)', 'CDW (g)'], y=[ann_pred[0], ann_pred[1], ann_pred[2]], marker_color='rgb(55, 83, 109)'),
        go.Bar(name='Light Model (GP)', x=['PHA (mg)', 'Biosurfactant (mg)', 'CDW (g)'], y=[gp_pred[0], gp_pred[1], gp_pred[2]], marker_color='rgb(26, 118, 255)')
    ])
    fig_live.update_layout(barmode='group', title='Real-Time Model Comparison', template=layout_template)
    st.plotly_chart(fig_live, use_container_width=True)

st.divider()

# -------------------------------------------------------------------------
# 5. PORTAL UI: WET-LAB VALIDATION (ERROR-RESISTANT)
# -------------------------------------------------------------------------
st.header("3. Validation: Upload Wet-Lab Results")

# Template Generation
template_df = pd.DataFrame({
    'Experiment_Number': [1, 2], 'Date': ['2026-03-14', '2026-03-15'], 'Time': ['08:00', '10:30'],
    'Glucose': [10.0, 12.5], 'NH4Cl': [1.0, 1.5], 'PO4': [3.5, 4.0], 'pH': [7.0, 7.2],
    'Actual_PHA': [158.0, 165.0], 'Actual_BS': [75.0, 68.0], 'Actual_CDW': [1.98, 1.45]
})
buffer = io.BytesIO()
with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
    template_df.to_excel(writer, index=False, sheet_name='CCD_Data')
excel_data = buffer.getvalue()

st.download_button(label="📥 Download Excel Template", data=excel_data, file_name="ccd_template.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

uploaded_file = st.file_uploader("Upload Populated Data (Excel or CSV)", type=["xlsx", "csv"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        
        required_cols = ['Glucose', 'NH4Cl', 'PO4', 'pH', 'Actual_PHA', 'Actual_BS', 'Actual_CDW']
        
        if all(col in df.columns for col in required_cols):
            # STAGE 1 CLEANING: Convert to numeric and force errors to NaN
            for col in required_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # STAGE 2 CLEANING: Remove any rows containing NaNs or blanks
            df = df.dropna(subset=required_cols)
            
            if len(df) == 0:
                st.error("Error: After cleaning blank rows, no valid data remains. Check your file.")
                st.stop()

            X_test = df[['Glucose', 'NH4Cl', 'PO4', 'pH']].values
            X_test_scaled = scaler_X.transform(X_test)
            ann_preds = scaler_Y.inverse_transform(ann.predict(X_test_scaled))
            
            df['Pred_PHA_Deep'], df['Pred_BS_Deep'], df['Pred_CDW_Deep'] = ann_preds[:, 0], ann_preds[:, 1], ann_preds[:, 2]
            
            def plot_parity(df, actual_col, pred_col, title, unit):
                fig = go.Figure(layout=layout_template)
                min_val, max_val = df[actual_col].min() * 0.9, df[actual_col].max() * 1.1
                fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', name='Ideal', line=dict(color='black', dash='dash')))
                fig.add_trace(go.Scatter(x=df[actual_col], y=df[pred_col], mode='markers', marker=dict(size=12, color='red', opacity=0.7)))
                fig.update_layout(title=title, xaxis_title=f"Actual ({unit})", yaxis_title=f"Predicted ({unit})")
                return fig

            def display_metrics(y_true, y_pred, unit):
                r2, rmse = r2_score(y_true, y_pred), np.sqrt(mean_squared_error(y_true, y_pred))
                mc1, mc2 = st.columns(2)
                mc1.metric("R² Score", f"{r2:.4f}")
                mc2.metric(f"RMSE ({unit})", f"{rmse:.2f}")

            t1, t2, t3 = st.tabs(["PHA", "Biosurfactant", "CDW"])
            with t1:
                st.plotly_chart(plot_parity(df, 'Actual_PHA', 'Pred_PHA_Deep', "PHA Parity", "mg"), use_container_width=True)
                display_metrics(df['Actual_PHA'], df['Pred_PHA_Deep'], "mg")
            with t2:
                st.plotly_chart(plot_parity(df, 'Actual_BS', 'Pred_BS_Deep', "BS Parity", "mg"), use_container_width=True)
                display_metrics(df['Actual_BS'], df['Pred_BS_Deep'], "mg")
            with t3:
                st.plotly_chart(plot_parity(df, 'Actual_CDW', 'Pred_CDW_Deep', "CDW Parity", "g"), use_container_width=True)
                display_metrics(df['Actual_CDW'], df['Pred_CDW_Deep'], "g")
            
            st.divider()
            st.dataframe(df.style.format(precision=2))
        else:
            st.error(f"Required columns missing. Expected: {required_cols}")
    except Exception as e:
        st.error(f"Critical Error: {str(e)}")
