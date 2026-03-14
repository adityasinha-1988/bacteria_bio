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
# 2. MODEL INITIALIZATION (ROBUST ENSEMBLE)
# -------------------------------------------------------------------------
@st.cache_resource
def build_final_ensemble_models():
    np.random.seed(42)
    n_samples = 150
    X_train = np.random.uniform(low=[5, 0.5, 1, 6], high=[15, 2, 5, 8], size=(n_samples, 4))
    
    Y_pha = 180 - 1.5 * (X_train[:, 0] - 12)**2 - 20 * (X_train[:, 1] - 1.0)**2 + np.random.normal(0, 5, n_samples)
    Y_bs = 80 - 2.0 * (X_train[:, 0] - 8)**2 - 15 * (X_train[:, 3] - 7.2)**2 + np.random.normal(0, 5, n_samples)
    Y_cdw = 0.5 + 0.1 * X_train[:, 0] + 0.5 * X_train[:, 1] + np.random.normal(0, 0.1, n_samples)
    
    Y_train = np.column_stack((np.clip(Y_pha, 0, None), np.clip(Y_bs, 0, None), np.clip(Y_cdw, 0, None)))
    
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_train)
    Y_scaled = scaler_Y.fit_transform(Y_train)
    
    # Deep Model: Multi-Output Bagging Ensemble
    base_ann = MLPRegressor(hidden_layer_sizes=(20, 20), activation='tanh', solver='lbfgs', max_iter=10000, alpha=0.05)
    bagged_ann = BaggingRegressor(estimator=base_ann, n_estimators=15, max_samples=0.85, n_jobs=-1, random_state=42)
    ann = MultiOutputRegressor(bagged_ann)
    ann.fit(X_scaled, Y_scaled)
    
    # Light Model: Gaussian Process
    kernel = Matern(nu=2.5)
    gp_pha = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=0.1)
    gp_bs = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=0.1)
    gp_cdw = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=0.1)
    
    gp_pha.fit(X_scaled, Y_scaled[:, 0])
    gp_bs.fit(X_scaled, Y_scaled[:, 1])
    gp_cdw.fit(X_scaled, Y_scaled[:, 2])
    
    return scaler_X, scaler_Y, ann, gp_pha, gp_bs, gp_cdw

scaler_X, scaler_Y, ann, gp_pha, gp_bs, gp_cdw = build_final_ensemble_models()

# -------------------------------------------------------------------------
# 3. PORTAL UI: TOP SECTION
# -------------------------------------------------------------------------
st.title("Dual PHA/Biosurfactant Prediction Tool")

st.header("1. Recommended Consensus Formulation")
st.success("**Target Optima:** Glucose: 10.07 g/L | NH₄Cl: 1.15 g/L | PO₄: 5.00 g/L | pH: 7.39")
c1, c2, c3 = st.columns(3)
c1.metric("PHA Yield", "171.06 mg")
c2.metric("BS Yield", "70.29 mg")
c3.metric("Bacteria Weight (CDW)", "1.95 g")

st.divider()

# -------------------------------------------------------------------------
# 4. PORTAL UI: LIVE PREDICTION (SLIDERS)
# -------------------------------------------------------------------------
st.header("2. Real-Time Parameter Simulation")
col_s, col_g = st.columns([1, 2])

with col_s:
    g_val = st.slider("Glucose (g/L)", 5.0, 20.0, 10.0)
    n_val = st.slider("NH₄Cl (g/L)", 0.1, 3.0, 1.0)
    p_val = st.slider("PO₄ (g/L)", 1.0, 10.0, 3.5)
    ph_val = st.slider("pH", 5.0, 9.0, 7.0)

x_in = scaler_X.transform([[g_val, n_val, p_val, ph_val]])
y_ann = scaler_Y.inverse_transform(ann.predict(x_in))[0]
y_gp = scaler_Y.inverse_transform([[gp_pha.predict(x_in)[0], gp_bs.predict(x_in)[0], gp_cdw.predict(x_in)[0]]])[0]

with col_g:
    fig = go.Figure(data=[
        go.Bar(name='Deep Model (ANN)', x=['PHA', 'BS', 'CDW'], y=y_ann, marker_color='#1f77b4'),
        go.Bar(name='Light Model (GP)', x=['PHA', 'BS', 'CDW'], y=y_gp, marker_color='#ff7f0e')
    ])
    fig.update_layout(barmode='group', template=layout_template, title="Comparison of Models")
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# -------------------------------------------------------------------------
# 5. PORTAL UI: EXCEL VALIDATION (CLEANING FIXED)
# -------------------------------------------------------------------------
st.header("3. Wet-Lab Data Validation")

# Excel Template
template = pd.DataFrame({
    'Experiment_Number': [1], 'Date': ['2026-03-14'], 'Time': ['08:00'],
    'Glucose': [10.0], 'NH4Cl': [1.25], 'PO4': [3.0], 'pH': [7.0],
    'Actual_PHA': [162.5], 'Actual_BS': [72.1], 'Actual_CDW': [1.85]
})
out = io.BytesIO()
with pd.ExcelWriter(out, engine='openpyxl') as w:
    template.to_excel(w, index=False)
st.download_button("📥 Download Excel Template", out.getvalue(), "template.xlsx")

up = st.file_uploader("Upload Lab Results", type=["xlsx", "csv"])

if up:
    try:
        df = pd.read_excel(up) if up.name.endswith('xlsx') else pd.read_csv(up)
        cols = ['Glucose', 'NH4Cl', 'PO4', 'pH', 'Actual_PHA', 'Actual_BS', 'Actual_CDW']
        
        if all(c in df.columns for c in cols):
            # --- CRITICAL FIX: FORCE NUMERIC ---
            # Isse "Glucose" jaise strings NaN ban jayenge aur drop ho jayenge
            for c in cols:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            df = df.dropna(subset=cols)
            
            if len(df) > 0:
                xt = scaler_X.transform(df[['Glucose', 'NH4Cl', 'PO4', 'pH']].values)
                pt = scaler_Y.inverse_transform(ann.predict(xt))
                
                df['Pred_PHA'], df['Pred_BS'], df['Pred_CDW'] = pt[:, 0], pt[:, 1], pt[:, 2]
                
                # Tabbed Results
                t1, t2, t3 = st.tabs(["PHA", "BS", "CDW"])
                targets = [('Actual_PHA', 'Pred_PHA', 'mg', t1), ('Actual_BS', 'Pred_BS', 'mg', t2), ('Actual_CDW', 'Pred_CDW', 'g', t3)]
                
                for act, pre, unit, tab in targets:
                    with tab:
                        # Parity Plot
                        f_p = go.Figure()
                        f_p.add_trace(go.Scatter(x=[df[act].min(), df[act].max()], y=[df[act].min(), df[act].max()], mode='lines', name='Ideal'))
                        f_p.add_trace(go.Scatter(x=df[act], y=df[pre], mode='markers', name='Lab Data'))
                        f_p.update_layout(template=layout_template, title=f"{act} Validation")
                        st.plotly_chart(f_p, use_container_width=True)
                        
                        # Stats
                        r2 = r2_score(df[act], df[pre])
                        rmse = np.sqrt(mean_squared_error(df[act], df[pre]))
                        st.write(f"**R² Score:** {r2:.4f} | **RMSE:** {rmse:.2f} {unit}")
                
                st.dataframe(df.style.format(precision=2))
            else:
                st.error("No valid numeric data found. Check your file format.")
        else:
            st.error(f"Columns missing! Need: {cols}")
    except Exception as e:
        st.error(f"Error processing file: {e}")
