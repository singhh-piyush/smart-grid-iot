# 05_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import time

st.set_page_config(
    page_title="Smart Grid IoT Analytics",
    layout="wide"
)

@st.cache_data
def load_data():
    df = pd.read_csv('data/processed/smart_grid_features.csv', parse_dates=['timestamp'])
    return df.sort_values('timestamp').reset_index(drop=True)

@st.cache_resource
def load_models():
    forecaster = joblib.load('models/xgb_forecaster.pkl')
    classifier = joblib.load('models/xgb_theft_classifier.pkl')
    return forecaster, classifier

df = load_data()
forecaster, classifier = load_models()

FORECAST_FEATURES = [
    'lag_1h', 'lag_2h', 'lag_3h', 'lag_6h', 'lag_12h', 'lag_24h', 'lag_48h', 'lag_168h',
    'rolling_mean_24h', 'rolling_std_24h', 'rolling_mean_7d',
    'delta_1h', 'voltage', 'current', 'power_factor', 'weather_encoded',
    'hour', 'day_of_week', 'month', 'is_weekend', 'is_peak_hour'
]

THEFT_FEATURES = ['consumption_kwh'] + FORECAST_FEATURES

st.sidebar.title("Smart Grid IoT")
st.sidebar.caption("ITDA301 Group Project - 2026")
st.sidebar.divider()

page = st.sidebar.radio("Navigation", [
    "Data Overview",
    "Demand Forecasting",
    "Theft Detection",
    "SHAP Explainability",
    "Live Simulation"
])

if page == "Data Overview":
    st.title("Smart Grid IoT - Data Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records",   f"{len(df):,}")
    c2.metric("Theft Records",   f"{df['theft_flag'].sum():,}",
              f"{df['theft_flag'].mean()*100:.1f}% of readings")
    c3.metric("Avg Consumption", f"{df['consumption_kwh'].mean():.3f} kWh")
    date_range = f"{df['timestamp'].min().date()} to {df['timestamp'].max().date()}"
    c4.metric("Date Range", date_range)

    st.divider()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['consumption_kwh'],
        mode='lines', name='Consumption',
        line=dict(color='steelblue', width=1.2)
    ))
    theft_points = df[df['theft_flag'] == 1]
    if not theft_points.empty:
        fig.add_trace(go.Scatter(
            x=theft_points['timestamp'], y=theft_points['consumption_kwh'],
            mode='markers', name='Theft Label',
            marker=dict(color='red', size=7, symbol='x')
        ))
    fig.update_layout(
        title='Consumption History - Full Dataset',
        xaxis_title='Date', yaxis_title='Consumption (kWh)',
        height=380, template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Average Consumption by Hour and Day")
    df['day_name'] = df['timestamp'].dt.day_name()
    pivot = df.pivot_table(
        values='consumption_kwh', index='day_name',
        columns='hour', aggfunc='mean'
    )
    day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    pivot = pivot.reindex([d for d in day_order if d in pivot.index])
    fig_heat = px.imshow(
        pivot, color_continuous_scale='YlOrRd',
        labels=dict(x="Hour of Day", y="Day", color="kWh"),
        title="Average Energy Consumption by Hour and Day"
    )
    st.plotly_chart(fig_heat, use_container_width=True)

elif page == "Demand Forecasting":
    st.title("Demand Forecasting - XGBoost Regression")

    clean_data = df[df['theft_flag'] == 0].reset_index(drop=True)
    clean_data = clean_data.dropna(subset=FORECAST_FEATURES)
    if len(clean_data) < 50:
        st.warning("Not enough clean data for forecasting evaluation.")
    else:
        split = int(len(clean_data) * 0.80)
        test_df  = clean_data.iloc[split:]
        X_test   = test_df[FORECAST_FEATURES]
        y_actual = test_df['consumption_kwh'].values
        y_pred   = forecaster.predict(X_test)

        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
        mae  = mean_absolute_error(y_actual, y_pred)
        r2   = r2_score(y_actual, y_pred)
        mape = np.mean(np.abs((y_actual - y_pred) / (y_actual + 1e-8))) * 100

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("RMSE",  f"{rmse:.4f} kWh")
        m2.metric("MAE",   f"{mae:.4f} kWh")
        m3.metric("R2",    f"{r2:.4f}")
        m4.metric("MAPE",  f"{mape:.2f}%")

        n_show = st.slider("Samples to display", 50, min(300, len(y_actual)), 150)
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=y_actual[-n_show:], name='Actual',
                                 line=dict(color='steelblue', width=1.5)))
        fig.add_trace(go.Scatter(y=y_pred[-n_show:], name='Predicted',
                                 line=dict(color='tomato', width=1.5, dash='dash')))
        fig.update_layout(title='Actual vs Predicted - Test Set',
                          xaxis_title='Sample', yaxis_title='Consumption (kWh)',
                          height=380, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

        feat_imp = pd.Series(
            forecaster.feature_importances_, index=FORECAST_FEATURES
        ).sort_values(ascending=False).head(12)
        fig2 = px.bar(
            x=feat_imp.values, y=feat_imp.index, orientation='h',
            title='Top 12 Features - Forecasting Model',
            labels={'x': 'Importance', 'y': 'Feature'},
            color=feat_imp.values, color_continuous_scale='Blues'
        )
        st.plotly_chart(fig2, use_container_width=True)

elif page == "Theft Detection":
    st.title("Theft Detection - XGBoost Classifier")

    # Only predict on test split to avoid showing in-sample results
    split_idx = int(len(df) * 0.80)
    test_data = df.iloc[split_idx:].dropna(subset=THEFT_FEATURES).copy()

    probs = classifier.predict_proba(test_data[THEFT_FEATURES])[:, 1]
    preds = classifier.predict(test_data[THEFT_FEATURES])
    test_data['theft_prob'] = probs
    test_data['theft_pred'] = preds

    threshold = st.slider("Detection threshold", 0.1, 0.9, 0.5, 0.05)
    test_data['flagged'] = (probs >= threshold).astype(int)

    f1, f2, f3 = st.columns(3)
    f1.metric("Records flagged", f"{test_data['flagged'].sum():,}")
    f2.metric("Flag rate",       f"{test_data['flagged'].mean()*100:.2f}%")
    f3.metric("Avg theft prob",  f"{probs.mean():.3f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=test_data['timestamp'], y=test_data['theft_prob'],
        fill='tozeroy', fillcolor='rgba(255,80,80,0.15)',
        line=dict(color='tomato', width=1.2), name='Theft Probability'
    ))
    fig.add_hline(y=threshold, line_dash='dash', line_color='red',
                  annotation_text=f"Threshold {threshold}")
    fig.update_layout(title='Theft Probability Over Time - Test Set',
                      yaxis_range=[0, 1], height=350, template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)

    st.image('outputs/figures/04_theft_classification_results.png',
             caption='Confusion Matrix / ROC Curve / Precision-Recall Curve')

elif page == "SHAP Explainability":
    st.title("SHAP Model Explainability")
    st.markdown("""
    SHAP (SHapley Additive exPlanations) shows why the model made each prediction.
    Each feature's contribution is shown as its average impact on the model output.
    """)

    tab1, tab2 = st.tabs(["Theft Detection", "Demand Forecasting"])
    with tab1:
        st.image('outputs/figures/06_shap_theft_beeswarm.png',
                 caption='Beeswarm - each dot is one record')
        st.image('outputs/figures/07_shap_theft_bar.png',
                 caption='Mean |SHAP| - average feature importance')
        st.image('outputs/figures/08_shap_theft_waterfall.png',
                 caption='Waterfall - one individual theft prediction broken down')
    with tab2:
        st.image('outputs/figures/09_shap_forecast_beeswarm.png',
                 caption='Beeswarm - feature impact on consumption forecast')

elif page == "Live Simulation":
    st.title("Live Meter Simulation")
    st.markdown("""
    Replays meter data point-by-point, running the theft detector on each new
    reading in real-time. Red markers indicate the model flagged a reading as theft.
    """)

    speed = st.select_slider("Simulation speed", ["Slow", "Normal", "Fast"], value="Normal")
    delay = {"Slow": 0.15, "Normal": 0.06, "Fast": 0.02}[speed]

    if st.button("Start Simulation", type="primary"):
        sim_data = (
            df.dropna(subset=THEFT_FEATURES)
            .sort_values('timestamp')
            .head(300)
        )

        chart_ph  = st.empty()
        status_ph = st.empty()

        consumption_log = []
        prob_log        = []
        flag_log        = []

        for i, (_, row) in enumerate(sim_data.iterrows()):
            x_row   = row[THEFT_FEATURES].values.reshape(1, -1)
            prob    = float(classifier.predict_proba(x_row)[0][1])
            flagged = prob >= 0.5

            consumption_log.append(row['consumption_kwh'])
            prob_log.append(prob)
            flag_log.append(flagged)

            if i % 3 == 0:
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=['Consumption (kWh)', 'Theft Probability'],
                    vertical_spacing=0.18
                )
                x_ax = list(range(len(consumption_log)))

                fig.add_trace(go.Scatter(
                    x=x_ax, y=consumption_log, mode='lines',
                    line=dict(color='steelblue', width=1.5),
                    name='Consumption'), row=1, col=1)

                theft_x = [j for j, f in enumerate(flag_log) if f]
                theft_y = [consumption_log[j] for j in theft_x]
                if theft_x:
                    fig.add_trace(go.Scatter(
                        x=theft_x, y=theft_y, mode='markers',
                        marker=dict(color='red', size=9, symbol='x'),
                        name='Flagged'), row=1, col=1)

                fig.add_trace(go.Scatter(
                    x=x_ax, y=prob_log, mode='lines',
                    fill='tozeroy', fillcolor='rgba(255,80,80,0.15)',
                    line=dict(color='tomato'), name='Theft Prob'),
                    row=2, col=1)
                fig.add_hline(y=0.5, line_dash='dash', line_color='red', row=2, col=1)
                fig.update_yaxes(range=[0, 1], row=2, col=1)
                fig.update_layout(height=480, template='plotly_white',
                                  showlegend=True, margin=dict(t=40))
                chart_ph.plotly_chart(fig, use_container_width=True)

            label = "FLAGGED" if flagged else "OK"
            status_ph.markdown(
                f"**Step {i+1}/{len(sim_data)}** - Current prob: {label} `{prob:.3f}`"
            )
            time.sleep(delay)

        st.success(f"Simulation complete. Flagged {sum(flag_log)} / {len(flag_log)} readings.")