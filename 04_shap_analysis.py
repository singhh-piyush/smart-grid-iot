# 04_shap_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('data/processed/smart_grid_features.csv', parse_dates=['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

FORECAST_FEATURES = [
    'lag_1h', 'lag_2h', 'lag_3h', 'lag_6h', 'lag_12h', 'lag_24h', 'lag_48h', 'lag_168h',
    'rolling_mean_24h', 'rolling_std_24h', 'rolling_mean_7d',
    'delta_1h',
    'voltage', 'current', 'power_factor', 'weather_encoded',
    'hour', 'day_of_week', 'month', 'is_weekend', 'is_peak_hour'
]

THEFT_FEATURES = ['consumption_kwh'] + FORECAST_FEATURES

classifier = joblib.load('models/xgb_theft_classifier.pkl')
forecaster = joblib.load('models/xgb_forecaster.pkl')

split_idx = int(len(df) * 0.80)
test_df = df.iloc[split_idx:]

sample = test_df.sample(n=min(500, len(test_df)), random_state=42)
X_sample = sample[THEFT_FEATURES]

# Theft classifier SHAP
print("Computing SHAP values for theft classifier...")
clf_explainer = shap.TreeExplainer(classifier)
clf_shap_values = clf_explainer.shap_values(X_sample)

plt.figure(figsize=(10, 7))
shap.summary_plot(clf_shap_values, X_sample, show=False, plot_size=(10, 7))
plt.title('Theft Classifier - SHAP Beeswarm', fontsize=13)
plt.tight_layout()
plt.savefig('outputs/figures/06_shap_theft_beeswarm.png', dpi=150, bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 7))
shap.summary_plot(clf_shap_values, X_sample, plot_type='bar', show=False, plot_size=(10, 7))
plt.title('Theft Classifier - Mean |SHAP| Feature Importance', fontsize=13)
plt.tight_layout()
plt.savefig('outputs/figures/07_shap_theft_bar.png', dpi=150, bbox_inches='tight')
plt.close()

# Waterfall for one theft prediction (find a row the model flagged)
probs = classifier.predict_proba(X_sample)[:, 1]
theft_idx = np.argmax(probs)
plt.figure(figsize=(10, 7))
shap.plots.waterfall(
    shap.Explanation(
        values=clf_shap_values[theft_idx],
        base_values=clf_explainer.expected_value,
        data=X_sample.iloc[theft_idx],
        feature_names=THEFT_FEATURES
    ),
    show=False
)
plt.title('Theft Classifier - Waterfall (highest-probability sample)', fontsize=11)
plt.tight_layout()
plt.savefig('outputs/figures/08_shap_theft_waterfall.png', dpi=150, bbox_inches='tight')
plt.close()

# Forecaster SHAP (use non-theft rows only, matching training)
print("Computing SHAP values for forecaster...")
clean_test = df[df['theft_flag'] == 0]
clean_test = clean_test[clean_test.index >= df.index[split_idx]]
clean_sample = clean_test.sample(n=min(500, len(clean_test)), random_state=42)
X_clean = clean_sample[FORECAST_FEATURES]

reg_explainer = shap.TreeExplainer(forecaster)
reg_shap_values = reg_explainer.shap_values(X_clean)

plt.figure(figsize=(10, 7))
shap.summary_plot(reg_shap_values, X_clean, show=False, plot_size=(10, 7))
plt.title('Forecaster - SHAP Beeswarm', fontsize=13)
plt.tight_layout()
plt.savefig('outputs/figures/09_shap_forecast_beeswarm.png', dpi=150, bbox_inches='tight')
plt.close()

print("SHAP analysis complete. Plots saved.")
