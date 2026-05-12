# 02_forecasting.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('data/processed/smart_grid_features.csv', parse_dates=['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

FEATURE_COLS = [
    'lag_1h', 'lag_2h', 'lag_3h', 'lag_6h', 'lag_12h', 'lag_24h', 'lag_48h', 'lag_168h',
    'rolling_mean_24h', 'rolling_std_24h', 'rolling_mean_7d',
    'delta_1h',
    'voltage', 'current', 'power_factor', 'weather_encoded',
    'hour', 'day_of_week', 'month', 'is_weekend', 'is_peak_hour'
]
TARGET = 'consumption_kwh'

# Remove theft rows - we want to forecast REAL consumption patterns
df_clean = df[df['theft_flag'] == 0].copy().reset_index(drop=True)

X = df_clean[FEATURE_COLS]
y = df_clean[TARGET]

# Chronological train/test split (no shuffle for time series)
split_idx = int(len(df_clean) * 0.80)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
dates_test = df_clean['timestamp'].iloc[split_idx:]

# FIX: carve a validation set from training data for early stopping.
# Using the test set as eval_set leaks information through stopping decisions.
val_split = int(len(X_train) * 0.90)
X_tr, X_val = X_train.iloc[:val_split], X_train.iloc[val_split:]
y_tr, y_val = y_train.iloc[:val_split], y_train.iloc[val_split:]

print(f"Training samples : {len(X_tr):,}")
print(f"Validation samples: {len(X_val):,}")
print(f"Test samples     : {len(X_test):,}")

xgb_reg = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.80,
    min_child_weight=5,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    tree_method='hist',
    n_jobs=-1,
    early_stopping_rounds=30,
    eval_metric='rmse'
)

# FIX: eval_set uses validation split from training data, not test data
xgb_reg.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    verbose=50
)
joblib.dump(xgb_reg, 'models/xgb_forecaster.pkl')

y_pred = xgb_reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100

print("\nForecasting Results")
print(f"RMSE : {rmse:.4f} kWh")
print(f"MAE  : {mae:.4f} kWh")
print(f"R2   : {r2:.4f}")
print(f"MAPE : {mape:.2f}%")

# FIX: CV runs on training data only, not the full dataset.
# Using full X would leak test data into CV folds.
tscv = TimeSeriesSplit(n_splits=5)
cv_rmse = []
for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
    m = xgb.XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        random_state=42, tree_method='hist', n_jobs=-1
    )
    m.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
    p = m.predict(X_train.iloc[val_idx])
    fold_rmse = np.sqrt(mean_squared_error(y_train.iloc[val_idx], p))
    cv_rmse.append(fold_rmse)
    print(f"  Fold {fold+1}: RMSE = {fold_rmse:.4f}")

print(f"\n5-Fold CV RMSE: {np.mean(cv_rmse):.4f} +/- {np.std(cv_rmse):.4f}")

# Plot 1: Actual vs Predicted (last 200 samples)
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(range(200), y_test.values[-200:], label='Actual', color='steelblue', linewidth=1.5)
ax.plot(range(200), y_pred[-200:], label='XGBoost Predicted',
        color='tomato', linewidth=1.5, linestyle='--')
ax.set_title('XGBoost Demand Forecasting - Actual vs Predicted (last 200 samples)', fontsize=13)
ax.set_xlabel('Sample Index')
ax.set_ylabel('Consumption (kWh)')
ax.legend()
plt.tight_layout()
plt.savefig('outputs/figures/01_forecast_actual_vs_pred.png', dpi=150)
plt.close()

# Plot 2: Residuals distribution
residuals = y_test.values - y_pred
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.scatter(y_pred, residuals, alpha=0.3, color='steelblue', s=10)
ax1.axhline(0, color='red', linestyle='--')
ax1.set_xlabel('Predicted')
ax1.set_ylabel('Residual')
ax1.set_title('Residual Plot')

ax2.hist(residuals, bins=50, color='steelblue', edgecolor='white')
ax2.axvline(0, color='red', linestyle='--')
ax2.set_xlabel('Residual Value')
ax2.set_ylabel('Frequency')
ax2.set_title('Residual Distribution')
plt.tight_layout()
plt.savefig('outputs/figures/02_forecast_residuals.png', dpi=150)
plt.close()

# Plot 3: Feature importance
feat_imp = pd.Series(xgb_reg.feature_importances_, index=FEATURE_COLS)
feat_imp = feat_imp.sort_values(ascending=True).tail(15)

fig, ax = plt.subplots(figsize=(10, 6))
feat_imp.plot(kind='barh', ax=ax, color='steelblue')
ax.set_title('XGBoost Feature Importance - Demand Forecasting', fontsize=13)
ax.set_xlabel('Importance Score')
plt.tight_layout()
plt.savefig('outputs/figures/03_forecast_feature_importance.png', dpi=150)
plt.close()

print("\nForecasting complete. Plots saved.")