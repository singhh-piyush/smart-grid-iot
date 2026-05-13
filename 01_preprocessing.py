import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
import joblib
import os

os.makedirs('models', exist_ok=True)
os.makedirs('outputs/figures', exist_ok=True)

df = pd.read_csv('data/processed/smart_grid_labelled.csv', parse_dates=['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

# Encode weather condition
if 'weather_condition' in df.columns:
    weather_col = 'weather_condition'
elif 'Weather_Condition' in df.columns:
    weather_col = 'Weather_Condition'
else:
    raise KeyError("weather_condition column not found in dataset")

df[weather_col] = df[weather_col].fillna('Unknown').astype(str)
weather_encoder = LabelEncoder()
df['weather_encoded'] = weather_encoder.fit_transform(df[weather_col])
joblib.dump(weather_encoder, 'models/weather_encoder.pkl')

# Handle missing values via KNN imputation on sensor columns
print(f"Missing before imputation:\n{df.isnull().sum()}")
numeric_cols = ['consumption_kwh', 'voltage', 'current', 'power_factor']
imputer = KNNImputer(n_neighbors=5)
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
joblib.dump(imputer, 'models/knn_imputer.pkl')

# FIX: only cap sensor columns, NOT consumption_kwh.
# Consumption is both the regression target and the theft signal -
# capping it would clip the 80% dip that theft rows rely on.
SENSOR_COLS = ['voltage', 'current', 'power_factor']
for col in SENSOR_COLS:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    df[col] = df[col].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)
print("Outlier capping done (sensor columns only).")

# Calendar features
df['hour']        = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['month']       = df['timestamp'].dt.month
df['is_weekend']  = (df['day_of_week'] >= 5).astype(int)
df['is_peak_hour'] = df['hour'].isin([7, 8, 9, 17, 18, 19, 20]).astype(int)

# Lag features - give the tree model memory of past consumption
LAG_HOURS = [1, 2, 3, 6, 12, 48, 168]
for lag in LAG_HOURS:
    df[f'lag_{lag}h'] = df['consumption_kwh'].shift(lag)

# FIX: previous_day_consumption is already in the dataset from the raw CSV.
# Rename it to lag_24h to keep naming consistent with other lag features.
if 'previous_day_consumption' in df.columns:
    df = df.rename(columns={'previous_day_consumption': 'lag_24h'})
else:
    df['lag_24h'] = df['consumption_kwh'].shift(24)

# Rolling statistics - shift(1) first to avoid leaking current-row data
shifted = df['consumption_kwh'].shift(1)
df['rolling_mean_24h'] = shifted.rolling(24, min_periods=1).mean()
df['rolling_std_24h']  = shifted.rolling(24, min_periods=1).std()
df['rolling_mean_7d']  = shifted.rolling(168, min_periods=1).mean()

# FIX: delta_1h was computed as diff(1) which includes the current row.
# The rate of change must be between t-2 and t-1 (both past), not t-1 and t.
df['delta_1h'] = shifted.diff(1)

# Drop rows with NaN from lagging
before = len(df)
df.dropna(inplace=True)
print(f"Rows dropped due to lag NaN: {before - len(df)}  ({before} -> {len(df)})")

# FIX: removed MinMaxScaler entirely. XGBoost is scale-invariant so scaling
# sensor features adds no value and risks train/inference mismatch.

df.to_csv('data/processed/smart_grid_features.csv', index=False)
print(f"\nFinal dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print("\nSaved -> data/processed/smart_grid_features.csv")