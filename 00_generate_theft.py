# 00_generate_theft.py
import pandas as pd
import numpy as np
import os

os.makedirs('data/processed', exist_ok=True)

# FIX: actual filename is iiot_smart_grid_dataset.csv, not smart_grid_dataset.csv
df = pd.read_csv('data/raw/iiot_smart_grid_dataset.csv')

# Normalise column names to snake_case
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(r'[\s\-]+', '_', regex=True)
)

print("Columns found:", df.columns.tolist())
print("Shape:", df.shape)
print(df.head(3))

# Map raw column names to the standard names used throughout the pipeline.
# Only rename columns that actually need shorter or cleaner names.
COLUMN_MAP = {
    'timestamp':                    'timestamp',
    'power_consumption_kwh':        'consumption_kwh',
    'voltage_v':                    'voltage',
    'current_a':                    'current',
    'power_factor':                 'power_factor',
    'grid_frequency_hz':            'grid_frequency',
    'reactive_power_kvar':          'reactive_power',
    'active_power_kw':              'active_power',
    'demand_response_event':        'demand_response',
    'temperature_c':                'temperature',
    'humidity_%':                    'humidity',
    'weather_condition':            'weather_condition',
    'solar_power_generation_kw':    'solar_generation',
    'wind_power_generation_kw':     'wind_generation',
    'previous_day_consumption_kwh': 'previous_day_consumption',
    'peak_load_hour':               'peak_load_hour',
    'energy_source_type':           'energy_source_type',
    'user_type':                    'user_type',
    'normalized_consumption':       'normalized_consumption',
    'energy_efficiency_score':      'energy_efficiency_score',
}
df = df.rename(columns=COLUMN_MAP)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# FIX: removed meter_id sort - this dataset has no Meter_ID column
df = df.sort_values('timestamp').reset_index(drop=True)

df['theft_flag'] = 0

# Randomly inject theft into 5% of rows.
# A tampered meter under-reports consumption by ~80%.
np.random.seed(42)
theft_indices = np.random.choice(
    df.index,
    size=int(0.05 * len(df)),
    replace=False
)

df.loc[theft_indices, 'consumption_kwh'] *= 0.20
df.loc[theft_indices, 'theft_flag'] = 1

print(f"\nTheft Injection Summary")
print(f"Total records  : {len(df):,}")
print(f"Theft records  : {df['theft_flag'].sum():,}  ({df['theft_flag'].mean()*100:.1f}%)")
print(f"Normal records : {(df['theft_flag'] == 0).sum():,}")
print(f"Post-injection avg consumption : {df['consumption_kwh'].mean():.4f} kWh")

df.to_csv('data/processed/smart_grid_labelled.csv', index=False)
print("\nSaved -> data/processed/smart_grid_labelled.csv")