# 00_generate_theft.py
import pandas as pd
import numpy as np
import os

os.makedirs('data/processed', exist_ok=True)

df = pd.read_csv('data/raw/iiot_smart_grid_dataset.csv')

# Normalise column names to snake_case
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(r'[\s\-]+', '_', regex=True)
)

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

df = df.sort_values('timestamp').reset_index(drop=True)

df['theft_flag'] = 0

# STRATIFIED BLOCK INJECTION
np.random.seed(42)
total_rows = len(df)
target_theft_rows = int(0.05 * total_rows)

# Create ~30 distinct "theft events"
num_events = 30
event_duration = target_theft_rows // num_events

theft_indices = set()
chunk_size = total_rows // num_events

for i in range(num_events):
    chunk_start = i * chunk_size
    start_idx = np.random.randint(chunk_start, chunk_start + chunk_size - event_duration)
    theft_indices.update(range(start_idx, start_idx + event_duration))

theft_indices = list(theft_indices)

# Instead of a flat 0.20, we use a random distribution between 0.10 and 0.45
# This simulates varying degrees of meter bypass/tampering
noise_consumption = np.random.uniform(0.10, 0.45, size=len(theft_indices))
noise_pf = np.random.uniform(0.10, 0.45, size=len(theft_indices))

df.loc[theft_indices, 'consumption_kwh'] *= noise_consumption
df.loc[theft_indices, 'power_factor'] *= noise_pf
df.loc[theft_indices, 'theft_flag'] = 1

print(f"\nTheft Injection Summary")
print(f"Total records  : {len(df):,}")
print(f"Theft records  : {df['theft_flag'].sum():,}  ({df['theft_flag'].mean()*100:.1f}%)")
print(f"Normal records : {(df['theft_flag'] == 0).sum():,}")
print(f"Post-injection avg consumption : {df['consumption_kwh'].mean():.4f} kWh")

df.to_csv('data/processed/smart_grid_labelled.csv', index=False)
print("\nSaved -> data/processed/smart_grid_labelled.csv")