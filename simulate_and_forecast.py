import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import os

# === SETTINGS ===
input_file = "data/Rajya_Sabha_Session_237_AU897_1.2.csv"
simulated_file = "data/simulated_energy_consumption.csv"
forecast_dir = "data/forecasts"
wards_per_state = 10

os.makedirs(forecast_dir, exist_ok=True)

# === STEP 1: Load and Prepare Input ===
df_states = pd.read_csv(input_file)
df_states.columns = ['state', 'annual_consumption_kwh']

months = pd.date_range(start='2014-04-01', end='2015-03-01', freq='MS').strftime('%Y-%m').tolist()
monthly_factors = np.array([1.05, 1.08, 1.12, 1.15, 1.10, 1.00, 0.95, 0.90, 0.92, 0.98, 1.00, 1.05])
monthly_factors /= monthly_factors.sum()

# === STEP 2: Simulate Monthly Ward-Level Data ===
simulated_data = []

for _, row in df_states.iterrows():
    state = row['state']
    annual_total = row['annual_consumption_kwh']
    ward_shares = np.random.dirichlet(np.ones(wards_per_state), size=1).flatten()

    for i in range(wards_per_state):
        ward_id = f"W{i+1}"
        ward_annual = annual_total * ward_shares[i]

        for m_idx, month in enumerate(months):
            # Add noise: ±5% variation
            noise = np.random.normal(loc=1.0, scale=0.05)  # mean=1.0, stddev=5%
            monthly_consumption = ward_annual * monthly_factors[m_idx] * noise

            simulated_data.append({
                'state': state,
                'ward_id': ward_id,
                'month': month,
                'consumption_kwh': round(monthly_consumption, 2)
            })

df_simulated = pd.DataFrame(simulated_data)
df_simulated.to_csv(simulated_file, index=False)
print(f"✅ Simulated data saved: {simulated_file}")

# === STEP 3: Forecast for All Wards ===

def forecast_ward(df, state_name, ward_id):
    df_ward = df[(df['state'] == state_name) & (df['ward_id'] == ward_id)][['month', 'consumption_kwh']].copy()
    df_ward['month'] = pd.to_datetime(df_ward['month'])
    df_ward.columns = ['ds', 'y']

    model = Prophet()
    model.fit(df_ward)

    future = model.make_future_dataframe(periods=12, freq='M')
    forecast = model.predict(future)

    forecast_out = forecast[['ds', 'yhat']].copy()
    forecast_out['state'] = state_name
    forecast_out['ward_id'] = ward_id
    forecast_out.rename(columns={'ds': 'month', 'yhat': 'predicted_consumption'}, inplace=True)

    out_file = f"{forecast_dir}/forecast_{state_name.replace(' ', '_')}_{ward_id}.csv"
    forecast_out.to_csv(out_file, index=False)
    print(f"📈 Saved forecast: {out_file}")
    return forecast_out

# Loop through all state-ward combinations
unique_combos = df_simulated[['state', 'ward_id']].drop_duplicates()

all_forecasts = []

for _, row in unique_combos.iterrows():
    state, ward_id = row['state'], row['ward_id']
    forecast_df = forecast_ward(df_simulated, state, ward_id)
    all_forecasts.append(forecast_df)

# Combine all forecasts for dashboard use
combined_forecasts = pd.concat(all_forecasts)
combined_forecasts.to_csv("data/all_forecasts.csv", index=False)
print("✅ Combined forecast saved: data/all_forecasts.csv")
