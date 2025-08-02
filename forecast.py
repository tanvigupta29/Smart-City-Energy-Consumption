# ==========================================
# Smart City Energy Simulation + Forecasting
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import os

# === STEP 1: Load State-Level Dataset ===

INPUT_FILE = "data\Rajya_Sabha_Session_237_AU897_1.2.csv"

df_states = pd.read_csv(INPUT_FILE)
df_states.columns = ['state', 'annual_consumption_kwh']  # Rename columns

# === STEP 2: Define Simulation Settings ===

wards_per_state = 10
months = pd.date_range(start='2014-04-01', end='2015-03-01', freq='MS').strftime('%Y-%m').tolist()

# Monthly variation (seasonal weights: higher in summer)
monthly_factors = np.array([1.05, 1.08, 1.12, 1.15, 1.1, 1.0, 0.95, 0.9, 0.92, 0.98, 1.0, 1.05])
monthly_factors /= monthly_factors.sum()  # Normalize to total = 1

# === STEP 3: Simulate Monthly Ward-Level Data ===

simulated_data = []

for _, row in df_states.iterrows():
    state = row['state']
    annual_total = row['annual_consumption_kwh']

    # Divide state consumption across wards randomly
    ward_shares = np.random.dirichlet(np.ones(wards_per_state), size=1).flatten()

    for i in range(wards_per_state):
        ward_id = f"W{i+1}"
        ward_annual = annual_total * ward_shares[i]

        # Distribute over months
        for m_idx, month in enumerate(months):
            monthly_consumption = ward_annual * monthly_factors[m_idx]
            simulated_data.append({
                'state': state,
                'ward_id': ward_id,
                'month': month,
                'consumption_kwh': round(monthly_consumption, 2)
            })

# Save the simulated data
df_simulated = pd.DataFrame(simulated_data)
df_simulated.to_csv("simulated_energy_consumption.csv", index=False)
print("✅ Simulated dataset saved as 'simulated_energy_consumption.csv'.")

# === STEP 4: Forecasting for Any One Ward (Optional Loop for All) ===

# Example: Forecast for Delhi - Ward W1
forecast_output_folder = "forecasts"
os.makedirs(forecast_output_folder, exist_ok=True)

def forecast_ward(df, state_name, ward_id):
    # Filter and format for Prophet
    df_ward = df[(df['state'] == state_name) & (df['ward_id'] == ward_id)][['month', 'consumption_kwh']].copy()
    df_ward['month'] = pd.to_datetime(df_ward['month'])
    df_ward.columns = ['ds', 'y']

    # Prophet forecast
    model = Prophet()
    model.fit(df_ward)

    future = model.make_future_dataframe(periods=12, freq='M')
    forecast = model.predict(future)

    # Plot
    model.plot(forecast)
    plt.title(f"Forecast: {state_name} - {ward_id}")
    plt.xlabel("Month")
    plt.ylabel("Consumption (kWh)")
    plt.tight_layout()
    plt.show()

    # Save forecast
    forecast_out = forecast[['ds', 'yhat']].copy()
    forecast_out['state'] = state_name
    forecast_out['ward_id'] = ward_id
    out_path = f"{forecast_output_folder}/forecast_{state_name}_{ward_id}.csv"
    forecast_out.to_csv(out_path, index=False)
    print(f"📈 Forecast saved to: {out_path}")

# Example usage:
forecast_ward(df_simulated, state_name='Delhi', ward_id='W1')
